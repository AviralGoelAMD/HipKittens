"""RoPE (interleaved) helpers for the gemm_rope kernel -- the weight/cos_sin permutation.

rope_perm places each interleaved pair (2k, 2k+1) at block-cols (c, c+128) so the two co-reside in
one lane. Apply it ONCE to the projection weight columns AND to cos_sin.
The kernel then rotates register-locally and stores in permuted order; correct for attention because
Q*K^T is invariant to a shared feature permutation of Q and K (V stays natural).
"""
import torch
from epilogue_testlib import DTYPE

HALF_BLOCK = 128      # co-residency stride (BlockTileN/2); single source: epilogue_args.cuh
BLOCK = 256           # BLOCK_SIZE; single source: epilogue_args.cuh


def rope_perm(N, H=HALF_BLOCK, BT=BLOCK):
    """perm[new_col] = old_col: pair k's even member (old col 2k) -> block-col (k%H); its odd member
    (old col 2k+1) -> block-col (k%H + H), same 256-block. Bijection of [0,N); requires N % 256 == 0."""
    assert N % BT == 0, f"rope_perm: N={N} must be a multiple of {BT}"
    k = torch.arange(N // 2)
    b, c = k // H, k % H
    even_slot, odd_slot = b * BT + c, b * BT + c + H   # even (cos) / odd (sin) block-cols
    perm = torch.empty(N, dtype=torch.long)
    perm[even_slot] = 2 * k
    perm[odd_slot] = 2 * k + 1
    return perm


def make_cos_sin(M, N, base=10000.0, device="cuda"):
    """Interleaved [cos,sin] table, shape (M,N): cos_sin[m,2k]=cos(m*theta_k), [m,2k+1]=sin(m*theta_k)."""
    k = torch.arange(N // 2, device=device)
    theta = base ** (-2.0 * k / N)
    ang = torch.arange(M, device=device).float()[:, None] * theta[None, :]    # [M, N/2]
    cs = torch.empty(M, N, device=device)
    cs[:, 0::2] = torch.cos(ang)
    cs[:, 1::2] = torch.sin(ang)
    return cs


def rope_ref(D, cos_sin):
    """Interleaved RoPE reference (fp32): O[2k]=x*cos+y*sin, O[2k+1]=-x*sin+y*cos (CODA gemm_rope)."""
    D = D.float(); cos_sin = cos_sin.to(DTYPE).float()   # mirror the kernel's bf16 cos_sin load (faithful regardless of caller)
    x, y = D[:, 0::2], D[:, 1::2]
    cos, sin = cos_sin[:, 0::2], cos_sin[:, 1::2]
    O = torch.empty_like(D)
    O[:, 0::2] = x * cos + y * sin
    O[:, 1::2] = -x * sin + y * cos
    return O

def make_rope(W):
    """Prepare a RoPE'd Q/K projection once: permute the (static) weight columns by rope_perm and
    transpose for the kernel. Returns forward(X, cos_sin) -> RoPE(X@W), owning BOTH permutations
    (the weight columns AND cos_sin) so callers pass natural W + cos_sin. Rebuild if W changes."""
    import tk_rope
    perm = rope_perm(W.shape[1])
    Wt = W[:, perm].to(device="cuda", dtype=DTYPE).t().contiguous()   # [N, d_model]: permuted + transposed

    def forward(X, cos_sin):                                          # cos_sin natural [M, N]
        X = X.to(device="cuda", dtype=DTYPE).contiguous()
        csp = cos_sin[:, perm].to(device="cuda", dtype=DTYPE).contiguous()   # same perm as the weight
        O = torch.empty(X.shape[0], W.shape[1], dtype=DTYPE, device="cuda")
        tk_rope.dispatch(X, Wt, O, csp)
        return O
    return forward
