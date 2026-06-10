"""block_chain.py - the fused GEMM-Residual-RMSNorm-GEMM Transformer sublayer (the "money pattern").

Chains the three Stage-2 kernels so the [M,N] intermediate never round-trips HBM:

    out = rmsnorm(X @ W0 + residual, gamma) @ W1

  K4   (tk_k4)      : h1 = X@W0 + residual ;  c = h1*gamma ;  partials = per-(group,row) Sigma(h1^2)
  aux  (tk_aux_rms) : partials            -> r = 1/rms(h1)   (per row, the cross-group combine, [C1])
  K5   (tk_k5)      : out = (c @ W1) * r[:,None]              (gamma already folded in by K4)

Axis reasoning (why gamma is in K4 and only r is in K5):
  rmsnorm(h1,gamma) @ W1 = (h1 * r * gamma) @ W1.
  - gamma scales h1's feature axis = the CONTRACTED dim of the 2nd GEMM, so it must be applied
    BEFORE that GEMM -> folded into K4's `c` (post-scaling K5's output by gamma[out-feature] would
    be the wrong axis).
  - r is per-row (the M axis); it is constant across a row and therefore commutes through @W1:
    ((c) @ W1) * r[:,None] == (c * r[:,None]) @ W1. So K5 may post-scale its output by r.
  Hence K5 runs with gamma = ones (a no-op gamma slot) and the real r.
"""
import torch
import tk_k4, tk_aux_rms, tk_k5

DTYPE = torch.bfloat16
EPS = 1e-5   # matches the aux kernel's rsqrt(.. + eps) and layer_norm.py


def fused_rmsnorm_block(X, W0, residual, gamma, W1):
    """out = rmsnorm(X@W0 + residual, gamma) @ W1, fully fused. All tensors bf16, CUDA, contiguous.
    X[M,K0], W0[K0,N], residual[M,N], gamma[N], W1[N,P] -> out[M,P].
    Shape constraints: M,N,P % 256 (256x256 tiling); K0,N % 128 (the GEMM contraction dims, [C13])."""
    M, K0 = X.shape
    N = W0.shape[1]
    P = W1.shape[1]
    W0t = W0.t().contiguous()                          # base GEMM takes B transposed (A @ Bt.t() == A@B)
    W1t = W1.t().contiguous()

    # --- K4: h1 = X@W0 + residual ; c = h1*gamma ; partials = Sigma(h1^2) split over N/64 groups ---
    c = torch.empty((M, N), dtype=DTYPE, device="cuda")
    save = torch.empty((M, N), dtype=DTYPE, device="cuda")       # h1 snapshot (unused in fwd; for bwd)
    partials = torch.zeros((N // 64, M), dtype=torch.float32, device="cuda")
    tk_k4.dispatch_micro(X, W0t, c, residual, gamma, partials, save)

    # --- aux: partials -> r = 1/rms(h1) per row ---
    r = torch.empty(M, dtype=DTYPE, device="cuda")
    tk_aux_rms.reduce(partials, r)

    # --- K5: out = (c @ W1) * r[:,None] ; gamma already applied in K4 -> ones here ---
    out = torch.empty((M, P), dtype=DTYPE, device="cuda")
    gamma_ones = torch.ones(P, dtype=DTYPE, device="cuda")
    tk_k5.dispatch_micro(c, W1t, out, r, gamma_ones)

    torch.cuda.synchronize()
    return out
