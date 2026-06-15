"""swiglu.py - SwiGLU helpers: the Path A gate_up column permutation, a load-once wrapper, and the
torch reference (natural layout). The permutation makes gate[j]/value[j] register-co-resident; it is
applied ONCE to the static weight (never the runtime activation)."""
import torch

H, BT = 128, 256   # BlockTileN/2, BLOCK_SIZE; single source: epilogue_args.cuh
DTYPE = torch.bfloat16


def gate_up_perm(d_ff):
    """Column permutation of W_gate_up [d_model, 2*d_ff] (natural: [0,d_ff)=gate, [d_ff,2d_ff)=value)
    so gate[j] and value[j] land 128 cols apart in one 256-block. Requires d_ff % 128 == 0."""
    assert d_ff % H == 0, f"d_ff={d_ff} must be a multiple of {H}"
    j = torch.arange(d_ff)
    b, c = j // H, j % H
    gate_slot, value_slot = b * BT + c, b * BT + c + H
    perm = torch.empty(2 * d_ff, dtype=torch.long)
    perm[gate_slot] = j
    perm[value_slot] = d_ff + j
    return perm


def swiglu_ref(X, W_gate_up):
    """Natural-layout fp32 reference: H = X@W_gate_up; out = silu(gate)*value -> [M, d_ff]."""
    Hh = X.float() @ W_gate_up.float()
    d_ff = W_gate_up.shape[1] // 2
    gate, value = Hh[:, :d_ff], Hh[:, d_ff:]
    return torch.nn.functional.silu(gate) * value


def make_swiglu(W_gate_up):
    """Prepare the SwiGLU projection once: permute the (static) weight's columns and transpose for
    the kernel. Returns forward(X) -> [M, d_ff]. Rebuild if W_gate_up changes (no auto-cache)."""
    import tk_swiglu
    d_ff = W_gate_up.shape[1] // 2
    Wt = W_gate_up[:, gate_up_perm(d_ff)].to(device="cuda", dtype=DTYPE).t().contiguous()  # [2*d_ff, d_model]

    def forward(X):
        X = X.to(device="cuda", dtype=DTYPE).contiguous()
        out = torch.empty(X.shape[0], d_ff, dtype=DTYPE, device="cuda")
        tk_swiglu.dispatch(X, Wt, out)
        torch.cuda.synchronize()
        return out
    return forward
