"""block_chain.py - the fused GEMM-Residual-RMSNorm-GEMM Transformer sublayer.

Chains three kernels so the [M,N] intermediate never round-trips HBM:

    out = rmsnorm(X @ W0 + residual, gamma) @ W1

  tk_residual_rms  : h1 = X@W0 + residual ;  c = h1*gamma ;  partials = per-(group,row) Sigma(h1^2)
  tk_aux_rms       : partials             -> r = 1/rms(h1)   (per row; the cross-group combine)
  tk_rmsnorm_scale : out = (c @ W1) * r[:,None]             (gamma already folded into c above)

Axis reasoning (why gamma folds into the first GEMM and only r post-applies in the second):
  rmsnorm(h1,gamma) @ W1 = (h1 * r * gamma) @ W1.
  - gamma scales h1's feature axis = the CONTRACTED dim of the second GEMM, so it must be applied
    BEFORE that GEMM -> folded into c (post-scaling the output by gamma[out-feature] would be the
    wrong axis).
  - r is per-row (the M axis); constant across a row, so it commutes through @W1:
    ((c) @ W1) * r[:,None] == (c * r[:,None]) @ W1. So r may post-scale the second GEMM's output.
  Hence the second GEMM runs with gamma = ones and the real r.
"""
import torch
import tk_residual_rms, tk_aux_rms, tk_rmsnorm_scale

DTYPE = torch.bfloat16
EPS = 1e-5   # matches the aux kernel's rsqrt(.. + eps) and layer_norm.py


def fused_rmsnorm_block(X, W0, residual, gamma, W1):
    """out = rmsnorm(X@W0 + residual, gamma) @ W1, fully fused. All tensors bf16, CUDA, contiguous.
    X[M,K0], W0[K0,N], residual[M,N], gamma[N], W1[N,P] -> out[M,P].
    Shape constraints: M,N,P % 256 (256x256 tiling); K0,N % 128 (the GEMM contraction dims)."""
    M, K0 = X.shape
    N = W0.shape[1]
    P = W1.shape[1]
    W0t = W0.t().contiguous()                          # base GEMM takes B transposed (A @ Bt.t() == A@B)
    W1t = W1.t().contiguous()

    # --- residual_rms: h1 = X@W0 + residual ; c = h1*gamma ; partials = Sigma(h1^2) over column groups ---
    c = torch.empty((M, N), dtype=DTYPE, device="cuda")
    save = torch.empty((M, N), dtype=DTYPE, device="cuda")       # h1 snapshot (unused in fwd; for bwd)
    partials = torch.zeros((N // 64, M), dtype=torch.float32, device="cuda")  # 64 = REG_BLOCK_N (column-group width)
    tk_residual_rms.dispatch(X, W0t, c, residual, gamma, partials, save)

    # --- aux: partials -> r = 1/rms(h1) per row ---
    r = torch.empty(M, dtype=DTYPE, device="cuda")
    tk_aux_rms.reduce(partials, r)

    # --- rmsnorm_scale: out = (c @ W1) * r[:,None] ; gamma already folded into c above -> ones here ---
    out = torch.empty((M, P), dtype=DTYPE, device="cuda")
    gamma_ones = torch.ones(P, dtype=DTYPE, device="cuda")
    tk_rmsnorm_scale.dispatch(c, W1t, out, r, gamma_ones)

    torch.cuda.synchronize()
    return out
