"""block_chain.py - the fused GEMM-Residual-RMSNorm-GEMM Transformer sublayer.

Chains three kernels so the [M,N] intermediate never round-trips HBM:

    out = rmsnorm(X @ W0 + residual, gamma) @ W1

  tk_residual_rms_partials : h1 = X@W0 + residual ;  c = h1*gamma ;  partials = per-(group,row) Sigma(h1^2)
  tk_rms_reduce    : partials             -> r = 1/rms(h1)   (per row; the cross-group combine)
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
import tk_residual_rms_partials, tk_rms_reduce, tk_rmsnorm_scale

DTYPE = torch.bfloat16
EPS = 1e-5   # single source: RMS_EPS in epilogue_args.cuh (the aux kernel's rsqrt(.. + eps)) -- keep in sync
REG_BLOCK_N = 64   # single source: REG_BLOCK_N in epilogue_args.cuh (partials group count = N // REG_BLOCK_N; backstopped by the binding group-count check)


def make_fused_rmsnorm_block(W0, W1):
    """Prepare the sublayer once: transpose the (static) weights a SINGLE time and hold them.
    Returns forward(X, residual, gamma) -> out[M,P]. If the weights change (e.g. after an
    optimizer step) rebuild via make_fused_rmsnorm_block(...) -- the transpose is NOT auto-cached,
    so an in-place weight update can never go stale silently.
    W0[K0,N], W1[N,P] ; X[M,K0], residual[M,N], gamma[N]. M,N,P % 256 ; K0,N % 128."""
    W0t = W0.t().contiguous()                          # base GEMM takes B transposed; weights static -> once
    W1t = W1.t().contiguous()
    N, P = W0.shape[1], W1.shape[1]
    gamma_ones = torch.ones(P, dtype=DTYPE, device="cuda")   # 2nd GEMM: gamma already folded into c -> ones

    def forward(X, residual, gamma):
        M = X.shape[0]
        c = torch.empty((M, N), dtype=DTYPE, device="cuda")
        save = torch.empty((M, N), dtype=DTYPE, device="cuda")        # h1 snapshot (unused in fwd; for bwd)
        partials = torch.empty((N // REG_BLOCK_N, M), dtype=torch.float32, device="cuda")  # kernel overwrites every (group,row)
        tk_residual_rms_partials.dispatch(X, W0t, c, residual, gamma, partials, save)
        r = torch.empty(M, dtype=DTYPE, device="cuda")
        tk_rms_reduce.reduce(partials, r)
        out = torch.empty((M, P), dtype=DTYPE, device="cuda")
        tk_rmsnorm_scale.dispatch(c, W1t, out, r, gamma_ones)         # r post-applied; gamma already in c
        torch.cuda.synchronize()
        return out
    return forward


def fused_rmsnorm_block(X, W0, residual, gamma, W1):
    """One-shot functional form (transposes the weights each call) -- for tests / one-offs.
    For a hot loop or a model, build once with make_fused_rmsnorm_block(W0, W1) and reuse forward()."""
    return make_fused_rmsnorm_block(W0, W1)(X, residual, gamma)
