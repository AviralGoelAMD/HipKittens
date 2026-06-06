#!/usr/bin/env python3
"""test_block.py - end-to-end correctness for the K4->aux->K5 money pattern (Task 2.5).

Asserts the fused 3-kernel chain equals torch  rmsnorm(X@W0 + residual, gamma) @ W1 --
the GEMM-Residual-RMSNorm-GEMM Transformer sublayer, with the [M,N] intermediate never
round-tripping HBM. Clears Checkpoint 2c.

Run from the epilogue dir after building tk_k4, tk_aux_rms, tk_k5:
    python3 util/test_block.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from epilogue_testlib import init_randn
from block_chain import fused_rmsnorm_block, EPS

# (M, K0, N, P): M/N/P % 256 (256x256 tiling); K0 & N are the K4/K5 contraction dims -> % 128 ([C13])
SHAPES = [(256, 256, 256, 256), (512, 512, 512, 512), (512, 1024, 512, 768), (768, 256, 768, 512)]
RTOL, ATOL = 2e-2, 2e-1   # two chained bf16 GEMMs + a bf16 intermediate -> looser than a single GEMM


def reference(X, W0, residual, gamma, W1):
    h1 = X.float() @ W0.float() + residual.float()                # GEMM + residual
    rms = torch.rsqrt(h1.pow(2).mean(-1, keepdim=True) + EPS)     # per-row 1/RMS
    hn = h1 * rms * gamma.float()                                 # RMSNorm * gamma
    return hn @ W1.float()                                        # second GEMM


def run_shape(m, k0, n, p):
    X = init_randn((m, k0)); W0 = init_randn((k0, n))
    residual = init_randn((m, n)); gamma = init_randn((n,)); W1 = init_randn((n, p))
    out = fused_rmsnorm_block(X, W0, residual, gamma, W1)
    ref = reference(X, W0, residual, gamma, W1)
    err = (out.float() - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    ok = torch.allclose(out.float(), ref, rtol=RTOL, atol=ATOL)
    print(f"  {str((m,k0,n,p)):<24} max_err={err:.3g} rel={rel:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    torch.manual_seed(0)
    allpass = True
    for s in SHAPES:
        allpass &= run_shape(*s)
    print("ALL PASSED" if allpass else "SOME FAILED")
    return allpass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
