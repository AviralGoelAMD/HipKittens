#!/usr/bin/env python3
"""test_stage2.py - committed correctness for the residual + RMSNorm-partials kernels.

These kernels are multi-output / different-signature and do NOT fit the single-output EPILOGUES
registry (test_epilogue.py: one `O` vs `ref(D)`), so they get this dedicated harness:

  partialrms          : tk_partialrms emits ONLY `partials` [N/REG_BLOCK_N, M] -> sum over the
                        groups == row sum-of-squares of D. Compared against the SAME base-GEMM D to
                        isolate the reduction from bf16 matmul error (tight tol).
  residual_rms        : tk_residual_rms emits c=h1*gamma, save=h1, partials, h1 = A@B + residual.
                        Checked against an fp32 ground-truth h1 (project bf16 tol).
  residual_rms -> aux : feed the partials into tk_aux_rms.reduce and require r == 1/rms(h1). Guards
                        the residual_rms<->aux layout contract before the full chain is wired.

Run from the epilogue dir after building tk_kernel, tk_partialrms, tk_residual_rms, tk_aux_rms:
    python3 util/test_stage2.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from epilogue_testlib import make_inputs, init_randn, init_empty, gemm_reference, RTOL, ATOL, DTYPE

SHAPES = [(256, 256, 256), (256, 512, 128), (512, 256, 256), (768, 768, 256),  # small/single-block (K % 128)
          (2048, 1024, 512), (512, 1024, 1024)]
EPS = 1e-5
# partials are an fp32 sum over K accumulations -> looser than the bf16 output tol
SQ_RTOL, SQ_ATOL = 2e-2, 1.0




def test_partialrms(base, mod, m, n, k):
    A, Bt = make_inputs(m, n, k)
    D = init_empty((m, n)); base.dispatch(A, Bt, D); torch.cuda.synchronize()  # SAME D the kernel squares
    groups = n // 64
    partials = torch.zeros((groups, m), dtype=torch.float32, device="cuda")
    c = init_empty((m, n))                                                            # unused C slot
    mod.dispatch(A, Bt, c, partials); torch.cuda.synchronize()
    got = partials.sum(0)                       # [M] sum over the N/64 groups
    ref = D.float().pow(2).sum(-1)              # [M] same D -> isolates the reduction
    assert torch.isfinite(got).all() and got.abs().max() > 0, f"{(m,n,k)}: partials degenerate (~0 or transposed => axis bug)"
    ok = torch.allclose(got, ref, rtol=SQ_RTOL, atol=SQ_ATOL)
    print(f"  partialrms {str((m,n,k)):<20} groups={groups:<3} max_rel={((got-ref).abs()/ref.clamp_min(1e-6)).max():.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def test_residual_rms(mod, m, n, k):
    A, Bt = make_inputs(m, n, k)
    residual = init_randn((m, n)); gamma = init_randn((n,))
    h1 = gemm_reference(A, Bt) + residual.float()                # fp32 ground-truth h1
    c = init_empty((m, n)); save = init_empty((m, n))
    partials = torch.zeros((n // 64, m), dtype=torch.float32, device="cuda")
    mod.dispatch(A, Bt, c, residual, gamma, partials, save); torch.cuda.synchronize()
    save_ok = torch.allclose(save.float(), h1, rtol=RTOL, atol=ATOL)
    out_ok  = torch.allclose(c.float(), h1 * gamma.float(), rtol=RTOL, atol=ATOL)
    sq_ok   = torch.allclose(partials.sum(0), h1.pow(2).sum(-1), rtol=SQ_RTOL, atol=SQ_ATOL)
    print(f"  resid_rms  {str((m,n,k)):<20} save={save_ok} out={out_ok} partials={sq_ok}  {'PASS' if (save_ok and out_ok and sq_ok) else 'FAIL'}")
    return save_ok and out_ok and sq_ok


def test_residual_rms_aux(resid, aux, m, n, k):
    """Real residual_rms -> aux path: partials -> aux -> r, require r == 1/rms(h1)."""
    A, Bt = make_inputs(m, n, k)
    residual = init_randn((m, n)); gamma = init_randn((n,))
    h1 = gemm_reference(A, Bt) + residual.float()
    c = init_empty((m, n)); save = init_empty((m, n))
    partials = torch.zeros((n // 64, m), dtype=torch.float32, device="cuda")
    resid.dispatch(A, Bt, c, residual, gamma, partials, save)
    r = torch.empty(m, dtype=DTYPE, device="cuda")
    aux.reduce(partials, r); torch.cuda.synchronize()
    ref = torch.rsqrt(h1.pow(2).mean(-1) + EPS)                  # 1/rms over the full N features
    ok = torch.allclose(r.float(), ref, rtol=SQ_RTOL, atol=1e-2)
    print(f"  resid->aux {str((m,n,k)):<20} max_err={(r.float()-ref).abs().max():.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    import importlib
    torch.manual_seed(0)
    base = importlib.import_module("tk_noop")
    prms = importlib.import_module("tk_partialrms")
    resid_rms = importlib.import_module("tk_residual_rms")
    aux  = importlib.import_module("tk_aux_rms")
    allpass = True
    print("[partialrms]"); 
    for s in SHAPES: allpass &= test_partialrms(base, prms, *s)
    print("[residual_rms]")
    for s in SHAPES: allpass &= test_residual_rms(resid_rms, *s)
    print("[residual_rms -> aux]")
    for s in SHAPES: allpass &= test_residual_rms_aux(resid_rms, aux, *s)
    print("ALL PASSED" if allpass else "SOME FAILED")
    return allpass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
