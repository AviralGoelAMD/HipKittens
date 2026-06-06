#!/usr/bin/env python3
"""test_stage2.py - committed correctness for the Stage-2 money-pattern kernels.

The Stage-2 kernels are multi-output / different-signature and do NOT fit the single-output
EPILOGUES registry (test_epilogue.py: one `O` vs `ref(D)`), so they get this dedicated harness:

  partialrms : tk_partialrms emits ONLY `partials` [N/64, M] -> sum over the N/64 groups == row
               sum-of-squares of D ([C1]). Compared against the SAME base-GEMM D to isolate the
               reduction from bf16 matmul error (tight tol).
  k4         : tk_k4 emits c=h1*gamma, save=h1, partials, with h1 = A@B + residual. Checked vs an
               fp32 ground-truth h1 (project bf16 tol, [C11]).
  k4_aux     : the real producer->consumer path -- feed K4's partials into tk_aux_rms.reduce and
               require r == 1/rms(h1). Guards the K4<->aux layout contract before the full
               K4->aux->K5 chain (Task 2.5) is wired.

Run from the epilogue dir after building tk_kernel, tk_partialrms, tk_k4, tk_aux_rms:
    python3 util/test_stage2.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from epilogue_testlib import make_inputs, init_randn, init_empty, gemm_reference, RTOL, ATOL, DTYPE

SHAPES = [(256, 256, 256), (256, 512, 128), (512, 256, 256), (768, 768, 256),  # small/single-block (K % 128, [C13])
          (2048, 1024, 512), (512, 1024, 1024)]
EPS = 1e-5
# partials are an fp32 sum over K accumulations -> looser than the bf16 output tol
SQ_RTOL, SQ_ATOL = 2e-2, 1.0


def _dummies(m, n):
    """The throwaway binding slots the positional aggregate-init needs ([C12b])."""
    return (torch.ones(1, dtype=torch.float32, device="cuda"),   # alpha
            torch.ones(m, dtype=DTYPE, device="cuda"),           # r
            torch.ones(n, dtype=DTYPE, device="cuda"))           # gamma


def test_partialrms(base, mod, m, n, k):
    A, Bt = make_inputs(m, n, k)
    D = init_empty((m, n)); base.dispatch_micro(A, Bt, D); torch.cuda.synchronize()  # SAME D the kernel squares
    groups = n // 64
    partials = torch.zeros((groups, m), dtype=torch.float32, device="cuda")
    c = init_empty((m, n))                                                            # unused C slot
    alpha, r, gamma = _dummies(m, n)
    residual = torch.zeros((m, n), dtype=DTYPE, device="cuda")
    mod.dispatch_micro(A, Bt, c, alpha, r, gamma, residual, partials); torch.cuda.synchronize()
    got = partials.sum(0)                       # [M] sum over the N/64 groups
    ref = D.float().pow(2).sum(-1)              # [M] same D -> isolates the reduction
    assert torch.isfinite(got).all() and got.abs().max() > 0, f"{(m,n,k)}: partials degenerate (~0/transposed => axis bug [C12d])"
    ok = torch.allclose(got, ref, rtol=SQ_RTOL, atol=SQ_ATOL)
    print(f"  partialrms {str((m,n,k)):<20} groups={groups:<3} max_rel={((got-ref).abs()/ref.clamp_min(1e-6)).max():.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def test_k4(mod, m, n, k):
    A, Bt = make_inputs(m, n, k)
    residual = init_randn((m, n)); gamma = init_randn((n,))
    h1 = gemm_reference(A, Bt) + residual.float()                # fp32 ground-truth h1
    c = init_empty((m, n)); save = init_empty((m, n))
    partials = torch.zeros((n // 64, m), dtype=torch.float32, device="cuda")
    alpha = torch.ones(1, dtype=torch.float32, device="cuda"); r = torch.ones(m, dtype=DTYPE, device="cuda")
    mod.dispatch_micro(A, Bt, c, alpha, r, gamma, residual, partials, save); torch.cuda.synchronize()
    save_ok = torch.allclose(save.float(), h1, rtol=RTOL, atol=ATOL)
    out_ok  = torch.allclose(c.float(), h1 * gamma.float(), rtol=RTOL, atol=ATOL)
    sq_ok   = torch.allclose(partials.sum(0), h1.pow(2).sum(-1), rtol=SQ_RTOL, atol=SQ_ATOL)
    print(f"  k4         {str((m,n,k)):<20} save={save_ok} out={out_ok} partials={sq_ok}  {'PASS' if (save_ok and out_ok and sq_ok) else 'FAIL'}")
    return save_ok and out_ok and sq_ok


def test_k4_aux(k4, aux, m, n, k):
    """Real K4 -> aux path: K4's partials -> aux -> r, require r == 1/rms(h1)."""
    A, Bt = make_inputs(m, n, k)
    residual = init_randn((m, n)); gamma = init_randn((n,))
    h1 = gemm_reference(A, Bt) + residual.float()
    c = init_empty((m, n)); save = init_empty((m, n))
    partials = torch.zeros((n // 64, m), dtype=torch.float32, device="cuda")
    alpha = torch.ones(1, dtype=torch.float32, device="cuda"); r_dummy = torch.ones(m, dtype=DTYPE, device="cuda")
    k4.dispatch_micro(A, Bt, c, alpha, r_dummy, gamma, residual, partials, save)
    r = torch.empty(m, dtype=DTYPE, device="cuda")
    aux.reduce(partials, r); torch.cuda.synchronize()
    ref = torch.rsqrt(h1.pow(2).mean(-1) + EPS)                  # 1/rms over the full N features
    ok = torch.allclose(r.float(), ref, rtol=SQ_RTOL, atol=1e-2)
    print(f"  k4->aux    {str((m,n,k)):<20} max_err={(r.float()-ref).abs().max():.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    import importlib
    torch.manual_seed(0)
    base = importlib.import_module("tk_kernel")
    prms = importlib.import_module("tk_partialrms")
    k4   = importlib.import_module("tk_k4")
    aux  = importlib.import_module("tk_aux_rms")
    allpass = True
    print("[partialrms]"); 
    for s in SHAPES: allpass &= test_partialrms(base, prms, *s)
    print("[k4]")
    for s in SHAPES: allpass &= test_k4(k4, *s)
    print("[k4 -> aux]")
    for s in SHAPES: allpass &= test_k4_aux(k4, aux, *s)
    print("ALL PASSED" if allpass else "SOME FAILED")
    return allpass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
