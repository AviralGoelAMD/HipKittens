#!/usr/bin/env python3
"""test_gemm.py [module=tk_kernel] - base GEMM correctness vs fp32 ground truth.

This is HipKittens' shipped `kernels/gemm/bf16fp32/test.py`, REUSED with its two proven
calibration bugs fixed and non-degeneracy guards added:
  - reference = fp32 ground truth `A.float()@B.float()`, NOT a bf16 `torch.matmul(A,B)` that
    carries the same ~0.4% bf16 error;
  - tolerance = `rtol=1e-2, atol=1e-1`, NOT a bare `rtol` (bare-rtol-no-atol is unsatisfiable
    for bf16 and fails the ORIGINAL unmodified kernel 7/7);
  - `assert_sane` on inputs AND output (finite, non-zero, non-constant) -> a dead kernel or a
    degenerate init can no longer pass vacuously.
Validates that "the C we get back" really is A@B -- the result test_epilogue ISOLATES against
and assumes correct. Shapes reuse HK test.py's rectangular set + small/single-block edges.

Run from the epilogue dir after building the base GEMM:
    python3 util/test_gemm.py            # tk_kernel
    python3 util/test_gemm.py tk_noop    # validate the no-op baseline too
"""
import os, sys, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from epilogue_testlib import make_inputs, init_empty, gemm_reference, assert_sane, RTOL, ATOL

SHAPES = [
    (256, 256, 256), (256, 512, 128), (512, 256, 256), (768, 768, 256),  # small / single-block (K % 128)
    (2048, 1024, 512), (4096, 8192, 2048), (8192, 2048, 4096),           # rectangular (reused from HK test.py)
    (512, 1024, 1024), (8192, 8192, 8192),                               # square
]


def main():
    mod = sys.argv[1] if len(sys.argv) > 1 else "tk_kernel"
    gemm = importlib.import_module(mod)
    torch.manual_seed(0)
    allpass = True
    print(f"GEMM correctness ({mod}) vs fp32 ground truth  (rtol={RTOL}, atol={ATOL})")
    for (m, n, k) in SHAPES:
        A, Bt = make_inputs(m, n, k)
        assert_sane("A", A); assert_sane("Bt", Bt)            # inits are non-degenerate
        C = init_empty((m, n))
        gemm.dispatch(A, Bt, C); torch.cuda.synchronize()
        assert_sane("C", C)                                    # result finite, non-zero, non-constant
        ref = gemm_reference(A, Bt)                             # fp32 ground truth
        cf = C.float()
        ok = torch.allclose(cf, ref, rtol=RTOL, atol=ATOL); allpass &= ok
        err = (cf - ref).abs().max().item()
        mean = (cf - ref).abs().mean().item()
        print(f"  {str((m,n,k)):<20} max|err|={err:<10.4g} mean={mean:<10.4g} |C|max={cf.abs().max():<9.1f} {'PASS' if ok else 'FAIL'}")
    print("ALL PASSED" if allpass else "SOME FAILED")
    sys.exit(0 if allpass else 1)


if __name__ == "__main__":
    main()
