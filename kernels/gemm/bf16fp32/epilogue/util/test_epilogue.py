#!/usr/bin/env python3
"""test_epilogue.py <kernel> - registry-driven correctness for a fused epilogue.

Uniform for any kernel in epilogue_testlib.EPILOGUES:
  (a) identity bit-exact : the epilogue at its identity param must equal the noop GEMM
                           output BYTE-FOR-BYTE (e.g. scale at alpha=1.0).
  (b) isolate vs noop    : epilogue output must match ref(noop_GEMM_output) across a param
                           sweep, under the project tolerance. Comparing against the noop
                           baseline (not an fp32 matmul) ISOLATES the epilogue from the
                           GEMM's own bf16 error -- that error is test_python.py's job.

Requires tk_noop AND tk_<kernel> built (in the epilogue dir). Run from the epilogue dir:
    python3 util/test_epilogue.py scale
"""
import os, sys, importlib
# make the built tk_*.so (one dir up, the epilogue root where `make` runs) importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from epilogue_testlib import EPILOGUES, make_inputs, gemm_base, init_empty, assert_sane, RTOL, ATOL, DTYPE

SHAPES = [(256, 256, 256), (256, 512, 128), (512, 256, 256), (768, 768, 256),   # small / single-block (K % 128, [C13])
          (2048, 1024, 512), (512, 1024, 1024), (2048, 2048, 2048), (8192, 8192, 8192)]


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in EPILOGUES:
        raise SystemExit(f"usage: test_epilogue.py <kernel>; registered: {list(EPILOGUES)}")
    name = sys.argv[1]
    spec = EPILOGUES[name]
    noop = importlib.import_module("tk_noop")
    fk = importlib.import_module(spec["module"])
    torch.manual_seed(0)
    allpass = True
    for (m, n, k) in SHAPES:
        A, Bt = make_inputs(m, n, k)
        Cn = gemm_base(noop, A, Bt, m, n)                  # bf16 GEMM baseline
        assert_sane("A", A); assert_sane("Bt", Bt); assert_sane("C[noop]", Cn)   # inputs + baseline must be real (no vacuous pass)

        # (a) identity == noop, bit-exact
        idf = spec["identity"]
        if idf is not None:
            args = idf(m, n, k)
            O = init_empty((m, n))
            fk.dispatch_micro(A, Bt, O, *args); torch.cuda.synchronize()
            ok = torch.equal(O, Cn); allpass &= ok
            print(f"{str((m,n,k)):<20} identity      | bit_exact={ok} | {'PASS' if ok else 'FAIL'}")

        # (b) isolate vs ref(noop_base) across the param sweep
        for args in spec["sweep"](m, n, k):
            O = init_empty((m, n))
            fk.dispatch_micro(A, Bt, O, *args); torch.cuda.synchronize()
            ref = init_empty((m, n))
            spec["ref"](Cn, ref, *args)
            ok = torch.allclose(O.float(), ref.float(), rtol=RTOL, atol=ATOL); allpass &= ok
            err = (O.float() - ref.float()).abs().max().item()
            print(f"{str((m,n,k)):<20} {spec['label'](args):<13} | max_err={err:.4g} | {'PASS' if ok else 'FAIL'}")
    print("ALL PASSED" if allpass else "SOME FAILED")
    return allpass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
