#!/usr/bin/env python3
"""fusion_win.py - measure the PAYOFF of epilogue fusion (fused vs unfused).

For each epilogue Kx it runs the job BOTH ways on the same inputs:
  fused   : one kernel  tk_<Kx>          (D kept in registers, epilogue applied, write O)
  unfused : base GEMM tk_kernel -> D in HBM, then the torch reference reads D, writes O
and reports: wall-clock speedup, HBM bytes saved (the D round-trip fusion skips), and a
correctness check that both produce the same O.

HBM-bytes-saved is the shape-independent TRUTH; wall-clock speedup is shape-dependent
(~1x when the GEMM is compute-bound, >1x when memory-bound). That's why we sweep shapes.

Kernels are defined ONCE in epilogue_testlib.EPILOGUES (shared with test_epilogue.py) -- the
SAME `ref` is the test's correctness oracle and this bench's unfused stage, so they can never
measure against different math.

Run from the epilogue dir (where the tk_*.so are built) on a gfx950 node:
    python3 util/fusion_win.py --kernels scale --json results_fusion.json
"""
import os, sys, argparse, importlib, json
# make the built tk_*.so (one dir up, the epilogue root where `make` runs) importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from epilogue_testlib import EPILOGUES, make_inputs, init_empty, DSIZE, RTOL, ATOL, DTYPE


def _time(fn, iters, warm):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms/iter


def run_one(kernel, shapes, iters, warm):
    base = importlib.import_module("tk_kernel")        # base GEMM, writes D to HBM
    spec = EPILOGUES[kernel]
    fk = importlib.import_module(spec["module"])       # fused kernel
    rows = []
    for (m, n, k) in shapes:
        A, Bt = make_inputs(m, n, k)
        O = init_empty((m, n)); D = init_empty((m, n))
        args = spec["args"](m, n, k)

        fused = lambda: fk.dispatch_micro(A, Bt, O, *args)
        def unfused():
            base.dispatch_micro(A, Bt, D)              # GEMM -> D materialized in HBM
            spec["ref"](D, O, *args)                   # read D back, apply epilogue, write O

        fused();   torch.cuda.synchronize(); Of = O.clone()
        unfused(); torch.cuda.synchronize(); Ou = O.clone()
        ok = torch.allclose(Of.float(), Ou.float(), rtol=RTOL, atol=ATOL)

        tf = _time(fused, iters, warm)
        tu = _time(unfused, iters, warm)
        saved = spec["hbm_passes"] * m * n * DSIZE
        rows.append({
            "kernel": kernel, "shape": [m, n, k], "correct": bool(ok),
            "fused_ms": round(tf, 4), "unfused_ms": round(tu, 4),
            "speedup": round(tu / tf, 3), "saved_MB": round(saved / 1e6, 1),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default="noop,scale", help="comma list of registered kernels")
    ap.add_argument("--shapes", default="8192x8192x8192,8192x8192x1024,8192x8192x512,4096x4096x4096",
                    help="comma list of MxNxK (M,N multiples of 256; K multiple of 64)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warm", type=int, default=50)
    ap.add_argument("--json", default="", help="write results manifest to this path")
    a = ap.parse_args()
    shapes = [tuple(int(x) for x in s.split("x")) for s in a.shapes.split(",")]
    torch.manual_seed(0)

    rows = []
    for kr in a.kernels.split(","):
        kr = kr.strip()
        if kr not in EPILOGUES:
            raise SystemExit(f"unknown kernel '{kr}'; registered: {list(EPILOGUES)}")
        rows += run_one(kr, shapes, a.iters, a.warm)

    print(f"{'kernel':<8}{'shape (m,n,k)':<20}{'fused ms':>10}{'unfused ms':>12}"
          f"{'speedup':>9}{'HBM saved':>11}{'ok':>4}")
    print("-" * 74)
    for r in rows:
        sh = "x".join(map(str, r["shape"]))
        print(f"{r['kernel']:<8}{sh:<20}{r['fused_ms']:>10.3f}{r['unfused_ms']:>12.3f}"
              f"{r['speedup']:>8.2f}x{r['saved_MB']:>8.0f}MB{('Y' if r['correct'] else 'N'):>4}")
    print("\nHBM saved = D write+read the unfused path pays and fusion skips (shape-independent).")
    print("speedup is shape-dependent: ~1x when GEMM compute-bound, >1x when memory-bound (skinny K).")
    if a.json:
        json.dump({"fusion_win": rows}, open(a.json, "w"), indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
