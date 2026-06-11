#!/usr/bin/env python3
"""fusion_win.py - measure the PAYOFF of epilogue fusion (fused vs unfused).

For each epilogue Kx it runs the job BOTH ways on the same inputs:
  fused   : one kernel  tk_<Kx>          (D kept in registers, epilogue applied, write O)
  unfused : base GEMM tk_kernel -> D in HBM, then the torch reference reads D, writes O
and reports: wall-clock speedup (COLD-cache MEDIAN), HBM bytes saved (the D round-trip
fusion skips), and a correctness check that both produce the same O.

Honest timing by construction: B1 rotates the re-read inputs (A, Bt) through a per-shape
pool sized > the GPU last-level cache, so each timed run reads them COLD from HBM (not a hot
cache); B2 takes per-iter samples and reports the MEDIAN (outlier-proof) + min. HBM-bytes-
saved is the shape-independent TRUTH; wall-clock speedup is shape-dependent (~1x compute-
bound, >1x memory-bound). That's why we sweep shapes.

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


def _pool_size(per_set_bytes, llc_bytes, budget_bytes):
    """B1: how many input sets to rotate so the pool exceeds the cache. One set already
    bigger than the cache -> N=1 (it's cold on its own). Capped by a memory budget."""
    if per_set_bytes >= llc_bytes:
        return 1
    n = llc_bytes // per_set_bytes + 1
    n = min(n, max(1, budget_bytes // per_set_bytes))
    return max(1, int(n))


def _bench(fn, iters, warm):
    """B2: per-iter CUDA-event timing -> median (headline), min, p90, mean (ms).
    `fn(i)` takes the iteration index so the caller can rotate buffers (B1)."""
    for i in range(warm): fn(i)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record(); fn(i); ends[i].record()
    torch.cuda.synchronize()
    s = sorted(starts[i].elapsed_time(ends[i]) for i in range(iters))
    n = len(s)
    return {"median": s[n // 2], "min": s[0],
            "p90": s[min(n - 1, int(0.9 * n))], "mean": sum(s) / n}


def run_one(kernel, shapes, iters, warm, llc_bytes, budget_bytes):
    base = importlib.import_module("tk_noop")          # pure-GEMM baseline (our no-op epilogue), writes D to HBM
    spec = EPILOGUES[kernel]
    fk = importlib.import_module(spec["module"])       # fused kernel
    rows = []
    for (m, n, k) in shapes:
        # B1: rotate the RE-READ inputs (A, Bt) so each timed iter reads them cold. Size the
        # pool so N*(A+Bt) exceeds the cache. D/O are output scratch (no cross-iter reuse) -> single.
        per_set = (m * k + n * k) * DSIZE
        N = _pool_size(per_set, llc_bytes, budget_bytes)
        Ap, Btp = [], []
        for _ in range(N):
            a_, bt_ = make_inputs(m, n, k); Ap.append(a_); Btp.append(bt_)
        D = init_empty((m, n)); O = init_empty((m, n))
        args = spec["args"](m, n, k)

        def fused(i, _fk=fk, _a=args):
            _fk.dispatch(Ap[i % N], Btp[i % N], O, *_a)

        def unfused(i, _b=base, _s=spec, _a=args):
            _b.dispatch(Ap[i % N], Btp[i % N], D)   # GEMM -> D materialized in HBM
            _s["ref"](D, O, *_a)                          # read D back, apply epilogue, write O

        fused(0);   torch.cuda.synchronize(); Of = O.clone()
        unfused(0); torch.cuda.synchronize(); Ou = O.clone()
        ok = torch.allclose(Of.float(), Ou.float(), rtol=RTOL, atol=ATOL)

        tf = _bench(fused, iters, warm)
        tu = _bench(unfused, iters, warm)
        saved = spec["hbm_passes"] * m * n * DSIZE
        rows.append({
            "kernel": kernel, "shape": [m, n, k], "correct": bool(ok), "pool": N,
            "fused_ms": round(tf["median"], 4), "unfused_ms": round(tu["median"], 4),
            "fused_min": round(tf["min"], 4), "unfused_min": round(tu["min"], 4),
            "speedup": round(tu["median"] / tf["median"], 3),
            "saved_MB": round(saved / 1e6, 1),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default="noop,scale", help="comma list of registered kernels")
    ap.add_argument("--shapes", default="8192x8192x8192,8192x8192x1024,8192x8192x512,4096x4096x4096",
                    help="comma list of MxNxK (M,N multiples of 256; K multiple of 64)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warm", type=int, default=50)
    ap.add_argument("--llc-mb", type=int, default=256,
                    help="B1 cold-cache threshold: rotate enough A/Bt sets to exceed this (MB). "
                         "Set >= the GPU last-level cache.")
    ap.add_argument("--budget-gb", type=int, default=24, help="cap total rotating-pool memory (GB)")
    ap.add_argument("--json", default="", help="write results manifest to this path")
    a = ap.parse_args()
    shapes = [tuple(int(x) for x in s.split("x")) for s in a.shapes.split(",")]
    llc_bytes = a.llc_mb * 1024 * 1024
    budget_bytes = a.budget_gb * 1024 * 1024 * 1024
    torch.manual_seed(0)

    rows = []
    for kr in a.kernels.split(","):
        kr = kr.strip()
        if kr not in EPILOGUES:
            raise SystemExit(f"unknown kernel '{kr}'; registered: {list(EPILOGUES)}")
        rows += run_one(kr, shapes, a.iters, a.warm, llc_bytes, budget_bytes)

    print(f"{'kernel':<8}{'shape (m,n,k)':<20}{'fused ms':>10}{'unfused ms':>12}"
          f"{'speedup':>9}{'HBM saved':>11}{'pool':>6}{'ok':>4}")
    print("-" * 80)
    for r in rows:
        sh = "x".join(map(str, r["shape"]))
        print(f"{r['kernel']:<8}{sh:<20}{r['fused_ms']:>10.3f}{r['unfused_ms']:>12.3f}"
              f"{r['speedup']:>8.2f}x{r['saved_MB']:>8.0f}MB{r['pool']:>6}{('Y' if r['correct'] else 'N'):>4}")
    print("\nfused/unfused ms = COLD-cache MEDIAN over timed iters (B1 rotating A/Bt + B2 median).")
    print("pool = # rotated A/Bt sets (1 = one set already exceeds the cache).")
    print("HBM saved = D write+read the unfused path pays and fusion skips (shape-independent).")
    print("speedup is shape-dependent: ~1x compute-bound, >1x memory-bound (skinny K).")
    if a.json:
        json.dump({"fusion_win": rows}, open(a.json, "w"), indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
