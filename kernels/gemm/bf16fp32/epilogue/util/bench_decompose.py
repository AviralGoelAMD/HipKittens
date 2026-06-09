#!/usr/bin/env python3
"""bench_decompose.py [--kernels ...] [--shapes ...] - B3: split the wall-clock into GEMM vs
epilogue, fused vs unfused, so you can see WHERE the fusion win is (not just one speedup ratio).

Totals match fusion_win (same B1 cold-cache + B2 median rig), each split in two:
  fused   total = t(tk_<k>)                         -> gemm = t(tk_noop) [GEMM+store];  epi = total - gemm
  unfused total = t(base->D ; torch ref)            -> gemm = t(base->D);               epi = total - gemm

The fused-epi split is the DIFFERENCE METHOD (approximate): the epilogue overlaps the mainloop
tail, so the delta can be small/noisy. For an exact intra-kernel split use device-side timing (B4).
Requires tk_kernel, tk_noop, and tk_<k> built. Run from the epilogue dir.
"""
import os, sys, importlib, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from epilogue_testlib import EPILOGUES, make_inputs, init_empty, DSIZE
from fusion_win import _bench, _pool_size


def run_one(kernel, shapes, iters, warm, llc_bytes, budget_bytes):
    base = importlib.import_module("tk_kernel")          # GEMM -> D
    noop = importlib.import_module("tk_noop")            # GEMM + store (fused-gemm proxy)
    spec = EPILOGUES[kernel]
    fk = importlib.import_module(spec["module"])         # fused
    rows = []
    for (m, n, k) in shapes:
        per_set = (m * k + n * k) * DSIZE
        N = _pool_size(per_set, llc_bytes, budget_bytes)
        Ap, Btp = [], []
        for _ in range(N):
            a_, bt_ = make_inputs(m, n, k); Ap.append(a_); Btp.append(bt_)
        D = init_empty((m, n)); O = init_empty((m, n))
        args = spec["args"](m, n, k)

        def fused(i):    fk.dispatch_micro(Ap[i % N], Btp[i % N], O, *args)   # GEMM + epilogue
        def f_gemm(i):   noop.dispatch_micro(Ap[i % N], Btp[i % N], O)        # GEMM + store
        def u_gemm(i):   base.dispatch_micro(Ap[i % N], Btp[i % N], D)        # GEMM -> D
        def unfused(i):  base.dispatch_micro(Ap[i % N], Btp[i % N], D); spec["ref"](D, O, *args)

        f_total = _bench(fused,   iters, warm)["median"]
        f_g     = _bench(f_gemm,  iters, warm)["median"]
        u_total = _bench(unfused, iters, warm)["median"]
        u_g     = _bench(u_gemm,  iters, warm)["median"]
        f_epi = max(f_total - f_g, 0.0)
        u_epi = max(u_total - u_g, 0.0)
        rows.append({
            "kernel": kernel, "shape": [m, n, k],
            "f_gemm": round(f_g, 4), "f_epi": round(f_epi, 4), "f_total": round(f_total, 4),
            "u_gemm": round(u_g, 4), "u_epi": round(u_epi, 4), "u_total": round(u_total, 4),
            "epi_x": round(u_epi / f_epi, 1) if f_epi > 1e-3 else None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default="scale,k5,resadd")
    ap.add_argument("--shapes", default="8192x8192x8192,8192x8192x1024,8192x8192x512,4096x4096x4096")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warm", type=int, default=50)
    ap.add_argument("--llc-mb", type=int, default=256)
    ap.add_argument("--budget-gb", type=int, default=24)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    shapes = [tuple(int(x) for x in s.split("x")) for s in a.shapes.split(",")]
    llc, budget = a.llc_mb * 1024 * 1024, a.budget_gb * 1024 * 1024 * 1024
    torch.manual_seed(0)

    rows = []
    for kr in a.kernels.split(","):
        kr = kr.strip()
        if kr not in EPILOGUES:
            raise SystemExit(f"unknown kernel '{kr}'; registered: {list(EPILOGUES)}")
        rows += run_one(kr, shapes, a.iters, a.warm, llc, budget)

    print(f"{'kernel':<8}{'shape (m,n,k)':<20}"
          f"{'f_gemm':>8}{'f_epi':>8}{'f_tot':>8}{'|':>3}{'u_gemm':>8}{'u_epi':>8}{'u_tot':>8}{'epi u/f':>9}")
    print("-" * 90)
    for r in rows:
        sh = "x".join(map(str, r["shape"]))
        ex = f"{r['epi_x']:.1f}x" if r["epi_x"] is not None else "  ~0"
        print(f"{r['kernel']:<8}{sh:<20}"
              f"{r['f_gemm']:>8.3f}{r['f_epi']:>8.3f}{r['f_total']:>8.3f}{'|':>3}"
              f"{r['u_gemm']:>8.3f}{r['u_epi']:>8.3f}{r['u_total']:>8.3f}{ex:>9}")
    print("\nf_* = fused (one kernel); u_* = unfused (GEMM->HBM then torch epilogue).")
    print("f_gemm=t(tk_noop) GEMM+store; f_epi=t(fused)-f_gemm (difference method, approximate).")
    print("u_gemm=t(GEMM->D); u_epi=t(unfused)-u_gemm. cold-cache median (B1+B2). 'epi u/f' = how much")
    print("cheaper the fused epilogue is than the unfused one -- that ratio IS the fusion win.")
    if a.json:
        json.dump({"decompose": rows}, open(a.json, "w"), indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
