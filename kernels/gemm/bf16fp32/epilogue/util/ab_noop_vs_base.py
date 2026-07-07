#!/usr/bin/env python3
"""ab_noop_vs_base.py - Review #8: A/B the templatized tk_noop (gemm_kernel<NoOpEpilogue>) against the
UNTOUCHED base GEMM (tk_kernel / micro_tk from 256_256_64_32_with16x32.cpp), to back the
"refactor is perf-neutral (noop == base GEMM at 8192^3)" claim with a measured number.

Both take the same (A, Bt, C) -- tk_kernel via dispatch_micro, tk_noop via dispatch -- and compute
A @ Bt.t(). Cold-cache rotating pool + per-iter median (same protocol as bench.py). At the
compute-bound squares the two should be within a few percent AND produce the same C.

Run inside the kreb container from the epilogue dir (needs tk_kernel + tk_noop built):
    python3 util/ab_noop_vs_base.py
"""
import os, sys, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # tk_*.so live here
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # util/ for the registry helpers
import torch
from epilogue_testlib import make_inputs, init_empty, DSIZE, RTOL, ATOL

LLC_BYTES = 256 * 1024 * 1024
BUDGET    = 8 * 1024**3
SHAPES    = [(4096, 4096, 4096), (8192, 8192, 8192)]   # compute-bound squares -- the perf-neutral regime
TOL       = 0.05                                        # perf-neutral = NO REGRESSION: noop must not be slower than base by > TOL (noop faster is fine)


def _pool_size(per_set):
    if per_set >= LLC_BYTES:
        return 1
    return max(1, min(LLC_BYTES // per_set + 1, max(1, BUDGET // per_set)))


def _bench(fn, iters=50, warm=10):
    for i in range(warm): fn(i)
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters): st[i].record(); fn(i); en[i].record()
    torch.cuda.synchronize()
    s = sorted(st[i].elapsed_time(en[i]) for i in range(iters))
    return s[len(s) // 2]


def main():
    base = importlib.import_module("tk_kernel")   # micro_tk -- the untouched base GEMM
    noop = importlib.import_module("tk_noop")      # gemm_kernel<NoOpEpilogue> -- the templatized copy
    print(f"{'shape (M,N,K)':<20}{'base ms':>10}{'noop ms':>10}{'noop/base':>11}{'match':>7}{'ok':>4}")
    allok = True
    for (M, N, K) in SHAPES:
        per_set = (M * K + N * K) * DSIZE
        Np = _pool_size(per_set)
        Ap, Btp = zip(*[make_inputs(M, N, K) for _ in range(Np)])   # cold-cache pool; rotate via [i % Np]
        Cb = init_empty((M, N)); Cn = init_empty((M, N))
        def fbase(i): base.dispatch_micro(Ap[i % Np], Btp[i % Np], Cb)
        def fnoop(i): noop.dispatch(Ap[i % Np], Btp[i % Np], Cn)
        fbase(0); fnoop(0); torch.cuda.synchronize()
        match = torch.allclose(Cb.float(), Cn.float(), rtol=RTOL, atol=ATOL)   # same GEMM -> same C
        tb, tn = _bench(fbase), _bench(fnoop)
        ratio = tn / tb
        ok = bool(match and ratio <= 1.0 + TOL)         # no regression; noop faster (ratio<1) passes
        allok &= ok
        print(f"{str((M, N, K)):<20}{tb:>10.4f}{tn:>10.4f}{ratio:>11.3f}{'Y' if match else 'N':>7}{'Y' if ok else 'N':>4}")
    print(f"PERF-NEUTRAL OK (noop within +{TOL:.0%} of base or faster)" if allok else f"PERF-NEUTRAL FAIL (noop slower by >{TOL:.0%} or mismatch)")
    return allok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
