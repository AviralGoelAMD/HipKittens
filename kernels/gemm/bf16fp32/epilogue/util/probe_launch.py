#!/usr/bin/env python3
"""probe_launch.py - split "our kernels are slower" from "our launcher is slower".

WHY THIS EXISTS
---------------
bench.py times with CUDA events, which measure elapsed time ON THE STREAM. If the CPU cannot
issue work fast enough, the GPU sits idle and that idle time lands inside the event window --
so a launch-bound kernel and a slow kernel look identical.

The discriminator is HOST time vs DEVICE time for the same loop:
  * device time  = CUDA events around N iterations (what bench.py reports)
  * host time    = perf_counter around the SAME N iterations with NO per-iteration sync
                   (one sync at the very end), i.e. how long the CPU took to ISSUE the work.

  host >= device  ->  the CPU is the bottleneck; the GPU is starved. LAUNCH-BOUND.
  host <  device  ->  the CPU runs ahead and the GPU is saturated. KERNEL-BOUND.

This is the same diagnosis that corrected the GDN B=1 conclusion (a 1.6x "kernel gap" that was
entirely host launch overhead).

It also reports a direct per-dispatch launch cost, measured on the SMALLEST legal shape
(M=N=BLOCK_SIZE, K=K_ALIGN) where the kernel's own work is negligible, so the host cost of a
dispatch is what remains.

Finally it probes whether the HK dispatches can be captured into a CUDA graph at all. They
launch on stream 0 (the legacy default stream), which is NOT capturable -- if that is what
happens, it is a finding, not a bug in this script: HK cannot be graph-captured until the
binding accepts a stream.

Run from the epilogue dir on a gfx950 node:
    python3 util/probe_launch.py
"""
import os, sys, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

DTYPE = torch.bfloat16
DEV = "cuda"
BLOCK_SIZE, K_ALIGN, REG_BLOCK_N = 256, 128, 64      # epilogue_args.cuh
EPS = 1e-5

# The shapes where the chain LOSES, plus one where it wins, so the probe covers the crossover.
SHAPES = [(256, 2048, 2048, 11264),      # 0.43x vs compiled  <- the worst case
          (2048, 2048, 2048, 11264),     # 0.81x
          (8192, 2048, 2048, 11264)]     # 1.09x  <- HK wins here


def randn(*s): return torch.randn(*s, dtype=DTYPE, device=DEV).contiguous()
def empty(*s): return torch.empty(*s, dtype=DTYPE, device=DEV).contiguous()


def timed(fn, iters, warm):
    """Return (device_ms_per_iter, host_ms_per_iter) for the SAME loop.

    device: CUDA events bracketing the whole loop (elapsed time on the stream).
    host:   perf_counter bracketing the same loop, NO sync inside -- pure issue cost.
    One sync at the end so the device number is complete."""
    for i in range(warm): fn(i)
    torch.cuda.synchronize()

    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    st.record()
    for i in range(iters): fn(i)
    en.record()
    t1 = time.perf_counter()          # CPU is done ISSUING here
    torch.cuda.synchronize()          # GPU is done EXECUTING here
    return st.elapsed_time(en) / iters, (t1 - t0) * 1e3 / iters


def probe_dispatch_cost(iters=2000):
    """Host cost of ONE tk_noop dispatch at the smallest legal shape.

    At 256x256x128 the kernel's own work is ~nothing, so the host time per call is dominated by
    pybind marshalling + hipLaunchKernel. This is the per-launch tax every HK kernel pays."""
    import tk_noop
    M = N = BLOCK_SIZE; K = K_ALIGN
    A, Bt, C = randn(M, K), randn(N, K), empty(M, N)
    for _ in range(200): tk_noop.dispatch(A, Bt, C)
    torch.cuda.synchronize()
    dev, host = timed(lambda i: tk_noop.dispatch(A, Bt, C), iters, 100)
    return {"shape": [M, N, K], "device_us": round(dev * 1e3, 2), "host_us": round(host * 1e3, 2)}


def probe_graph_capture():
    """Can the HK dispatches be captured into a CUDA graph?

    They launch on stream 0 (legacy default). Capturing the legacy default stream is illegal, so
    this is expected to FAIL -- and that failure is the point: it means HK cannot ride a CUDA
    graph until the binding takes a stream."""
    import tk_noop
    M = N = BLOCK_SIZE; K = K_ALIGN
    A, Bt, C = randn(M, K), randn(N, K), empty(M, N)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): tk_noop.dispatch(A, Bt, C)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            tk_noop.dispatch(A, Bt, C)
        g.replay(); torch.cuda.synchronize()
        return {"capturable": True, "error": None}
    except Exception as e:
        return {"capturable": False, "error": f"{type(e).__name__}: {str(e)[:220]}"}


def probe_chain(M, K0, N, P, iters, warm):
    import tk_residual_rms_partials, tk_rms_reduce, tk_rmsnorm_scale
    X, W0, res = randn(M, K0), randn(K0, N), randn(M, N)
    g_, W1 = randn(N), randn(N, P)
    W0t, W1t = W0.t().contiguous(), W1.t().contiguous()
    c, save = empty(M, N), empty(M, N)
    partials = torch.empty((N // REG_BLOCK_N, M), dtype=torch.float32, device=DEV)
    r, out = empty(M), empty(M, P)
    gamma_ones = torch.ones(P, dtype=DTYPE, device=DEV)

    def hk(i):                                   # 3 separate pybind dispatches
        tk_residual_rms_partials.dispatch(X, W0t, c, res, g_, partials, save)
        tk_rms_reduce.reduce(partials, r)
        tk_rmsnorm_scale.dispatch(c, W1t, out, r, gamma_ones)

    def _chain(X, W0, res, gamma, W1):
        h1 = X @ W0 + res
        var = h1.float().pow(2).mean(-1, keepdim=True)
        hn = (h1 * torch.rsqrt(var + EPS) * gamma.float()).to(DTYPE)
        return hn @ W1

    cc = torch.compile(_chain, mode="max-autotune-no-cudagraphs", dynamic=False)
    cc(X, W0, res, g_, W1); torch.cuda.synchronize()          # compile OUTSIDE timing

    hk_dev, hk_host = timed(hk, iters, warm)
    tc_dev, tc_host = timed(lambda i: cc(X, W0, res, g_, W1), iters, warm)
    eg_dev, eg_host = timed(lambda i: _chain(X, W0, res, g_, W1), iters, warm)

    return {"shape": [M, K0, N, P], "hk_launches": 3,
            "hk_device_ms": round(hk_dev, 4),   "hk_host_ms": round(hk_host, 4),
            "torch_device_ms": round(tc_dev, 4), "torch_host_ms": round(tc_host, 4),
            "eager_device_ms": round(eg_dev, 4), "eager_host_ms": round(eg_host, 4),
            # >1 means the CPU cannot keep the GPU fed -> launch-bound
            "hk_host_over_device": round(hk_host / hk_dev, 3),
            "torch_host_over_device": round(tc_host / tc_dev, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    out = {}
    print("=== per-dispatch host cost (smallest legal shape, kernel work ~ 0) ===")
    out["dispatch"] = probe_dispatch_cost()
    d = out["dispatch"]
    print(f"  tk_noop @ {d['shape']}:  device {d['device_us']} us   host {d['host_us']} us")
    print(f"  -> a 3-dispatch chain pays ~{round(3*d['host_us'],1)} us of HOST issue cost per iteration\n")

    print("=== CUDA-graph capturability of the HK dispatches ===")
    out["graph"] = probe_graph_capture()
    print(f"  capturable: {out['graph']['capturable']}")
    if out["graph"]["error"]: print(f"  error: {out['graph']['error']}")
    print()

    print("=== chain: host (issue) vs device (execute) ===")
    print(f"{'shape':<26}{'hk dev':>9}{'hk host':>9}{'h/d':>7}{'tc dev':>9}{'tc host':>9}{'h/d':>7}{'verdict':>16}")
    out["chain"] = []
    for sh in SHAPES:
        r = probe_chain(*sh, a.iters, a.warm)
        out["chain"].append(r)
        verdict = "LAUNCH-BOUND" if r["hk_host_over_device"] >= 0.95 else "kernel-bound"
        print(f"{str(tuple(sh)):<26}{r['hk_device_ms']:>9}{r['hk_host_ms']:>9}"
              f"{r['hk_host_over_device']:>7}{r['torch_device_ms']:>9}{r['torch_host_ms']:>9}"
              f"{r['torch_host_over_device']:>7}{verdict:>16}")

    print("\n=== interpretation ===")
    for r in out["chain"]:
        M = r["shape"][0]
        launch_us = 3 * out["dispatch"]["host_us"]
        share = 100 * (launch_us / 1e3) / r["hk_device_ms"]
        print(f"  M={M:<6} HK device {r['hk_device_ms']} ms; 3 launches ~= {round(launch_us,1)} us "
              f"= {share:.1f}% of it")

    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
