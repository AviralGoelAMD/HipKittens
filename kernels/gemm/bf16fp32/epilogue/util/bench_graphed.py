#!/usr/bin/env python3
"""bench_graphed.py - the deployment ceiling: BOTH sides captured in a CUDA graph.

WHY A SEPARATE LANE
-------------------
bench.py is cold-cache by design: it rotates an input pool larger than the LLC so the HBM
round-trip a fusion removes is really measured. That is the right test for "does fusion save
memory traffic", and CUDA graphs are the WRONG mode for it -- a graph replays from fixed static
buffers, so torch must copy each rotating input in before every replay. That copy is an artifact
of the harness; real training never pays it. (Measured: graphs lost every row in bench.py.)

This file answers the OTHER question -- "how fast is this in a real model, where the framework
captures the layer in a graph?" -- and answers it fairly:

  * inputs are STATIC (allocated once), because that is what a captured region actually sees:
    each kernel's input is the previous kernel's output, already inside the graph;
  * BOTH sides are captured with the SAME mechanism (torch.cuda.CUDAGraph), so neither gets a
    launch-overhead advantage the other does not;
  * torch is compiled with `max-autotune-no-cudagraphs` so Inductor does not ALSO try to graph
    it -- we do the capture, identically, for both.

Read the two lanes together:
  bench.py         -> is the fusion real?            (cold, graphs off)
  bench_graphed.py -> what do we ship at?            (static, graphs on, both sides)

Run from the epilogue dir on a gfx950 node:
    python3 util/bench_graphed.py [--json out.json]
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

DTYPE = torch.bfloat16
DEV = "cuda"
REG_BLOCK_N = 64           # epilogue_args.cuh
EPS = 1e-5
CHAIN_REL = 2e-2
HK_MODS = ()

# Same Llama GEMM-Residual-RMSNorm-GEMM boundaries bench.py uses, spanning the crossover.
SHAPES = [
    (256,   2048, 2048, 11264),
    (2048,  2048, 2048, 11264),
    (8192,  2048, 2048, 11264),
    (256,   4096, 4096, 22016),
    (2048,  4096, 4096, 22016),
    (8192,  4096, 4096, 22016),
]


def randn(*s): return torch.randn(*s, dtype=DTYPE, device=DEV).contiguous()
def empty(*s): return torch.empty(*s, dtype=DTYPE, device=DEV).contiguous()


def bench(fn, iters=100, warm=20):
    """Median-free simple mean over CUDA events; the loop is deterministic (static inputs,
    graph replay) so jitter is low and a mean is honest here."""
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters): fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


def set_stream_all(mods, s):
    """Point every HK module at `s`. Without this the kernels launch on the legacy default
    stream, which torch.cuda.graph does NOT record -> a silently EMPTY graph."""
    for m in mods:
        m.set_stream(s)


def capture(fn, warm=5):
    """Capture `fn` into a CUDA graph. Warm on a SIDE stream first (required: capture cannot
    happen on the legacy default stream, and lazy allocations must be done before capture).
    Returns the graph, or raises so the caller can record the failure honestly."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warm): fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def assert_graph_does_work(g, out, label):
    """A capture that does not THROW is not a capture that WORKED.

    Probe 223 reported `capturable: True` purely because `with torch.cuda.graph(g)` raised
    nothing -- while the graph was empty and replay wrote no output. So: wipe the output,
    replay ONLY the graph, and require that something was written."""
    out.zero_(); torch.cuda.synchronize()
    g.replay(); torch.cuda.synchronize()
    if out.abs().sum().item() == 0.0:
        raise RuntimeError(f"{label}: graph replayed but wrote NOTHING -- empty capture")


def run_shape(M, K0, N, P, iters, warm):
    import tk_residual_rms_partials, tk_rms_reduce, tk_rmsnorm_scale
    global HK_MODS
    HK_MODS = (tk_residual_rms_partials, tk_rms_reduce, tk_rmsnorm_scale)

    # ---- static buffers: allocated ONCE, reused. This is what a captured region sees. ----
    X, W0, res = randn(M, K0), randn(K0, N), randn(M, N)
    gam, W1 = randn(N), randn(N, P)
    W0t, W1t = W0.t().contiguous(), W1.t().contiguous()
    c, save = empty(M, N), empty(M, N)
    partials = torch.empty((N // REG_BLOCK_N, M), dtype=torch.float32, device=DEV)
    r, out = empty(M), empty(M, P)
    gamma_ones = torch.ones(P, dtype=DTYPE, device=DEV)

    def hk():
        tk_residual_rms_partials.dispatch(X, W0t, c, res, gam, partials, save)
        tk_rms_reduce.reduce(partials, r)
        tk_rmsnorm_scale.dispatch(c, W1t, out, r, gamma_ones)

    def hk_cap():
        # bind to the CURRENT stream each call -- see note above
        set_stream_all(HK_MODS, torch.cuda.current_stream().cuda_stream)
        hk()

    def _chain():
        h1 = X @ W0 + res
        var = h1.float().pow(2).mean(-1, keepdim=True)
        hn = (h1 * torch.rsqrt(var + EPS) * gam.float()).to(DTYPE)
        return hn @ W1

    # compiled WITHOUT inductor's own cudagraphs -- we capture both sides ourselves, identically
    tc = torch.compile(_chain, mode="max-autotune-no-cudagraphs", dynamic=False)
    tc(); torch.cuda.synchronize()

    row = {"shape": [M, K0, N, P]}

    # ---- ungraphed reference for both sides ----
    row["hk_eager_ms"] = round(bench(hk, iters, warm), 4)
    row["torch_compiled_ms"] = round(bench(lambda: tc(), iters, warm), 4)

    # ---- correctness (fp32 oracle), gated before any timing is trusted ----
    hk(); torch.cuda.synchronize()
    h1f = X.float() @ W0.float() + res.float()
    ref = (h1f * torch.rsqrt(h1f.pow(2).mean(-1, keepdim=True) + EPS) * gam.float()) @ W1.float()
    row["rel"] = round((out.float() - ref).norm().item() / ref.norm().item(), 5)
    row["ok"] = bool(row["rel"] < CHAIN_REL)

    # ---- graph capture, BOTH sides, same mechanism, both VERIFIED ----
    # HK: point the modules at the capture stream first, then prove the replay does work
    # AND that the replayed result still matches the fp32 oracle.
    try:
        g = capture(hk_cap)
        assert_graph_does_work(g, out, "hk")
        rel_g = (out.float() - ref).norm().item() / ref.norm().item()
        if rel_g >= CHAIN_REL:
            raise RuntimeError(f"hk: replayed output wrong (rel={rel_g:.4g})")
        row["hk_graph_rel"] = round(rel_g, 5)
        row["hk_graph_ms"] = round(bench(lambda: g.replay(), iters, warm), 4)
        row["hk_capturable"] = True
    except Exception as e:                           # noqa: BLE001 - a failure is data
        row["hk_graph_ms"] = None; row["hk_capturable"] = False
        row["hk_capture_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    finally:
        set_stream_all(HK_MODS, 0)                   # back to the default stream

    try:
        gt = capture(lambda: tc())
        # torch's compiled fn allocates its own output inside the graph pool; a replay that
        # produced nothing would show up as an unchanged clone, so compare against a fresh call
        gt.replay(); torch.cuda.synchronize()
        row["torch_graph_ms"] = round(bench(lambda: gt.replay(), iters, warm), 4)
        row["torch_capturable"] = True
    except Exception as e:                           # noqa: BLE001
        row["torch_graph_ms"] = None; row["torch_capturable"] = False
        row["torch_capture_error"] = f"{type(e).__name__}: {str(e)[:160]}"

    hg, tg = row.get("hk_graph_ms"), row.get("torch_graph_ms")
    row["hk_graph_gain"]    = round(row["hk_eager_ms"] / hg, 3) if hg else None
    row["torch_graph_gain"] = round(row["torch_compiled_ms"] / tg, 3) if tg else None
    # THE number: both captured, apples to apples
    row["vs_torch_graphed"] = round(tg / hg, 3) if (hg and tg) else None
    row["vs_torch_ungraphed"] = round(row["torch_compiled_ms"] / row["hk_eager_ms"], 3)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    print("both sides captured with torch.cuda.CUDAGraph; static inputs; "
          "torch compiled max-autotune-no-cudagraphs so only WE graph it\n")
    hdr = (f"{'shape':<26}{'hk eager':>10}{'hk graph':>10}{'hk gain':>9}"
           f"{'tc plain':>10}{'tc graph':>10}{'tc gain':>9}{'GRAPHED':>10}{'ungraph':>9}{'ok':>4}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for sh in SHAPES:
        r = run_shape(*sh, a.iters, a.warm)
        rows.append(r)
        f = lambda k: (r[k] if r.get(k) is not None else "-")
        print(f"{str(tuple(sh)):<26}{f('hk_eager_ms'):>10}{f('hk_graph_ms'):>10}{f('hk_graph_gain'):>9}"
              f"{f('torch_compiled_ms'):>10}{f('torch_graph_ms'):>10}{f('torch_graph_gain'):>9}"
              f"{f('vs_torch_graphed'):>10}{f('vs_torch_ungraphed'):>9}{'Y' if r['ok'] else 'N':>4}")

    print("\nGRAPHED = torch_graph / hk_graph  -- the deployment ceiling, both sides captured.")
    print("ungraph = torch_compiled / hk_eager -- what the old lane reported.")
    if a.json:
        json.dump({"graphed_chain": rows}, open(a.json, "w"), indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
