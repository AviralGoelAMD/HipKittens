#!/usr/bin/env python3
"""bench.py - one benchmark for every GEMM-epilogue kernel + the fused chain.

Per epilogue: HK fused (one kernel) vs torch unfused (base GEMM -> D in HBM, then the torch `ref`
reads D and applies the epilogue). Reports COLD-cache median speedup + the HBM bytes the fusion
saves. The fused chain (rmsnorm(X@W0+res,gamma)@W1) additionally compares against a Triton baseline
(Tri Dao rms_norm) -- the apples-to-apples full-RMSNorm fusion story.

Honest by construction: B1 rotates the re-read inputs through a pool > the LLC so each timed run is
cold; B2 takes the per-iter median. Kernels + their `ref` come from the single EPILOGUES registry,
so the unfused stage is the same math test_all.py validates.

Run from the epilogue dir on a gfx950 node:
    python3 util/bench.py [--kernels scale,rmsnorm_scale] [--json results.json]
"""
import os, sys, argparse, importlib, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # tk_*.so
import torch
from epilogue_testlib import EPILOGUES, make_inputs, init_empty, init_randn, DSIZE, RTOL, ATOL, DTYPE
from block_chain import fused_rmsnorm_block, EPS

LLC_BYTES = 256 * 1024 * 1024
BUDGET    = 8 * 1024**3
SHAPES       = [(2048, 1024, 512), (4096, 4096, 4096), (8192, 8192, 8192)]
CHAIN_SHAPES = [(2048, 2048, 2048, 2048), (4096, 4096, 4096, 4096)]              # (M,K0,N,P)

# Optional Triton rms_norm (Tri Dao). Absent -> the chain's Triton column is skipped.
_TRITON = None
try:
    _HK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
    sys.path.insert(0, os.path.join(_HK, "training", "llama"))
    from llama.ops.triton.layer_norm import rms_norm_fn as _TRITON
except Exception:
    _TRITON = None


def _pool_size(per_set, llc, budget):
    if per_set >= llc: return 1
    return max(1, min(llc // per_set + 1, max(1, budget // per_set)))


def _bench(fn, iters=50, warm=10):
    for i in range(warm): fn(i)
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters): st[i].record(); fn(i); en[i].record()
    torch.cuda.synchronize()
    s = sorted(st[i].elapsed_time(en[i]) for i in range(iters))
    return s[len(s) // 2]


def run_epilogue(kernel, iters, warm):
    base = importlib.import_module("tk_noop")
    spec = EPILOGUES[kernel]
    fk = importlib.import_module(spec["module"])
    rows = []
    for (m, n, k) in SHAPES:
        per_set = (m * k + n * k) * DSIZE
        N = _pool_size(per_set, LLC_BYTES, BUDGET)
        Ap, Btp = zip(*[make_inputs(m, n, k) for _ in range(N)])
        D = init_empty((m, n)); O = init_empty((m, n)); args = spec["args"](m, n, k)
        fused   = lambda i: fk.dispatch(Ap[i % N], Btp[i % N], O, *args)
        def unfused(i):
            base.dispatch(Ap[i % N], Btp[i % N], D); spec["ref"](D, O, *args)
        fused(0); torch.cuda.synchronize(); Of = O.clone()
        unfused(0); torch.cuda.synchronize()
        ok = torch.allclose(Of.float(), O.float(), rtol=RTOL, atol=ATOL)
        tf, tu = _bench(fused, iters, warm), _bench(unfused, iters, warm)
        saved = spec["hbm_passes"] * m * n * DSIZE
        rows.append({"kernel": kernel, "shape": [m, n, k], "correct": bool(ok),
                     "fused_ms": round(tf, 4), "torch_ms": round(tu, 4),
                     "speedup": round(tu / tf, 3), "saved_MB": round(saved / 1e6, 1)})
    return rows


def run_chain(iters, warm):
    rows = []
    for (M, K0, N, P) in CHAIN_SHAPES:
        X = init_randn((M, K0)); W0 = init_randn((K0, N)); res = init_randn((M, N))
        gamma = init_randn((N,)); W1 = init_randn((N, P))
        def hk(i):   fused_rmsnorm_block(X, W0, res, gamma, W1)
        def torch_(i):
            h1 = X.float() @ W0.float() + res.float()
            hn = (h1 * torch.rsqrt(h1.pow(2).mean(-1, keepdim=True) + EPS) * gamma.float()).to(DTYPE)
            _ = hn @ W1
        row = {"shape": [M, K0, N, P], "hk_ms": round(_bench(hk, iters, warm), 4),
               "torch_ms": round(_bench(torch_, iters, warm), 4)}
        if _TRITON is not None:
            def tri(i):
                h1 = (X @ W0 + res)
                hn = _TRITON(h1, gamma, None, eps=EPS)
                _ = hn @ W1
            try:
                tri(0); torch.cuda.synchronize()
                row["triton_ms"] = round(_bench(tri, iters, warm), 4)
            except Exception as e:
                row["triton_ms"] = f"n/a ({type(e).__name__})"
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default=",".join(k for k in EPILOGUES if k != "noop"))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--no-chain", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    out = {"epilogues": [], "chain": []}
    print(f"{'kernel':<16}{'shape':<22}{'fused ms':>10}{'torch ms':>10}{'speedup':>9}{'saved MB':>10}{'ok':>4}")
    for kern in a.kernels.split(","):
        for r in run_epilogue(kern, a.iters, a.warm):
            out["epilogues"].append(r)
            print(f"{r['kernel']:<16}{str(tuple(r['shape'])):<22}{r['fused_ms']:>10}{r['torch_ms']:>10}"
                  f"{r['speedup']:>9}{r['saved_MB']:>10}{'Y' if r['correct'] else 'N':>4}")
    if not a.no_chain:
        print(f"\n{'chain (M,K0,N,P)':<24}{'hk ms':>10}{'torch ms':>10}{'triton ms':>12}")
        for r in run_chain(a.iters, a.warm):
            out["chain"].append(r)
            print(f"{str(tuple(r['shape'])):<24}{r['hk_ms']:>10}{r['torch_ms']:>10}{str(r.get('triton_ms','n/a')):>12}")
    if a.json:
        with open(a.json, "w") as f: json.dump(out, f, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
