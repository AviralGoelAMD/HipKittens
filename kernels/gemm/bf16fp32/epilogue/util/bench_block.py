#!/usr/bin/env python3
"""bench_block.py - payoff of the K4->aux->K5 money-pattern chain (Task 2.5 Step 5).

Runs  out = rmsnorm(X@W0 + residual, gamma) @ W1  three ways on the same inputs:
  fused   : tk_k4 -> tk_aux_rms -> tk_k5   (residual-add + RMSNorm folded into the GEMM epilogues;
            X@W0 stays fp32 into the add+reduction; only bf16 c=[M,N] round-trips between GEMMs)
  triton  : tk_kernel (X@W0) -> Triton fused residual+rms_norm (Tri Dao) -> tk_kernel (hn@W1)
  torch   : tk_kernel (X@W0) -> torch eager (h+residual, rmsnorm*gamma) -> tk_kernel (hn@W1)

NOT a bit-identical microbenchmark -- it is an IMPLEMENTATION comparison of the same function:
  - the two GEMMs are the SAME tk_kernel in all three paths (identical FLOPs/mainloop), so the
    delta is purely how residual-add + RMSNorm are done;
  - every path is gated against the SAME fp32 ground truth (proof they compute one function);
  - KNOWN asymmetries, disclosed: (a) fused keeps X@W0 in fp32 through the add+RMS reduction
    while the baselines round it to bf16 first (fused is *more accurate*, favorable to fused);
    (b) K4 writes a bwd-only `save`=[M,N] tile the baselines don't (one extra HBM write,
    UNfavorable to fused). They roughly offset; the wall-clock is a defensible real-world number.
  - eps=1e-5 and dropout_p=0 are pinned on the Triton path so it is the SAME normalization
    (Triton's rms_norm_fn defaults to eps=1e-6 -- a different function if left unset).

Honest timing: B1 rotates the re-read inputs (X, W0t, residual, gamma, W1t) through a pool sized
> the GPU LLC so each timed iter reads them cold; B2 reports the per-iter MEDIAN (+min).

Run from the epilogue dir on a gfx950 node (after building tk_kernel, tk_k4, tk_aux_rms, tk_k5):
    python3 util/bench_block.py --json results_block.json
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import tk_kernel, tk_k4, tk_aux_rms, tk_k5
try:
    import tk_k4_fwd                      # forward-only K4 (no save tile); optional
except Exception:
    tk_k4_fwd = None
from epilogue_testlib import init_randn, init_empty, DSIZE, DTYPE
# optional Triton fused residual+rms_norm baseline (Tri Dao's; the reference HK Llama uses)
try:
    _hk = os.path.abspath(__file__)
    for _ in range(6): _hk = os.path.dirname(_hk)   # util->epilogue->bf16fp32->gemm->kernels->HipKittens
    sys.path.insert(0, os.path.join(_hk, "training/llama/llama/ops/triton"))
    from layer_norm import rms_norm_fn as _rms_norm_fn
except Exception:
    _rms_norm_fn = None

# B1 cold-cache pool sizing + B2 CUDA-event median timing (inlined to keep this bench self-contained).
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

EPS = 1e-5
# (M, K0, N, P): M/N/P % 256 (tiling); K0,N % 128 (GEMM contraction dims, [C13]).
# compute-bound (square) -> memory-bound (skinny K0).
DEFAULT_SHAPES = [
    # square, compute-bound, increasing
    (512, 512, 512, 512), (1024, 1024, 1024, 1024), (2048, 2048, 2048, 2048),
    (4096, 4096, 4096, 4096), (8192, 8192, 8192, 8192),
    # skinny K0 -> memory-bound first GEMM
    (2048, 512, 2048, 512), (4096, 512, 4096, 512), (8192, 1024, 8192, 1024),
    # rectangular / transformer-ish (up-proj N>K0, down-proj P<N, varied token count M)
    (4096, 2048, 8192, 2048), (8192, 2048, 2048, 2048), (2048, 4096, 1024, 4096),
    (4096, 4096, 11008, 4096),                 # Llama-7B FFN width (11008 = 43*256)
    # heavily memory-bound: tiny contraction K0 (128/256) and/or few tokens M w/ large weights
    # (low arithmetic intensity -> the [M,N] intermediate round-trip dominates; fusion's sweet spot)
    (8192, 128, 8192, 256), (8192, 256, 8192, 256), (16384, 256, 8192, 256),
    (16384, 256, 4096, 256), (16384, 512, 2048, 512), (8192, 256, 2048, 256),
    (4096, 128, 8192, 256), (32768, 512, 1024, 512),
    (256, 8192, 8192, 8192), (512, 8192, 8192, 8192),   # few tokens (decode-like) -> weight-bound
]


def _parse_shapes(s):
    out = []
    for tok in s.split(","):
        parts = tok.strip().lower().split("x")
        if len(parts) != 4:
            raise SystemExit(f"bad shape '{tok}', want MxK0xNxP (e.g. 4096x4096x4096x4096)")
        out.append(tuple(int(v) for v in parts))
    return out


def _valid(shape):
    m, k0, n, p = shape
    return m % 256 == 0 and n % 256 == 0 and p % 256 == 0 and k0 % 128 == 0 and n % 128 == 0


def _ref(X, W0t, residual, gamma, W1t):
    h1 = X.float() @ W0t.t().float() + residual.float()
    rms = torch.rsqrt(h1.pow(2).mean(-1, keepdim=True) + EPS)
    hn = h1 * rms * gamma.float()
    return hn @ W1t.t().float()


def run_one(shape, iters, warm, llc_bytes, budget_bytes):
    m, k0, n, p = shape
    per_set = (m * k0 + k0 * n + m * n + n * p + n) * DSIZE   # X+W0t+residual+gamma+W1t
    N = _pool_size(per_set, llc_bytes, budget_bytes)
    Xp, W0tp, Rp, Gp, W1tp = [], [], [], [], []
    for _ in range(N):
        Xp.append(init_randn((m, k0)))
        W0tp.append(init_randn((n, k0)))          # W0 transposed -> [N,K0] (kernel takes Bt)
        Rp.append(init_randn((m, n)))
        Gp.append(init_randn((n,)))
        W1tp.append(init_randn((p, n)))           # W1 transposed -> [P,N]

    # fused scratch (write-only across iters -> single set)
    c = init_empty((m, n)); save = init_empty((m, n)); out_f = init_empty((m, p))
    partials = torch.zeros((n // 64, m), dtype=torch.float32, device="cuda")
    r = torch.empty(m, dtype=DTYPE, device="cuda")
    alpha = torch.ones(1, dtype=torch.float32, device="cuda")
    r_dummy = torch.ones(m, dtype=DTYPE, device="cuda")
    gamma_ones = torch.ones(p, dtype=DTYPE, device="cuda")
    h = init_empty((m, n)); out_tor = init_empty((m, p)); out_tri = init_empty((m, p))
    out_fwd = init_empty((m, p))

    def fused(i):
        j = i % N
        tk_k4.dispatch_micro(Xp[j], W0tp[j], c, alpha, r_dummy, Gp[j], Rp[j], partials, save)
        tk_aux_rms.reduce(partials, r)
        tk_k5.dispatch_micro(c, W1tp[j], out_f, alpha, r, gamma_ones)

    def fused_fwd(i):                      # forward-only: K4 WITHOUT the bwd-only save tile
        j = i % N
        tk_k4_fwd.dispatch_micro(Xp[j], W0tp[j], c, alpha, r_dummy, Gp[j], Rp[j], partials)
        tk_aux_rms.reduce(partials, r)
        tk_k5.dispatch_micro(c, W1tp[j], out_fwd, alpha, r, gamma_ones)

    def torch_path(i):
        j = i % N
        tk_kernel.dispatch_micro(Xp[j], W0tp[j], h)                 # h = X@W0 (bf16)
        h1 = h.float() + Rp[j].float()
        hn = (h1 * torch.rsqrt(h1.pow(2).mean(-1, keepdim=True) + EPS) * Gp[j].float()).to(DTYPE)
        tk_kernel.dispatch_micro(hn, W1tp[j], out_tor)             # out = hn@W1

    def triton_path(i):
        j = i % N
        tk_kernel.dispatch_micro(Xp[j], W0tp[j], h)                 # h = X@W0 (bf16) -- same GEMM
        # prenorm=False: return only the normed output (residual still added internally); avoids
        # writing the unused residual_out tensor -> the baseline at its best (fair to Triton).
        hn = _rms_norm_fn(h, Gp[j], None, residual=Rp[j], prenorm=False, eps=EPS, dropout_p=0.0)
        if isinstance(hn, tuple): hn = hn[0]
        tk_kernel.dispatch_micro(hn.contiguous(), W1tp[j], out_tri) # out = hn@W1

    # correctness gate: EVERY path vs the SAME fp32 ground truth (relative max-error)
    ref = _ref(Xp[0], W0tp[0], Rp[0], Gp[0], W1tp[0]); den = ref.abs().max().item()
    def _rel(fn, buf):
        fn(0); torch.cuda.synchronize()
        return (buf.float() - ref).abs().max().item() / den
    rel_f = _rel(fused, out_f)
    rel_tor = _rel(torch_path, out_tor)
    rel_tri = _rel(triton_path, out_tri) if _rms_norm_fn is not None else None
    rel_fwd = _rel(fused_fwd, out_fwd) if tk_k4_fwd is not None else None
    ok = (rel_f < 3e-2 and rel_tor < 3e-2
          and (rel_tri is None or rel_tri < 3e-2)
          and (rel_fwd is None or rel_fwd < 3e-2))
    out_bare = init_empty((m, p))
    def bare2gemm(i):                     # two raw GEMMs, NO norm -> the GEMM-only floor
        j = i % N
        tk_kernel.dispatch_micro(Xp[j], W0tp[j], h)
        tk_kernel.dispatch_micro(h, W1tp[j], out_bare)

    tf = _bench(fused, iters, warm)
    ttor = _bench(torch_path, iters, warm)
    ttri = _bench(triton_path, iters, warm) if _rms_norm_fn is not None else None
    tfwd = _bench(fused_fwd, iters, warm) if tk_k4_fwd is not None else None
    tg = _bench(bare2gemm, iters, warm)
    gemm_ms = tg["median"]
    # stage = full chain - bare two-GEMM floor. For fused this is the epilogue+aux MARGINAL cost
    # (norm work fused inside/around the GEMMs); for triton/torch it is the separate norm kernel(s).
    def _stage(t): return round(t["median"] - gemm_ms, 4) if t is not None else None
    stg_fwd = _stage(tfwd); stg_tri = _stage(ttri); stg_tor = _stage(ttor)
    def _sr(s):  # stage-only ratio vs fused; None when the fused stage is below the noise floor
        return round(s / stg_fwd, 2) if (stg_fwd and stg_fwd > 0.003 and s is not None and s > 0) else None
    saved = 2 * m * n * DSIZE   # [M,N] intermediate passes fused folds away (conservative; see docstring)
    row = {"shape": list(shape), "pool": N, "correct": bool(ok),
           "rel_fused": round(rel_f, 4), "rel_torch": round(rel_tor, 4),
           "fused_save_ms": round(tf["median"], 4), "torch_ms": round(ttor["median"], 4),
           "interm_saved_MB": round(saved / 1e6, 1)}
    row["triton_ms"] = round(ttri["median"], 4) if ttri is not None else None
    row["rel_triton"] = round(rel_tri, 4) if rel_tri is not None else None
    if tfwd is not None:
        row["fwd_ms"] = round(tfwd["median"], 4); row["rel_fwd"] = round(rel_fwd, 4)
        row["fwd_vs_triton"] = round(ttri["median"] / tfwd["median"], 3) if ttri is not None else None
        row["fwd_vs_torch"] = round(ttor["median"] / tfwd["median"], 3)
    else:
        row["fwd_ms"] = None; row["rel_fwd"] = None; row["fwd_vs_triton"] = None; row["fwd_vs_torch"] = None
    row["gemm2_ms"] = round(gemm_ms, 4)
    row["stage_fwd_ms"] = stg_fwd; row["stage_triton_ms"] = stg_tri; row["stage_torch_ms"] = stg_tor
    row["stage_tri_over_fwd"] = _sr(stg_tri); row["stage_tor_over_fwd"] = _sr(stg_tor)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--llc-mb", type=int, default=256)
    ap.add_argument("--budget-gb", type=int, default=24)
    ap.add_argument("--json", default=None)
    ap.add_argument("--shapes", default=None, help="comma list MxK0xNxP, e.g. 4096x4096x4096x4096,2048x512x2048x512")
    a = ap.parse_args()
    shapes = _parse_shapes(a.shapes) if a.shapes else DEFAULT_SHAPES
    skipped = [s for s in shapes if not _valid(s)]
    shapes = [s for s in shapes if _valid(s)]
    for s in skipped:
        print(f"[skip] {tuple(s)} violates M/N/P%256 or K0/N%128 ([C13])")
    rows = [run_one(s, a.iters, a.warm, a.llc_mb * 2**20, a.budget_gb * 2**30) for s in shapes]
    if _rms_norm_fn is None:
        print("[note] Triton rms_norm_fn unavailable -> triton column = n/a")
    if tk_k4_fwd is None:
        print("[note] tk_k4_fwd not built -> fwd column = n/a (build: util/build.sh --no-base k4_fwd)")
    hdr = (f"{'M,K0,N,P':<22} {'ok':>3} {'fwd_ms':>8} {'save_ms':>8} {'triton_ms':>10} {'torch_ms':>9} "
           f"{'fwd/tri':>8} {'fwd/tor':>8}")
    print(hdr); print("-" * len(hdr))
    for x in rows:
        _s = lambda v: v if v is not None else "n/a"
        fvt = f"{x['fwd_vs_triton']}x" if x['fwd_vs_triton'] is not None else "n/a"
        fvo = f"{x['fwd_vs_torch']}x" if x['fwd_vs_torch'] is not None else "n/a"
        print(f"{str(tuple(x['shape'])):<22} {('Y' if x['correct'] else 'N'):>3} "
              f"{str(_s(x['fwd_ms'])):>8} {x['fused_save_ms']:>8} {str(_s(x['triton_ms'])):>10} "
              f"{x['torch_ms']:>9} {fvt:>8} {fvo:>8}")
    # --- stage-isolated: full chain MINUS the bare two-GEMM floor = the residual+RMSNorm stage ---
    print()
    h2 = (f"{'M,K0,N,P':<22} {'gemm2_ms':>9} {'stg_fwd':>8} {'stg_tri':>8} {'stg_tor':>8} "
          f"{'tri/fwd':>8} {'tor/fwd':>8}")
    print(h2); print("-" * len(h2))
    for x in rows:
        _s2 = lambda v: v if v is not None else "n/a"
        tr = f"{x['stage_tri_over_fwd']}x" if x.get('stage_tri_over_fwd') is not None else "n/a"
        to = f"{x['stage_tor_over_fwd']}x" if x.get('stage_tor_over_fwd') is not None else "n/a"
        print(f"{str(tuple(x['shape'])):<22} {x['gemm2_ms']:>9} {str(_s2(x['stage_fwd_ms'])):>8} "
              f"{str(_s2(x['stage_triton_ms'])):>8} {str(_s2(x['stage_torch_ms'])):>8} {tr:>8} {to:>8}")
    print("[note] stage = full chain - bare 2-GEMM floor (difference method); ~0 or negative = "
          "stage below the timing noise floor (GEMM-dominated/compute-bound).")
    if a.json:
        with open(a.json, "w") as f: json.dump(rows, f, indent=2)
        print(f"\nwrote {a.json}")
    return all(x["correct"] for x in rows)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
