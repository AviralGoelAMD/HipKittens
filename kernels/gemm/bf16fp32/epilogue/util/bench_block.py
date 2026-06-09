#!/usr/bin/env python3
"""bench_block.py - payoff of the K4->aux->K5 money-pattern chain (Task 2.5 Step 5).

Runs  out = rmsnorm(X@W0 + residual, gamma) @ W1  BOTH ways on the same inputs:
  fused   : tk_k4 -> tk_aux_rms -> tk_k5  (residual-add + RMSNorm folded into the two GEMM
            epilogues; only the bf16 c=[M,N] round-trips HBM between the GEMMs)
  unfused : tk_kernel (X@W0) -> torch (h+residual, rmsnorm*gamma) -> tk_kernel (hn@W1)
            (the realistic 4-op path: gemm + add + rmsnorm + gemm, materializing h/h1/hn)

The GEMM kernel (tk_kernel) is IDENTICAL in both paths, so the wall-clock delta is attributable
to folding the elementwise passes into the epilogues, not to a different matmul.

Honest timing: B1 rotates the re-read inputs (X, W0t, residual, gamma, W1t) through a pool sized
> the GPU LLC so each timed iter reads them cold; B2 reports the per-iter MEDIAN (+min). A
correctness gate requires fused == unfused == fp32 torch reference before any number is trusted.

Run from the epilogue dir on a gfx950 node (after building tk_kernel, tk_k4, tk_aux_rms, tk_k5):
    python3 util/bench_block.py --json results_block.json
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import tk_kernel, tk_k4, tk_aux_rms, tk_k5
from epilogue_testlib import init_randn, init_empty, DSIZE, DTYPE
from fusion_win import _bench, _pool_size

EPS = 1e-5
# (M, K0, N, P): M/N/P % 256 (tiling); K0,N % 128 (GEMM contraction dims, [C13]).
# compute-bound (square) -> memory-bound (skinny K0).
DEFAULT_SHAPES = [(1024, 1024, 1024, 1024), (2048, 2048, 2048, 2048),
                  (4096, 4096, 4096, 4096), (2048, 512, 2048, 512)]


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
    h = init_empty((m, n)); out_u = init_empty((m, p))

    def fused(i):
        j = i % N
        tk_k4.dispatch_micro(Xp[j], W0tp[j], c, alpha, r_dummy, Gp[j], Rp[j], partials, save)
        tk_aux_rms.reduce(partials, r)
        tk_k5.dispatch_micro(c, W1tp[j], out_f, alpha, r, gamma_ones)

    def unfused(i):
        j = i % N
        tk_kernel.dispatch_micro(Xp[j], W0tp[j], h)                 # h = X@W0
        h1 = h.float() + Rp[j].float()
        hn = (h1 * torch.rsqrt(h1.pow(2).mean(-1, keepdim=True) + EPS) * Gp[j].float()).to(DTYPE)
        tk_kernel.dispatch_micro(hn, W1tp[j], out_u)               # out = hn@W1

    # correctness gate: fused == unfused == fp32 reference
    fused(0); torch.cuda.synchronize(); Of = out_f.clone()
    unfused(0); torch.cuda.synchronize(); Ou = out_u.clone()
    ref = _ref(Xp[0], W0tp[0], Rp[0], Gp[0], W1tp[0])
    den = ref.abs().max().item()
    # gate each path against the fp32 TRUTH (fused and unfused are each ~1% bf16-lossy, so they
    # need not match each other bit-for-bit; both must match ground truth). Relative max-error.
    rel_f = (Of.float() - ref).abs().max().item() / den
    rel_u = (Ou.float() - ref).abs().max().item() / den
    ok = rel_f < 3e-2 and rel_u < 3e-2

    tf = _bench(fused, iters, warm)
    tu = _bench(unfused, iters, warm)
    # intermediates fusion folds into the epilogues: unfused materializes h1 and hn (2 extra
    # [M,N] write+read passes) that the fused path avoids; it pays one [M,N] `c` round-trip
    # (+ a bwd-only `save` write). Net [M,N] HBM traffic avoided, conservative:
    saved = 2 * m * n * DSIZE
    return {"shape": list(shape), "pool": N, "correct": bool(ok),
            "rel_fused": round(rel_f, 4), "rel_unfused": round(rel_u, 4),
            "fused_ms": round(tf["median"], 4), "unfused_ms": round(tu["median"], 4),
            "fused_min": round(tf["min"], 4), "unfused_min": round(tu["min"], 4),
            "speedup": round(tu["median"] / tf["median"], 3),
            "interm_saved_MB": round(saved / 1e6, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--llc-mb", type=int, default=256)
    ap.add_argument("--budget-gb", type=int, default=24)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = [run_one(s, a.iters, a.warm, a.llc_mb * 2**20, a.budget_gb * 2**30) for s in DEFAULT_SHAPES]
    hdr = f"{'M,K0,N,P':<24} {'pool':>4} {'ok':>3} {'rel_f':>7} {'fused_ms':>9} {'unfused_ms':>11} {'speedup':>8} {'interm_MB':>9}"
    print(hdr); print("-" * len(hdr))
    for x in rows:
        print(f"{str(tuple(x['shape'])):<24} {x['pool']:>4} {('Y' if x['correct'] else 'N'):>3} "
              f"{x['rel_fused']:>7} {x['fused_ms']:>9} {x['unfused_ms']:>11} {x['speedup']:>7}x {x['interm_saved_MB']:>9}")
    if a.json:
        with open(a.json, "w") as f: json.dump(rows, f, indent=2)
        print(f"\nwrote {a.json}")
    return all(x["correct"] for x in rows)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
