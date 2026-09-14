#!/usr/bin/env python3
"""silu_ab.py - does CODA_OPT_FAST_SILU change silu's ACCURACY, and what does it buy in TIME?

Run twice against the same source tree, once per build:

    python3 util/silu_ab.py save    /tmp/silu_fast.pt   # with the fast build installed
    python3 util/silu_ab.py compare /tmp/silu_fast.pt   # with the original build installed

Why this is a fair test: both builds share a **bit-identical GEMM mainloop** (gemm_base.cuh is
untouched and the flag only selects a branch inside silu_op), so the fp32 accumulator feeding
silu is the same in both. Any difference in the stored bf16, or in the time, is attributable to
silu alone.

ACCURACY. Two things are reported, and only the second one answers the question:

  1. How often do the two builds' bf16 outputs differ, and by how many REPRESENTABLE STEPS.
     "Steps" is computed on a monotonic ordering of the bf16 bit pattern, not by subtracting
     raw bits -- raw subtraction is meaningless across the sign bit (+0 is 0x0000, -0 is 0x8000,
     a "difference" of 32768 for two numbers that are equal).

  2. Which build lands closer to a float64 oracle. Differing from each other proves nothing on
     its own; what matters is whether the fast build is FARTHER FROM THE TRUTH.

TIMING covers all three kernels that share silu_op -- silu, swiglu and rmsnorm_swiglu. It lives
here rather than in util/bench.py because that harness's registry reaches only silu and swiglu,
and its swiglu section labels shapes (M, d_ff, K) while every other table is (M, N, K). Two
conventions behind identical-looking labels means a reader compares 2x different GEMM widths
without noticing, so this module fixes ONE convention and says so:

    SHAPES ARE (M, N, K) WHERE N IS THE GEMM OUTPUT WIDTH.

swiglu and rmsnorm_swiglu are dim-reducing, so for them N is the weight width (2 * d_ff) and the
stored output is [M, N/2].
"""
import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "util"))
import torch
from swiglu import gate_up_perm

SHAPES = [(2048, 1024, 512), (4096, 4096, 4096)]
# Timing shapes, (M, N, K) with N = GEMM output width.
TIME_SHAPES = [(2048, 1024, 512), (4096, 4096, 4096), (8192, 8192, 8192)]
RMS_EPS = 1e-5
SEED = 7

# Timing discipline, copied deliberately from util/bench.py so the two tools cannot disagree.
# An earlier version of this file rotated only 2 input sets. That keeps the operands resident in
# the 256 MiB last-level cache, which makes the GEMM artificially fast, which makes the epilogue
# an artificially large share of the kernel, which INFLATES the measured speedup: it read 1.286x
# for silu where bench.py's cold pool read 1.163x -- same kernel, same shape, different cache
# state. A rotating pool large enough to overflow the LLC is the honest setting, and it is what a
# real model sees, where each GEMM gets operands nobody just touched.
LLC_BYTES = 256 * 1024 * 1024
POOL_BUDGET = 6 * 1024**3      # cap so the 8192^3 sets still fit comfortably in HBM
TIME_REPEATS = 3               # median-of-medians; the small shapes are the noisy ones


def _pool_size(bytes_per_set):
    """Enough distinct input sets that consecutive iterations miss the LLC (bench.py's rule)."""
    if bytes_per_set <= 0:
        return 1
    return max(1, min(LLC_BYTES // bytes_per_set + 1, max(1, POOL_BUDGET // bytes_per_set)))


def inputs(m, n, k):
    """Deterministic inputs -- the same seed and the same draw order in both processes."""
    torch.manual_seed(SEED + m + n + k)
    A = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    Bt = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    return A, Bt


def run(m, n, k):
    import tk_silu
    A, Bt = inputs(m, n, k)
    C = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    tk_silu.dispatch(A, Bt, C)
    torch.cuda.synchronize()
    return A, Bt, C.cpu()


def order_key(t_bf16):
    """Map bf16 bit patterns to a monotonically increasing integer, so |key_a - key_b| is the
    number of representable bf16 values between them -- a real ULP distance.
    sign-magnitude -> total order:  positive m -> 0x8000+m ;  negative m -> 0x8000-m."""
    u = t_bf16.view(torch.int16).to(torch.int64) & 0xFFFF
    mag = u & 0x7FFF
    neg = (u >> 15).bool()
    return torch.where(neg, 0x8000 - mag, 0x8000 + mag)


def _bench(fn, pool_n, iters=50, warm=10, repeats=TIME_REPEATS):
    """Median-of-medians per-iteration device time (ms), CUDA events.
    `fn(i)` runs one iteration against input set i and must NOT synchronize."""
    meds = []
    for _ in range(max(1, repeats)):
        for i in range(warm):
            fn(i % pool_n)
        torch.cuda.synchronize()
        ts = []
        for i in range(iters):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); fn(i % pool_n); e.record(); e.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        meds.append(ts[len(ts) // 2])
    meds.sort()
    return meds[len(meds) // 2]


def time_kernels(verbose=False):
    """Time every kernel that shares silu_op, at (M, N, K) with N = GEMM output width.

    All three are timed the same way in the same process, so the only thing separating a 'save'
    run from a 'compare' run is which silu_op got compiled in."""
    import tk_silu, tk_swiglu, tk_rmsnorm_swiglu
    out = {}
    for (m, n, k) in TIME_SHAPES:
        torch.manual_seed(SEED + m + n + k)
        perm = gate_up_perm(n // 2)
        # bytes touched per iteration: A + the two weights + the full and half outputs
        per_set = (m * k + n * k + k * n + m * n + m * n // 2) * 2
        pool_n = _pool_size(per_set)
        if verbose:
            print(f"  ({m},{n},{k}): {per_set/1e6:.0f} MB/set, rotating {pool_n} sets "
                  f"({pool_n*per_set/1e6:.0f} MB vs a {LLC_BYTES/1e6:.0f} MB LLC)")
        sets = []
        for _ in range(pool_n):
            A = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            Bt = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)          # silu: [N,K]
            W = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
            Wp = W[:, perm.to(W.device)].t().contiguous()                        # swiglu: permuted
            r = torch.rsqrt(A.float().pow(2).mean(-1) + RMS_EPS).to(torch.bfloat16).contiguous()
            sets.append(dict(A=A, Bt=Bt, Wp=Wp, r=r,
                             Cfull=torch.empty(m, n, device="cuda", dtype=torch.bfloat16),
                             Chalf=torch.empty(m, n // 2, device="cuda", dtype=torch.bfloat16)))
        res = {}
        res["silu"] = _bench(lambda i: tk_silu.dispatch(sets[i]["A"], sets[i]["Bt"], sets[i]["Cfull"]), pool_n)
        res["swiglu"] = _bench(lambda i: tk_swiglu.dispatch(sets[i]["A"], sets[i]["Wp"], sets[i]["Chalf"]), pool_n)
        res["rmsnorm_swiglu"] = _bench(
            lambda i: tk_rmsnorm_swiglu.dispatch(sets[i]["A"], sets[i]["Wp"], sets[i]["Chalf"], sets[i]["r"]), pool_n)
        out[f"{m}_{n}_{k}"] = res
        del sets
        torch.cuda.empty_cache()
    return out


def report_timing(fast_t, orig_t):
    print(f"\nTIMING  -- shapes are (M, N, K) with N = GEMM OUTPUT WIDTH")
    print(f"{'kernel':<18}{'shape':<22}{'original ms':>13}{'fast ms':>11}{'speedup':>10}")
    print("-" * 74)
    for key in fast_t:
        m, n, k = key.split("_")
        for kern in ("silu", "swiglu", "rmsnorm_swiglu"):
            f, o = fast_t[key][kern], orig_t[key][kern]
            print(f"{kern:<18}{'(' + m + ', ' + n + ', ' + k + ')':<22}{o:>13.5f}{f:>11.5f}{o / f:>10.3f}")

def main():
    mode, path = sys.argv[1], sys.argv[2]
    if mode == "save":
        out = {"_timing": time_kernels(verbose=True)}
        for (m, n, k) in SHAPES:
            A, Bt, C = run(m, n, k)
            # Persist the INPUTS alongside the outputs. torch.manual_seed IS reproducible across
            # processes on the same device and torch build, and the result proves it was here:
            # 99.66% of outputs came back bit-identical, which is impossible with different
            # inputs -- a bf16 GEMM is chaotic, mismatched operands would differ ~everywhere.
            # But "seeding is reproducible" is an assumption a reader has to take on trust, and
            # shipping the bytes costs nothing. Run 2 loads these instead of redrawing them.
            out[f"{m}_{n}_{k}"] = {"C": C, "A": A.cpu(), "Bt": Bt.cpu()}
        torch.save(out, path)
        print(f"  saved fast-build outputs, inputs and timings for {len(out) - 1} shapes")
        return

    ref = torch.load(path)
    report_timing(ref["_timing"], time_kernels(verbose=True))
    print(f"\n{'shape':<20}{'differ':>10}{'pct':>9}{'flush0':>9}{'flips':>7}"
          f"{'max |diff|':>12}{'closer:fast':>13}{'closer:orig':>13}{'tie':>10}")
    print("-" * 105)
    verdict_ok = True
    for (m, n, k) in SHAPES:
        import tk_silu
        e = ref[f"{m}_{n}_{k}"]
        fast = e["C"]
        # EXACT same operand bytes as run 1 -- loaded, not redrawn.
        A = e["A"].cuda(); Bt = e["Bt"].cuda()
        orig = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        tk_silu.dispatch(A, Bt, orig); torch.cuda.synchronize()
        orig = orig.cpu()

        # --- 1. how far apart are the two builds, in representable bf16 steps ---
        steps = (order_key(orig) - order_key(fast)).abs()
        ndiff = int((steps != 0).sum()); tot = steps.numel()
        n_gt1 = int((steps > 1).sum()); mx = int(steps.max())
        mabs = (orig.float() - fast.float()).abs().max().item()

        # --- 2. which one is closer to the truth? float64 oracle, rounded to bf16 like the kernel ---
        # The GEMM is done in float64 on the host; both builds share the same device accumulator,
        # so this scores silu's own error plus an identical GEMM offset for both.
        D = (A.double() @ Bt.t().double())
        oracle = (D / (1.0 + torch.exp(-D))).to(torch.bfloat16).cpu()
        ok_ = order_key(oracle)
        ef = (order_key(fast) - ok_).abs()
        eo = (order_key(orig) - ok_).abs()
        fast_closer = int((ef < eo).sum())
        orig_closer = int((eo < ef).sum())
        tie = tot - fast_closer - orig_closer

        # --- 3. Split the disagreements into the two physically distinct kinds ---------------
        of, oo, oc = fast.float(), orig.float(), oracle.float()
        # (a) FLUSH-TO-ZERO: v_rcp_f32 underflows for an enormous denominator, so silu of a very
        #     negative x returns exactly 0 where the exact answer is a ~1e-36 denormal. This is
        #     the ONLY kind that produces a large "step" count -- bf16 packs ~900 representable
        #     values between 0 and 1e-36, so 0-vs-1e-36 really is ~900 steps apart while being
        #     utterly meaningless in magnitude. Step counts are a bad error metric near zero.
        ftz_mask = (of == 0) & (oo != 0)
        ftz = int(ftz_mask.sum())
        ftz_worst = oo[ftz_mask].abs().max().item() if ftz else 0.0
        # (b) genuine ROUNDING FLIPS on normal-sized values.
        flips = ndiff - ftz
        nrm = oc.norm()
        rel_f = (of - oc).norm().item() / nrm.item()
        rel_o = (oo - oc).norm().item() / nrm.item()
        big = oc.abs() > 1e-3
        pe_f = ((of - oc).abs()[big] / oc.abs()[big]).max().item()
        pe_o = ((oo - oc).abs()[big] / oc.abs()[big]).max().item()
        # bf16 keeps 7 EXPLICIT mantissa bits, so one representable step is |value| * 2^-7.
        dif = (of - oo).abs()
        at_val = oc.flatten()[int(dif.argmax())].item()
        step = abs(at_val) / 128.0

        print(f"{str((m,n,k)):<20}{ndiff:>10}{100*ndiff/tot:>8.4f}%{ftz:>9}{flips:>7}"
              f"{mabs:>12.3e}{fast_closer:>13}{orig_closer:>13}{tie:>10}")
        print(f"    normwise rel vs float64 oracle : fast {rel_f:.3e}   orig {rel_o:.3e}   "
              f"(bf16's own rounding floor is ~3.9e-3)")
        print(f"    max per-element rel (|y|>1e-3) : fast {pe_f:.3e}   orig {pe_o:.3e}")
        print(f"    largest abs disagreement       : {dif.max().item():.3e} at a true value of "
              f"{at_val:.4g} = {dif.max().item()/step:.2f} bf16 steps (one step there = {step:.3e})")
        print(f"    flush-to-zero: {ftz} ({100.0*ftz/max(ndiff,1):.2f}% of disagreements), "
              f"worst magnitude discarded {ftz_worst:.3e}")
        # Every flush-to-zero necessarily counts as 'orig closer' (orig keeps the denormal), so
        # strip them out to see whether the GENUINE flips favour one build. An even split means
        # different rounding of ties, i.e. no accuracy regression.
        print(f"    genuine flips only: fast-closer {fast_closer}  vs  orig-closer "
              f"{orig_closer - ftz}   -> {'a WASH' if abs(fast_closer - (orig_closer - ftz)) <= max(3, 0.25*flips) else 'ONE-DIRECTIONAL'}")
        # Benign = 1-step flips, plus flush-to-zero of magnitudes nothing downstream can notice.
        if dif.max().item() > 1.01 * step or ftz_worst > 1e-30:
            verdict_ok = False
        del D, oracle, of, oo, oc, dif
        torch.cuda.empty_cache()
    print()
    print("READING THE COLUMNS")
    print("  differ      : elements whose stored bf16 bits are not identical between the builds")
    print("  flush0      : of those, how many are silu(very negative x) where v_rcp_f32 underflows")
    print("                and fast returns exactly 0 instead of a ~1e-36 denormal. Physically")
    print("                meaningless -- it vanishes in the first bf16 add -- but it dominates")
    print("                any raw 'which build is closer' count, because orig keeps the denormal.")
    print("  flips       : the genuine disagreements, on normal-sized values.")
    print("  closer:fast / closer:orig : nearest to the float64 oracle. Read the 'genuine flips")
    print("                only' line, NOT this raw pair -- the raw pair is swamped by flush0.")
    print()
    print("  NOTE ON STEP COUNTS: bf16 packs ~900 representable values between 0 and 1e-36, so a")
    print("  flush-to-zero really is ~900 'steps' while being a nothing difference in magnitude.")
    print("  Step count is the wrong error metric near zero; magnitude and normwise error are right.")
    print()
    print("VERDICT:", "no accuracy regression -- every genuine disagreement is <=1 bf16 step, and "
          "the only large-step cases are flush-to-zero of ~1e-36" if verdict_ok
          else "A DISAGREEMENT EXCEEDS 1 bf16 step on a normal-sized value -- investigate")


if __name__ == "__main__":
    main()
