# CODA Epilogue Phase 1 Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the CODA epilogue framework and five epilogues (noop, silu, scale, swiglu, partialrms) on the fork, one owner-reviewed file per commit. Each file must pass a compile gate on this node or a correctness gate on gfx950.

**Architecture:** Each chunk starts from the PR #90 file at `b41ff33f` and is moved to `kernels/cdna4/gemm/bf16fp32/epilogue/`. The importer trims it to the code that already-landed kernels use. The owner reviews it, agreed edits are applied, then it is gated, committed, and pushed. Gate tooling is throwaway: it lives under `/scratch/users/avirgoel/coda-review/` and is never committed.

**Tech Stack:** HIP C++20 / HipKittens (CDNA4), hipcc (HIP 7.1 on this node), pybind11, PyTorch on AUSTIN MI355X via `srun` + pyxis.

**Spec:** `docs/superpowers/specs/2026-10-03-coda-epilogue-incremental-review-design.md`

---

## Reading this plan

- **This plan has one deliberate exception to "complete code in every step."** The spec makes the owner's review the decision point. So each chunk's Step 3 ("apply agreed edits") has no fixed code. Its content is the edit list agreed in Step 2. Every other step is exact: the import command, the trim, the gate source and command, and the commit/push. Where a chunk lands only part of a #90 file, the starting code is spelled out in full.
- **The human checkpoint is Step 2 of every chunk task.** An executor must stop there and wait for the owner. Nothing is committed without the owner's explicit "approve" in Step 6.
- **Gate scripts adapt to review.** If review deletes or renames something a gate source mentions, update the throwaway gate source to match before running it. The gate tests the landing file, not #90.

## Shared variables (every task)

```bash
WT=/home/AMD/avirgoel/rockKE/coda/HipKittens/.worktrees/coda-epilogue-reviewed
EPI=$WT/kernels/cdna4/gemm/bf16fp32/epilogue
PR=fork/users/avirgoel/upstream-forward-epilogue          # PR #90 head b41ff33f
P90=kernels/gemm/bf16fp32/epilogue                         # #90's directory
S=/scratch/users/avirgoel/coda-review                      # throwaway gate area (beegfs)
G=$S/gates
```

Precondition check (run once per session):

```bash
git -C $WT rev-parse --abbrev-ref HEAD            # expect: users/avirgoel/coda-epilogue-reviewed
git -C $WT rev-parse $PR                          # expect: b41ff33fbeeadcc6bafb6c33e5654ef0c01f18b7
```

## Commit message template (every chunk)

```
epilogue: <file> (chunk <N>/21)

Source: PR #90 kernels/gemm/bf16fp32/epilogue/<file> @ b41ff33f
Review changes:
- <change> -- <reason>
Gate: <exact command>
<gate output lines: GATE PASS + resource lines, or the gfx950 PASS lines + node/GPU/image>
```

Push (explicit refspec, never `main`):

```bash
git -C $WT push fork users/avirgoel/coda-epilogue-reviewed
```

## File map (Phase 1 end state)

```
kernels/cdna4/gemm/bf16fp32/epilogue/
  epilogue_args.cuh          constants, gl typedefs, gemm_args_base
  stream.cuh                 per-module launch stream + set_stream/get_stream binding
  gemm_base.cuh              NoOpEpilogue, gemm_kernel<Epilogue,Globals>, launch<> (3 output modes)
  Makefile                   local knobs + include ../../../../common.mk
  .gitignore                 build artifacts
  ops/base.cuh               block_coords, store_C, store_swiglu
  ops/activations.cuh        fast_silu, silu_op
  ops/vec.cuh                apply_scale
  ops/reductions.cuh         partial_row_sum_sq
  epilogues/{silu,scale,swiglu,partialrms}.cuh
  bindings/gemm_{noop,silu,scale,swiglu,partialrms}.cpp
```

---

### Task 0: Gate tooling, AUSTIN probe, draft PR

**Files (throwaway, not in git):**
- Create: `$G/compile_gate.sh`, `$G/check_gfx950.py`, `$G/austin_run.sh`, `$S/pr_body.md`

- [ ] **Step 1: Write the compile gate**

`$G/compile_gate.sh`:

```bash
#!/usr/bin/env bash
# Throwaway compile gate for the CODA epilogue review. Never committed.
# Usage: compile_gate.sh device|full <tu.cpp>
#        compile_gate.sh so <src.cpp> <module_name>
set -uo pipefail
WT=/home/AMD/avirgoel/rockKE/coda/HipKittens/.worktrees/coda-epilogue-reviewed
EPI=$WT/kernels/cdna4/gemm/bf16fp32/epilogue
OUT=/scratch/users/avirgoel/coda-review/gates/out
mkdir -p "$OUT"
MODE=$1; SRC=$2
INC=(-I"$EPI" -I"$EPI/ops" -I"$EPI/epilogues" -I"$WT/include" -I/opt/rocm/include/hip)
FLAGS=(-x hip -std=c++20 -O3 -w -DKITTENS_CDNA4 --offload-arch=gfx950 -Rpass-analysis=kernel-resource-usage)
case "$MODE" in
  device) CMD=(hipcc "${FLAGS[@]}" --cuda-device-only -c "${INC[@]}" "$SRC" -o "$OUT/gate.o") ;;
  full)   CMD=(hipcc "${FLAGS[@]}" -c "${INC[@]}" "$SRC" -o "$OUT/gate.o") ;;
  so)     CMD=(hipcc "${FLAGS[@]}" -DTK_MODULE_NAME="$3" -shared -fPIC "${INC[@]}"
               $(python3 -m pybind11 --includes) "$SRC" -o "$OUT/$3$(python3-config --extension-suffix)") ;;
  *) echo "usage: $0 device|full <tu.cpp> | so <src.cpp> <module>"; exit 2 ;;
esac
echo "+ ${CMD[*]}"
if "${CMD[@]}" > "$OUT/gate.log" 2>&1; then
  grep -E 'Function Name|VGPRs|AGPRs|Occupancy|Spill' "$OUT/gate.log" | sed 's/^.*remark: *//; s/ \[-Rpass.*//'
  echo "GATE PASS"
else
  cat "$OUT/gate.log"; echo "GATE FAIL"; exit 1
fi
```

Run: `mkdir -p $G/tu && chmod +x $G/compile_gate.sh`

- [ ] **Step 2: Self-test the compile gate on upstream's base GEMM**

The gate's include flags don't cover the base GEMM's folder, so this step calls `hipcc` directly with the same flags:

```bash
cd $WT && hipcc -x hip -std=c++20 -O3 -w -DKITTENS_CDNA4 --offload-arch=gfx950 --cuda-device-only -c \
  -Rpass-analysis=kernel-resource-usage -Iinclude -I/opt/rocm/include/hip $(python3 -m pybind11 --includes) \
  kernels/cdna4/gemm/bf16fp32/256_256_64_32_with16x32.cpp -o /tmp/base.o 2>&1 | grep -E 'VGPRs|AGPRs|Occupancy|Spill'; rm -f /tmp/base.o
```

Expected (measured 2026-10-03): `VGPRs: 210`, `AGPRs: 0`, `Occupancy [waves/SIMD]: 2`, both spills `0`. This is the reference for chunk 4.

- [ ] **Step 3: Write the gfx950 check script**

`$G/check_gfx950.py`:

```python
"""Throwaway gfx950 gate for the CODA epilogue review. Never committed.

Usage: python -u check_gfx950.py --kernel {noop,silu,scale,swiglu,partialrms} --build-dir DIR [--base-dir DIR]
"""
import argparse
import importlib
import sys

import torch
import torch.nn.functional as F

SHAPES = [(256, 256, 128), (4096, 4096, 4096), (8192, 2048, 4096), (4096, 4096, 128)]  # (M, N, K)
RTOL = ATOL = 1e-2          # same criterion as torch.testing.assert_close(rtol, atol)
ALPHA = 0.37
DEV = "cuda"
KERNELS = ["noop", "silu", "scale", "swiglu", "partialrms"]


def operands(M, N, K, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    A = (torch.randn(M, K, generator=g, device=DEV) / K ** 0.5).to(torch.bfloat16)
    B = torch.randn(N, K, generator=g, device=DEV).to(torch.bfloat16)   # kernel takes b as [N, K]
    return A, B


def gemm_fp32(A, B):
    return A.float() @ B.float().T


def gate_up_perm(d_ff):
    """PR #90 util/swiglu.py: puts gate[j] and value[j] 128 columns apart in one 256-block."""
    j = torch.arange(d_ff)
    b, c = j // 128, j % 128
    perm = torch.empty(2 * d_ff, dtype=torch.long)
    perm[b * 256 + c] = j
    perm[b * 256 + c + 128] = d_ff + j
    return perm


def run(mod, kernel, A, B):
    """Launch `kernel` on (A, B). Returns (got, ref) as fp32 tensors of the kernel's output shape."""
    M, N = A.shape[0], B.shape[0]
    if kernel == "noop":
        out = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
        mod.dispatch(A, B, out)
        ref = gemm_fp32(A, B)
    elif kernel == "silu":
        out = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
        mod.dispatch(A, B, out)
        ref = F.silu(gemm_fp32(A, B))
    elif kernel == "scale":
        out = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
        mod.dispatch(A, B, out, torch.full((1,), ALPHA, dtype=torch.float32, device=DEV))
        ref = ALPHA * gemm_fp32(A, B)
    elif kernel == "swiglu":
        d_ff = N // 2
        # B's rows are W_gate_up's columns in natural order: [0, d_ff) gate, [d_ff, 2*d_ff) value.
        Bp = B[gate_up_perm(d_ff).to(DEV)].contiguous()
        out = torch.empty(M, d_ff, dtype=torch.bfloat16, device=DEV)
        mod.dispatch(A, Bp, out)
        H = gemm_fp32(A, B)
        ref = F.silu(H[:, :d_ff]) * H[:, d_ff:]
    elif kernel == "partialrms":
        out = torch.zeros(N // 64, M, dtype=torch.float32, device=DEV)
        mod.dispatch(A, B, out)
        sq = gemm_fp32(A, B).square().view(M, N // 256, 8, 32)
        # group grp = col*4 + wc owns 32-col chunks wc and wc+4 of 256-col block `col` (block_coords)
        ref = (sq[:, :, :4, :].sum(-1) + sq[:, :, 4:, :].sum(-1)).reshape(M, N // 64).T
    else:
        raise ValueError(kernel)
    torch.cuda.synchronize()
    return out.float(), ref


def check_accuracy(mod, kernel):
    ok = True
    for i, (M, N, K) in enumerate(SHAPES):
        got, ref = run(mod, kernel, *operands(M, N, K, seed=i))
        good = torch.allclose(got, ref, rtol=RTOL, atol=ATOL)
        print(f"{kernel} {M}x{N}x{K}: max_abs={(got - ref).abs().max().item():.3e} "
              f"{'PASS' if good else 'FAIL'}", flush=True)
        ok &= good
    return ok


def check_rejects(mod, kernel):
    ok = True
    for M, N, K, why in [(384, 256, 128, "M % 256 != 0"), (256, 256, 192, "K % 128 != 0")]:
        try:
            run(mod, kernel, *operands(M, N, K, seed=99))
        except RuntimeError as e:
            print(f"reject {why}: PASS ({e})", flush=True)
        else:
            print(f"reject {why}: FAIL (no error raised)", flush=True)
            ok = False
    return ok


def check_bit_exact(mod):
    base = importlib.import_module("tk_kernel")      # upstream untouched base GEMM
    ok = True
    for i, (M, N, K) in enumerate(SHAPES):
        A, B = operands(M, N, K, seed=i)
        ours = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
        theirs = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
        mod.dispatch(A, B, ours)
        base.dispatch_micro(A, B, theirs)
        torch.cuda.synchronize()
        same = torch.equal(ours, theirs)
        print(f"bit-exact vs base {M}x{N}x{K}: {'PASS' if same else 'FAIL'}", flush=True)
        ok &= same
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", required=True, choices=KERNELS)
    p.add_argument("--build-dir", required=True)
    p.add_argument("--base-dir")
    args = p.parse_args()
    sys.path[:0] = [args.build_dir] + ([args.base_dir] if args.base_dir else [])
    mod = importlib.import_module(f"tk_{args.kernel}")
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name} arch: {props.gcnArchName} torch: {torch.__version__}", flush=True)
    ok = check_accuracy(mod, args.kernel)
    ok &= check_rejects(mod, args.kernel)
    if args.kernel == "noop":
        ok &= check_bit_exact(mod)
    print("GATE PASS" if ok else "GATE FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the AUSTIN runner**

`$G/austin_run.sh`:

```bash
#!/usr/bin/env bash
# Throwaway: stage the review worktree to /scratch, build, run the gfx950 gate on AUSTIN MI355X.
# Usage: austin_run.sh <noop|silu|scale|swiglu|partialrms>
set -euo pipefail
K=$1
S=/scratch/users/avirgoel/coda-review
G=$S/gates
WT=/home/AMD/avirgoel/rockKE/coda/HipKittens/.worktrees/coda-epilogue-reviewed
IMG=/cluster/images/rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0-20260430.sqsh
rm -rf "$S/src" && mkdir -p "$S/src"
tar cf - -C "$WT" --exclude=__pycache__ --exclude='*.so' include kernels/common.mk kernels/cdna4/gemm/bf16fp32 \
  | tar xf - -C "$S/src"
git -C "$WT" rev-parse HEAD > "$S/src/STAGED_COMMIT"
srun --constraint=AUSTIN --gres=gpu:gfx950-mi355x:1 --time=00:20:00 --job-name=coda-gate-$K \
  --container-image="$IMG" --container-mounts=/scratch:/scratch \
  --container-workdir="$S/src/kernels/cdna4/gemm/bf16fp32/epilogue" \
  bash -lc "set -euo pipefail; echo node=\$(hostname) staged=\$(cat $S/src/STAGED_COMMIT) image=$IMG; \
    export PYTHONPATH=$S/pydeps:\${PYTHONPATH:-}; \
    make KERNEL=$K; (cd .. && make); \
    python -u $G/check_gfx950.py --kernel $K --build-dir . --base-dir .."
```

Run: `chmod +x $G/austin_run.sh`

- [ ] **Step 5: Probe AUSTIN (is `/scratch` shared? are hipcc, torch, pybind11 in the image?)**

```bash
IMG=/cluster/images/rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0-20260430.sqsh
srun --constraint=AUSTIN --gres=gpu:gfx950-mi355x:1 --time=00:10:00 --job-name=coda-probe \
  --container-image="$IMG" --container-mounts=/scratch:/scratch \
  bash -lc 'hostname; ls /scratch/users/avirgoel/coda-review/gates/; which hipcc make; hipcc --version | head -2;
            python3 -c "import torch; p=torch.cuda.get_device_properties(0); print(torch.__version__, p.name, p.gcnArchName)";
            python3 -m pybind11 --includes || echo NO_PYBIND11'
```

Expected:
- `ls` shows `austin_run.sh check_gfx950.py compile_gate.sh tu`, which proves `/scratch` is shared.
- `hipcc` is found.
- The torch line contains `gfx950`.

Outcomes:
- If it prints `NO_PYBIND11`, run on the login node: `python3 -m pip install --target $S/pydeps pybind11`. Python 3.12 matches the image. Then re-run the probe with `export PYTHONPATH=$S/pydeps;` prefixed and expect include paths.
- If `ls` fails, `/scratch` is node-local. Stop and report to the owner. The October fallback is the kernel_factory transport tool (vault `runbook-austin-gpu-verify`).

- [ ] **Step 6: Push the branch and open the draft PR**

`$S/pr_body.md`:

```markdown
Reviewed, file-by-file rebuild of HazyResearch/HipKittens#90 (CODA GEMM-epilogue fusion, forward only).
Spec: `docs/superpowers/specs/2026-10-03-coda-epilogue-incremental-review-design.md`.
Each commit = one owner-reviewed file + its gate evidence. Draft until Phases 1-3 are done.

## Phase 1 checklist
- [ ] 1 epilogue_args.cuh
- [ ] 2 stream.cuh
- [ ] 3 ops/base.cuh (block_coords, store_C)
- [ ] 4 gemm_base.cuh (store-only launch)
- [ ] 5 bindings/gemm_noop.cpp
- [ ] 6 Makefile -- gfx950: noop
- [ ] 7 .gitignore
- [ ] 8 ops/activations.cuh
- [ ] 9 epilogues/silu.cuh
- [ ] 10 bindings/gemm_silu.cpp -- gfx950: silu
- [ ] 11 ops/vec.cuh (apply_scale)
- [ ] 12 epilogues/scale.cuh
- [ ] 13 bindings/gemm_scale.cpp -- gfx950: scale
- [ ] 14 ops/base.cuh + store_swiglu
- [ ] 15 gemm_base.cuh + out_cols branch
- [ ] 16 epilogues/swiglu.cuh
- [ ] 17 bindings/gemm_swiglu.cpp -- gfx950: swiglu
- [ ] 18 ops/reductions.cuh
- [ ] 19 gemm_base.cuh + partials-only branch
- [ ] 20 epilogues/partialrms.cuh
- [ ] 21 bindings/gemm_partialrms.cpp -- gfx950: partialrms

## Deviations from #90
| File | Change | Reason |
|---|---|---|
| (all) | moved `kernels/gemm/bf16fp32/epilogue/` -> `kernels/cdna4/gemm/bf16fp32/epilogue/` | upstream reorganized into `kernels/cdna{3,4,5}/` |
```

```bash
git -C $WT push fork users/avirgoel/coda-epilogue-reviewed
gh pr create -R AviralGoelAMD/HipKittens --draft --base main --head users/avirgoel/coda-epilogue-reviewed \
  --title "CODA epilogue: reviewed rebuild of HazyResearch/HipKittens#90" --body-file $S/pr_body.md
```

Expected: a PR URL. Record its number as `$FPR` for later `gh pr edit $FPR -R AviralGoelAMD/HipKittens --body-file $S/pr_body.md` updates.

---

### Task 1: `epilogue_args.cuh` (chunk 1, compile gate)

**Files:** Create `$EPI/epilogue_args.cuh`. Throwaway: `$G/tu/c01.cpp`.

- [ ] **Step 1: Import**

```bash
mkdir -p $EPI/ops $EPI/epilogues $EPI/bindings
git -C $WT show $PR:$P90/epilogue_args.cuh > $EPI/epilogue_args.cuh
```

Trim (only-what's-used rule): delete the `RMS_EPS` line. No Phase 1 file uses it; it returns with the RMS epilogues.

- [ ] **Step 2: Review session (owner checkpoint).** Present the file and its findings, then wait. Known talking points:
  - The 2×2 fan-out is explained in three places: the `SUBTILES_PER_DIM` trailing comment, the `static_assert` message, and later `ops/base.cuh`.
  - Trailing whitespace after `BLOCK_SIZE = 256;`.
  - `#define NUM_WARPS`/`NUM_THREADS` macros sit next to `constexpr` constants.
  - `using namespace kittens;` in a header.
  - The comment on `gemm_args_base` mentions epilogues that haven't landed yet.
- [ ] **Step 3: Apply the agreed edits** to `$EPI/epilogue_args.cuh`.
- [ ] **Step 4: Gate**

`$G/tu/c01.cpp`:

```cpp
#include "epilogue_args.cuh"
static_assert(BLOCK_SIZE == 256 && HALF_BLOCK_SIZE == 128 && K_STEP == 64);
static_assert(WARPS_M == 2 && WARPS_N == 4 && NUM_WARPS == 8 && NUM_THREADS == 512);
static_assert(REG_BLOCK_M == 128 && REG_BLOCK_N == 64);
static_assert(HALF_REG_BLOCK_M == 64 && HALF_REG_BLOCK_N == 32);
static_assert(SUBTILES_PER_DIM == 2 && K_ALIGN == 128);
__global__ void gate_args(gemm_args_base g) { (void)g; }
```

Run: `$G/compile_gate.sh device $G/tu/c01.cpp`. Expected: `GATE PASS`.

- [ ] **Step 5: Final diff.** Show `cat $EPI/epilogue_args.cuh` and the gate output to the owner.
- [ ] **Step 6: On "approve": commit and push**

```bash
git -C $WT add kernels/cdna4/gemm/bf16fp32/epilogue/epilogue_args.cuh
git -C $WT commit    # template above, chunk 1/21
git -C $WT push fork users/avirgoel/coda-epilogue-reviewed
```

Tick chunk 1 in `$S/pr_body.md`, add the deviation rows, then run `gh pr edit $FPR -R AviralGoelAMD/HipKittens --body-file $S/pr_body.md`.

---

### Task 2: `stream.cuh` (chunk 2, compile gate + host round-trip)

**Files:** Create `$EPI/stream.cuh`. Throwaway: `$G/tu/c02.cpp`.

- [ ] **Step 1: Import**

```bash
git -C $WT show $PR:$P90/stream.cuh > $EPI/stream.cuh
```

- [ ] **Step 2: Review session (owner checkpoint).** Talking points:
  - Does `stream.cuh` belong in Phase 1 at all? If dropped, follow the spec note: `launch` uses the null stream, the bindings omit `hkstream::bind`, and Steps 3–6 become "delete the file; skip the chunk".
  - The 20-line "WHY" comment records #90's review history (the "11 structs" remark).
  - A function-local `static` stream is per shared object, so each `tk_*` module has its own stream.
- [ ] **Step 3: Apply the agreed edits.**
- [ ] **Step 4: Gate**

`$G/tu/c02.cpp`:

```cpp
#include <pybind11/pybind11.h>
#include "stream.cuh"
PYBIND11_MODULE(gate_stream, m) { hkstream::bind(m); }
```

```bash
$G/compile_gate.sh so $G/tu/c02.cpp gate_stream
python3 -c "import sys; sys.path.insert(0,'$G/out'); import gate_stream as s; \
assert s.get_stream()==0; s.set_stream(4096); assert s.get_stream()==4096; print('stream round-trip OK')"
```

Expected: `GATE PASS`, then `stream round-trip OK`.

- [ ] **Step 5: Final diff** to the owner.
- [ ] **Step 6: On "approve": commit and push.** `git -C $WT add kernels/cdna4/gemm/bf16fp32/epilogue/stream.cuh`, commit (chunk 2/21), push, then update the PR body.

---

### Task 3: `ops/base.cuh` — `block_coords`, `store_C` (chunk 3, compile gate)

**Files:** Create `$EPI/ops/base.cuh`. Throwaway: `$G/tu/c03.cpp`.

- [ ] **Step 1: Import**

```bash
git -C $WT show $PR:$P90/ops/base.cuh > $EPI/ops/base.cuh
```

Trim: delete the `store_swiglu` function template and its 4-line comment above it. It returns in chunk 14.

- [ ] **Step 2: Review session (owner checkpoint).** Talking points:
  - The 6-line 2×2 comment duplicates `epilogue_args.cuh`.
  - The `block_coords` arithmetic, worked through with numbers: row `r`, warp row `wr` → subtile rows `2r·2+wr` and `2r·2+2+wr`, in 64-row units.
  - `using namespace kittens;` in a header.
- [ ] **Step 3: Apply the agreed edits.**
- [ ] **Step 4: Gate**

`$G/tu/c03.cpp`:

```cpp
#include "base.cuh"
using Accum = rt_fl<HALF_REG_BLOCK_M, HALF_REG_BLOCK_N, col_l, rt_16x16_s>[2][2];
__global__ void gate_store_C(const gemm_args_base g, int row, int col, int wr, int wc) {
    Accum C;
    for (auto& r : C) for (auto& t : r) zero(t);
    store_C(g, C, row, col, wr, wc);
}
```

Run: `$G/compile_gate.sh device $G/tu/c03.cpp`. Expected: `GATE PASS` with 0 spills.

- [ ] **Step 5: Final diff** to the owner.
- [ ] **Step 6: On "approve": commit and push.** `git -C $WT add kernels/cdna4/gemm/bf16fp32/epilogue/ops/base.cuh`, commit (chunk 3/21), push, then update the PR body.

---

### Task 4: `gemm_base.cuh` — store-only (chunk 4, compile gate; multi-session review)

**Files:** Create `$EPI/gemm_base.cuh`. Throwaway: `$G/tu/c04.cpp`.

- [ ] **Step 1: Import**

```bash
git -C $WT show $PR:$P90/gemm_base.cuh > $EPI/gemm_base.cuh
```

Trim: replace the whole `launch` function, including its comment block, with this store-only starting point:

```cpp
// Launch path for store epilogues: fail-loud shape preconditions + checked HIP calls.
template<typename Epilogue, typename Globals>
void launch(Globals g, hipStream_t stream = hkstream::cur()) {
    const int M = g.a.rows(), N = g.b.rows(), K = g.a.cols();   // a = [M,K], b = [N,K]
    if (M % BLOCK_SIZE || N % BLOCK_SIZE)
        throw std::runtime_error("GEMM: M and N must be multiples of BLOCK_SIZE (256)");
    if (K % K_ALIGN)
        throw std::runtime_error("GEMM: K must be a multiple of 128");
    if (g.b.cols() != K)
        throw std::runtime_error("GEMM: operand shape mismatch (need a=[M,K], b=[N,K])");
    if (g.c.rows() != M || g.c.cols() != N)
        throw std::runtime_error("GEMM: output c shape mismatch (need c=[M,N])");
    const size_t mem = MAX_SHARED_MEMORY;
    CHECK_CUDA_ERROR(hipFuncSetAttribute((void*)gemm_kernel<Epilogue, Globals>, hipFuncAttributeMaxDynamicSharedMemorySize, mem));
    gemm_kernel<Epilogue, Globals><<<dim3((N / BLOCK_SIZE) * (M / BLOCK_SIZE)), dim3(NUM_THREADS), mem, stream>>>(g, M, N, K);
    CHECK_CUDA_ERROR(hipGetLastError());
}
```

If chunk 2 dropped `stream.cuh`, also remove `#include "stream.cuh"` and change the default to `hipStream_t stream = 0`.

- [ ] **Step 2: Review session (owner checkpoint; may span several sessions).** Review it as a diff against upstream's base GEMM:

```bash
git -C $WT diff --no-index -- kernels/cdna4/gemm/bf16fp32/256_256_64_32_with16x32.cpp \
  kernels/cdna4/gemm/bf16fp32/epilogue/gemm_base.cuh
```

Sessions:
- (a) The mainloop differences. The earlier measurement found about 23 differing lines in the loop body.
- (b) The `Epilogue::apply` hook and the `static_assert` contract.
- (c) `launch`.

Required check: are upstream fixes made after #90 copied the loop (for example #88's int64 index conversion, `fb91a990`) present in the copy? Run `git -C $WT log --oneline be1c9184 -- kernels/cdna4/gemm/bf16fp32/256_256_64_32_with16x32.cpp include/` and diff the indexing code.

- [ ] **Step 3: Apply the agreed edits.**
- [ ] **Step 4: Gate**

`$G/tu/c04.cpp`:

```cpp
#include "gemm_base.cuh"
template __global__ void gemm_kernel<NoOpEpilogue, gemm_args_base>(const gemm_args_base, int, int, int);
template void launch<NoOpEpilogue, gemm_args_base>(gemm_args_base, hipStream_t);
```

```bash
$G/compile_gate.sh device $G/tu/c04.cpp
$G/compile_gate.sh full   $G/tu/c04.cpp
```

Expected: both `GATE PASS`. The device run must show `VGPRs: 210`, `AGPRs: 0`, `Occupancy [waves/SIMD]: 2`, and 0 spills, matching upstream `micro_tk` (Task 0 Step 2). Any difference is a review finding that must be explained before approval.

- [ ] **Step 5: Final diff** to the owner (re-run the Step 2 `diff --no-index`).
- [ ] **Step 6: On "approve": commit and push.** `git -C $WT add kernels/cdna4/gemm/bf16fp32/epilogue/gemm_base.cuh`, commit (chunk 4/21), push, then update the PR body.

---

### Task 5: `bindings/gemm_noop.cpp` (chunk 5, `.so` build gate)

**Files:** Create `$EPI/bindings/gemm_noop.cpp`.

- [ ] **Step 1: Import**

```bash
git -C $WT show $PR:$P90/bindings/gemm_noop.cpp > $EPI/bindings/gemm_noop.cpp
```

- [ ] **Step 2: Review session (owner checkpoint).** Talking points:
  - The file has no trailing newline.
  - `py::bind_function` argument order must match the field order of `gemm_args_base`.
  - What `TK_MODULE_NAME` is, and who defines it (the Makefile, in chunk 6).
- [ ] **Step 3: Apply the agreed edits.**
- [ ] **Step 4: Gate**

```bash
$G/compile_gate.sh so $EPI/bindings/gemm_noop.cpp tk_noop
python3 -c "import sys; sys.path.insert(0,'$G/out'); import tk_noop; print(tk_noop.dispatch.__doc__); print('stream', tk_noop.get_stream())"
```

Expected: `GATE PASS`, a `dispatch(...)` signature with three arguments, and `stream 0`. Drop the `get_stream` part if chunk 2 removed streams.

- [ ] **Step 5: Final diff** to the owner.
- [ ] **Step 6: On "approve": commit and push.** `git -C $WT add kernels/cdna4/gemm/bf16fp32/epilogue/bindings/gemm_noop.cpp`, commit (chunk 5/21), push, then update the PR body.

---

### Task 6: `Makefile` (chunk 6, first gfx950 gate: noop)

**Files:** Create `$EPI/Makefile`.

- [ ] **Step 1: Starting point.** #90's 54-line Makefile predates `kernels/common.mk`, so the starting point is this rewrite. Show #90's version next to it with `git -C $WT show $PR:$P90/Makefile`.

```make
# Build one epilogue binding as a Python extension: make KERNEL=<noop|silu|...>
GPU_TARGET ?= CDNA4
KERNEL     ?= noop
TARGET     ?= tk_$(KERNEL)
SRC        := bindings/gemm_$(KERNEL).cpp
PYTHON     ?= python3

EXTRA_CPPFLAGS += -I. -Iops -Iepilogues
EXTRA_HIPFLAGS += -DTK_MODULE_NAME=$(TARGET)

include ../../../../common.mk
```

- [ ] **Step 2: Review session (owner checkpoint).** Talking points:
  - #90's `-MMD` header-dependency tracking is gone. With it gone, `make` won't rebuild after a header-only edit. Options: `make clean` habit, or `EXTRA_ICXXFLAGS += -MMD -MP` plus `-include`.
  - `common.mk` builds at `-O3` (`COMP_LEVEL=profile`). #90's Makefile set no `-O` flag.
  - `clean` in `common.mk` removes `$(TARGET).*so` only for the current `KERNEL`.
- [ ] **Step 3: Apply the agreed edits.**
- [ ] **Step 4a: Local build gate**

```bash
cd $EPI && make KERNEL=noop && ls tk_noop*.so && make KERNEL=noop clean
```

Expected: the build succeeds, the remark shows `VGPRs: 210`, and the `.so` is listed, then removed.

- [ ] **Step 4b: gfx950 gate**

```bash
$G/austin_run.sh noop 2>&1 | tee $S/gate-c06.log
```

Expected:
- `node=smci355-...`, a staged commit, and the image line.
- `device: AMD Instinct MI355X arch: gfx950...`.
- Four `noop <shape>: max_abs=... PASS` lines.
- Two `reject ...: PASS` lines.
- Four `bit-exact vs base ...: PASS` lines.
- `GATE PASS`.

- [ ] **Step 5: Final diff + gate log** to the owner.
- [ ] **Step 6: On "approve": commit and push.** `git -C $WT add kernels/cdna4/gemm/bf16fp32/epilogue/Makefile`, commit (chunk 6/21, paste the gate log lines), push, then update the PR body.

---

### Task 7: `.gitignore` (chunk 7)

**Files:** Create `$EPI/.gitignore`.

- [ ] **Step 1: Import**

```bash
git -C $WT show $PR:$P90/.gitignore > $EPI/.gitignore
```

- [ ] **Step 2: Review session (owner checkpoint).** Talking points: most patterns are for artifacts this build never produces, such as `*.hipfb`, `results_*.json`, `workloads/`, and `LAYOUT_NOTES.md`. The build output is `tk_*.cpython-*.so` plus `__pycache__/`.
- [ ] **Step 3: Apply the agreed edits.**
- [ ] **Step 4: Gate (artifacts are ignored)**

```bash
cd $EPI && make KERNEL=noop && git -C $WT status --porcelain; make KERNEL=noop clean
```

Expected: `git status --porcelain` prints only `?? kernels/cdna4/gemm/bf16fp32/epilogue/.gitignore` and no `.so`.

- [ ] **Step 5: Final diff** to the owner.
- [ ] **Step 6: On "approve": commit and push.** `git -C $WT add kernels/cdna4/gemm/bf16fp32/epilogue/.gitignore`, commit (chunk 7/21), push, then update the PR body.

---

### Task 8: `ops/activations.cuh` (chunk 8, compile gate)

**Files:** Create `$EPI/ops/activations.cuh`. Throwaway: `$G/tu/c08.cpp`.

- [ ] **Step 1: Import**

```bash
git -C $WT show $PR:$P90/ops/activations.cuh > $EPI/ops/activations.cuh
```

- [ ] **Step 2: Review session (owner checkpoint).** Talking points:
  - The `x * rcp(1 + exp2(-x·log2e))` math, and its accuracy claim ("~1 ULP").
  - Behavior when `x` is very negative (`rcp` underflows to 0) and very positive.
  - `#include <type_traits>` is unused.
  - The `coda_ops` namespace versus a global `silu_op`.
- [ ] **Step 3: Apply the agreed edits.**
- [ ] **Step 4: Gate**

`$G/tu/c08.cpp`:

```cpp
#include "activations.cuh"
using Tile = rt_fl<HALF_REG_BLOCK_M, HALF_REG_BLOCK_N, col_l, rt_16x16_s>;
__global__ void gate_silu(float* out, float x) {
    out[0] = coda_ops::fast_silu::op<float>(x);
    Tile t; zero(t); silu_op(t);
    out[1] = t.tiles[0][0].data[0].x;
}
```

Run: `$G/compile_gate.sh device $G/tu/c08.cpp`. Expected: `GATE PASS`. If `t.tiles[0][0].data[0].x` doesn't compile against this HK version, replace that line with `(void)t;`. The numeric check happens in chunk 10.

- [ ] **Step 5: Final diff** to the owner.
- [ ] **Step 6: On "approve": commit and push.** `git -C $WT add kernels/cdna4/gemm/bf16fp32/epilogue/ops/activations.cuh`, commit (chunk 8/21), push, then update the PR body.

---

### Task 9: `epilogues/silu.cuh` (chunk 9, compile gate)

**Files:** Create `$EPI/epilogues/silu.cuh`. Throwaway: `$G/tu/c09.cpp`.

- [ ] **Step 1: Import**

```bash
git -C $WT show $PR:$P90/epilogues/silu.cuh > $EPI/epilogues/silu.cuh
```

- [ ] **Step 2: Review session (owner checkpoint).** Talking points: the four explicit `silu_op(C[i][j])` calls versus a loop, and the comment's claim "No extra inputs -> gemm_args_base".
- [ ] **Step 3: Apply the agreed edits.**
- [ ] **Step 4: Gate**

`$G/tu/c09.cpp`:

```cpp
#include "gemm_base.cuh"
#include "silu.cuh"
template __global__ void gemm_kernel<SiluEpilogue, gemm_args_base>(const gemm_args_base, int, int, int);
template void launch<SiluEpilogue, gemm_args_base>(gemm_args_base, hipStream_t);
```

```bash
$G/compile_gate.sh device $G/tu/c09.cpp && $G/compile_gate.sh full $G/tu/c09.cpp
```

Expected: `GATE PASS` twice, with 0 spills. Record VGPRs in the commit.

- [ ] **Step 5: Final diff** to the owner.
- [ ] **Step 6: On "approve": commit and push.** `git -C $WT add kernels/cdna4/gemm/bf16fp32/epilogue/epilogues/silu.cuh`, commit (chunk 9/21), push, then update the PR body.

---

### Task 10: `bindings/gemm_silu.cpp` (chunk 10, gfx950 gate: silu)

**Files:** Create `$EPI/bindings/gemm_silu.cpp`.

- [ ] **Step 1: Import**

```bash
git -C $WT show $PR:$P90/bindings/gemm_silu.cpp > $EPI/bindings/gemm_silu.cpp
```

- [ ] **Step 2: Review session (owner checkpoint).** Compare it with the landed `gemm_noop.cpp`. Apply any convention agreed in chunk 5.
- [ ] **Step 3: Apply the agreed edits.**
- [ ] **Step 4: Gate**

```bash
cd $EPI && make KERNEL=silu && make KERNEL=silu clean
$G/austin_run.sh silu 2>&1 | tee $S/gate-c10.log
```

Expected: the local build succeeds. On AUSTIN: four `silu <shape>: ... PASS` lines, two `reject ...: PASS` lines, and `GATE PASS`.

- [ ] **Step 5: Final diff + gate log** to the owner.
- [ ] **Step 6: On "approve": commit and push.** `git -C $WT add kernels/cdna4/gemm/bf16fp32/epilogue/bindings/gemm_silu.cpp`, commit (chunk 10/21, include the gate log), push, then update the PR body.

---

### Task 11: `ops/vec.cuh` — `apply_scale` only (chunk 11, compile gate)

**Files:** Create `$EPI/ops/vec.cuh`. Throwaway: `$G/tu/c11.cpp`.

- [ ] **Step 1: Import**

```bash
git -C $WT show $PR:$P90/ops/vec.cuh > $EPI/ops/vec.cuh
```

Trim: delete `apply_inv_rms` and `apply_gamma` (both function templates) and the header comment lines describing per-row and per-column vectors. Keep `#pragma once`, the includes, `using namespace kittens;`, and `apply_scale` with its comment. Since `<type_traits>` was only used by the deleted functions, delete that include too.

- [ ] **Step 2: Review session (owner checkpoint).** Talking points:
  - `apply_scale` reads `g.alpha[{0,0,0,0}]` from global memory once per thread. Is that 512 loads per block? Compare with a kernel argument.
  - The signature differs from the other ops: no `row/col/wr/wc`.
- [ ] **Step 3: Apply the agreed edits.**
- [ ] **Step 4: Gate**

`$G/tu/c11.cpp`:

```cpp
#include "vec.cuh"
struct AlphaG { gl<float,1,1,1,1> alpha{nullptr,nullptr,nullptr,nullptr,nullptr}; };
using Accum = rt_fl<HALF_REG_BLOCK_M, HALF_REG_BLOCK_N, col_l, rt_16x16_s>[2][2];
__global__ void gate_scale(const AlphaG g, const gemm_args_base o) {
    Accum C;
    for (auto& r : C) for (auto& t : r) zero(t);
    apply_scale(g, C);
    store_C(o, C, 0, 0, 0, 0);
}
```

Run: `$G/compile_gate.sh device $G/tu/c11.cpp`. Expected: `GATE PASS`.

- [ ] **Step 5: Final diff** to the owner.
- [ ] **Step 6: On "approve": commit and push.** `git -C $WT add kernels/cdna4/gemm/bf16fp32/epilogue/ops/vec.cuh`, commit (chunk 11/21), push, then update the PR body.

---

### Task 12: `epilogues/scale.cuh` (chunk 12, compile gate)

**Files:** Create `$EPI/epilogues/scale.cuh`. Throwaway: `$G/tu/c12.cpp`.

- [ ] **Step 1: Import**

```bash
git -C $WT show $PR:$P90/epilogues/scale.cuh > $EPI/epilogues/scale.cuh
```

- [ ] **Step 2: Review session (owner checkpoint).** Talking points:
  - `ScaleGlobals` repeats `a, b, c` instead of reusing `gemm_args_base`. #90 did this on purpose for flat pybind binding.
  - The `alpha{nullptr,...}` default member initializer.
- [ ] **Step 3: Apply the agreed edits.**
- [ ] **Step 4: Gate**

`$G/tu/c12.cpp`:

```cpp
#include "gemm_base.cuh"
#include "scale.cuh"
template __global__ void gemm_kernel<ScaleEpilogue, ScaleGlobals>(const ScaleGlobals, int, int, int);
template void launch<ScaleEpilogue, ScaleGlobals>(ScaleGlobals, hipStream_t);
```

```bash
$G/compile_gate.sh device $G/tu/c12.cpp && $G/compile_gate.sh full $G/tu/c12.cpp
```

Expected: `GATE PASS` twice, with 0 spills.

- [ ] **Step 5: Final diff** to the owner.
- [ ] **Step 6: On "approve": commit and push.** `git -C $WT add kernels/cdna4/gemm/bf16fp32/epilogue/epilogues/scale.cuh`, commit (chunk 12/21), push, then update the PR body.

---

### Task 13: `bindings/gemm_scale.cpp` (chunk 13, gfx950 gate: scale)

**Files:** Create `$EPI/bindings/gemm_scale.cpp`.

- [ ] **Step 1: Import**

```bash
git -C $WT show $PR:$P90/bindings/gemm_scale.cpp > $EPI/bindings/gemm_scale.cpp
```

- [ ] **Step 2: Review session (owner checkpoint).** Talking point: the Python caller must pass `alpha` as a 1-element fp32 CUDA tensor. Is that documented anywhere a caller would see it?
- [ ] **Step 3: Apply the agreed edits.**
- [ ] **Step 4: Gate**

```bash
cd $EPI && make KERNEL=scale && make KERNEL=scale clean
$G/austin_run.sh scale 2>&1 | tee $S/gate-c13.log
```

Expected: four `scale <shape>: ... PASS` lines, two `reject ...: PASS` lines, and `GATE PASS`.

- [ ] **Step 5: Final diff + gate log** to the owner.
- [ ] **Step 6: On "approve": commit and push.** `git -C $WT add kernels/cdna4/gemm/bf16fp32/epilogue/bindings/gemm_scale.cpp`, commit (chunk 13/21, include the gate log), push, then update the PR body.

---

### Task 14: `ops/base.cuh` + `store_swiglu` (chunk 14, compile gate)

**Files:** Modify `$EPI/ops/base.cuh` (append). Throwaway: `$G/tu/c14.cpp`.

- [ ] **Step 1: Starting point.** Append #90's `store_swiglu` and its comment to the end of `$EPI/ops/base.cuh`:

```cpp
// Half-width store for the dim-reducing SwiGLU epilogue. The reduced result lives in the gate
// sub-tiles C[*][0] (block-col co.n[0]); the partner value sub-tile C[*][1] (co.n[0]+128) is
// already consumed. Output is [M, d_ff]: the gate sub-tile at the gate half of its 256-block maps
// to output col b*128+c (natural feature order). In HALF_REG_BLOCK_N sub-tile units:
template<typename Globals, typename Accum>
__device__ inline void store_swiglu(const Globals& g, const Accum& C, int row,int col,int wr,int wc){
    constexpr int NSUB_BLOCK = BLOCK_SIZE / HALF_REG_BLOCK_N;        // 8 sub-tiles span one 256 block
    constexpr int NSUB_HALF  = (BLOCK_SIZE / 2) / HALF_REG_BLOCK_N;  // 4 sub-tiles span the 128 gate half
    subtile_coords co = block_coords(row,col,wr,wc);
    int o0 = (co.n[0] / NSUB_BLOCK) * NSUB_HALF + (co.n[0] % NSUB_BLOCK);
    store(g.c, C[0][0], {0,0,co.m[0], o0});
    store(g.c, C[1][0], {0,0,co.m[1], o0});
}
```

Apply any conventions agreed for `store_C` in chunk 3 (comment style, naming).

- [ ] **Step 2: Review session (owner checkpoint).** Talking points:
  - Prove `o0` with numbers. `co.n[0] = col·8 + wc` with `wc ∈ [0,4)`, so `co.n[0] % 8 = wc < 4`, and the output sub-tile is `col·4 + wc`. Column `(col·4+wc)·32` of `[M, N/2]` is correct.
  - `(BLOCK_SIZE / 2)` duplicates `HALF_BLOCK_SIZE`.
- [ ] **Step 3: Apply the agreed edits.**
- [ ] **Step 4: Gate**

`$G/tu/c14.cpp`:

```cpp
#include "base.cuh"
using Accum = rt_fl<HALF_REG_BLOCK_M, HALF_REG_BLOCK_N, col_l, rt_16x16_s>[2][2];
__global__ void gate_store_swiglu(const gemm_args_base g, int row, int col, int wr, int wc) {
    Accum C;
    for (auto& r : C) for (auto& t : r) zero(t);
    store_swiglu(g, C, row, col, wr, wc);
}
```

Run: `$G/compile_gate.sh device $G/tu/c14.cpp`. Expected: `GATE PASS`. Also re-run `$G/compile_gate.sh device $G/tu/c03.cpp` to confirm `store_C` still compiles.

- [ ] **Step 5: Final diff** (`git -C $WT diff`) to the owner.
- [ ] **Step 6: On "approve": commit and push.** `git -C $WT add kernels/cdna4/gemm/bf16fp32/epilogue/ops/base.cuh`, commit (chunk 14/21, subject `epilogue: ops/base.cuh: add store_swiglu (chunk 14/21)`), push, then update the PR body.

---

### Task 15: `gemm_base.cuh` + `out_cols` branch (chunk 15, compile gate)

**Files:** Modify `$EPI/gemm_base.cuh` (`launch`). Throwaway: `$G/tu/c15.cpp`.

- [ ] **Step 1: Starting point.** In `launch`, replace the store-only check

```cpp
    if (g.c.rows() != M || g.c.cols() != N)
        throw std::runtime_error("GEMM: output c shape mismatch (need c=[M,N])");
```

with #90's optional-trait form:

```cpp
    int expect_cols = N;                                        // dim-reducing epilogues narrow it
    if constexpr (requires { Epilogue::out_cols(N); }) expect_cols = Epilogue::out_cols(N);
    if (g.c.rows() != M || g.c.cols() != expect_cols)
        throw std::runtime_error("GEMM: output c shape mismatch (need c=[M, out_cols(N)])");
```

- [ ] **Step 2: Review session (owner checkpoint).** Talking points:
  - A trait detected through `requires` silently does nothing if it's misspelled. For example, `out_col` instead of `out_cols` falls back to `N` and fails at runtime with a shape error, not at compile time.
  - Alternatives: a required `out_cols` on every epilogue, or a `static_assert`.
- [ ] **Step 3: Apply the agreed edits.**
- [ ] **Step 4: Gate**

`$G/tu/c15.cpp`:

```cpp
#include "gemm_base.cuh"
struct HalfWidth {
    static constexpr int out_cols(int n) { return n / 2; }
    template<typename G, typename A>
    static __device__ inline void apply(const G& g, A& C, int r, int c, int wr, int wc) { store_swiglu(g, C, r, c, wr, wc); }
};
static_assert(HalfWidth::out_cols(512) == 256);
template void launch<HalfWidth, gemm_args_base>(gemm_args_base, hipStream_t);
template void launch<NoOpEpilogue, gemm_args_base>(gemm_args_base, hipStream_t);
```

```bash
$G/compile_gate.sh full $G/tu/c15.cpp && $G/compile_gate.sh device $G/tu/c04.cpp
```

Expected: `GATE PASS` twice. The c04 re-run must still show `VGPRs: 210`, 0 spills.

- [ ] **Step 5: Final diff** (`git -C $WT diff`) to the owner.
- [ ] **Step 6: On "approve": commit and push.** `git -C $WT add kernels/cdna4/gemm/bf16fp32/epilogue/gemm_base.cuh`, commit (chunk 15/21, subject `epilogue: gemm_base.cuh: dim-reducing launch (chunk 15/21)`), push, then update the PR body.

---

### Task 16: `epilogues/swiglu.cuh` (chunk 16, compile gate)

**Files:** Create `$EPI/epilogues/swiglu.cuh`. Throwaway: `$G/tu/c16.cpp`.

- [ ] **Step 1: Import**

```bash
git -C $WT show $PR:$P90/epilogues/swiglu.cuh > $EPI/epilogues/swiglu.cuh
```

- [ ] **Step 2: Review session (owner checkpoint).** Talking points:
  - The weight-permutation contract: the caller must permute `W_gate_up`'s columns with `gate_up_perm`. Nothing in C++ enforces or documents that for a caller.
  - `d_ff % 128 == 0` follows from `N % 256 == 0`.
  - The comment says "Path A", which is #90 history.
- [ ] **Step 3: Apply the agreed edits.**
- [ ] **Step 4: Gate**

`$G/tu/c16.cpp`:

```cpp
#include "gemm_base.cuh"
#include "swiglu.cuh"
template __global__ void gemm_kernel<SwigluEpilogue, gemm_args_base>(const gemm_args_base, int, int, int);
template void launch<SwigluEpilogue, gemm_args_base>(gemm_args_base, hipStream_t);
static_assert(SwigluEpilogue::out_cols(512) == 256);
```

```bash
$G/compile_gate.sh device $G/tu/c16.cpp && $G/compile_gate.sh full $G/tu/c16.cpp
```

Expected: `GATE PASS` twice, with 0 spills.

- [ ] **Step 5: Final diff** to the owner.
- [ ] **Step 6: On "approve": commit and push.** `git -C $WT add kernels/cdna4/gemm/bf16fp32/epilogue/epilogues/swiglu.cuh`, commit (chunk 16/21), push, then update the PR body.

---

### Task 17: `bindings/gemm_swiglu.cpp` (chunk 17, gfx950 gate: swiglu)

**Files:** Create `$EPI/bindings/gemm_swiglu.cpp`.

- [ ] **Step 1: Import**

```bash
git -C $WT show $PR:$P90/bindings/gemm_swiglu.cpp > $EPI/bindings/gemm_swiglu.cpp
```

- [ ] **Step 2: Review session (owner checkpoint).** Talking point: the docstring says "requires gate_up_perm'd weight", but `gate_up_perm` is a Python helper that doesn't exist in this PR until Phase 2. Decide where the permutation contract is documented now.
- [ ] **Step 3: Apply the agreed edits.**
- [ ] **Step 4: Gate**

```bash
cd $EPI && make KERNEL=swiglu && make KERNEL=swiglu clean
$G/austin_run.sh swiglu 2>&1 | tee $S/gate-c17.log
```

Expected: four `swiglu <shape>: ... PASS` lines, two `reject ...: PASS` lines, and `GATE PASS`. The check compares the `[M, N/2]` output against `silu(H[:, :N/2]) * H[:, N/2:]`.

- [ ] **Step 5: Final diff + gate log** to the owner.
- [ ] **Step 6: On "approve": commit and push.** `git -C $WT add kernels/cdna4/gemm/bf16fp32/epilogue/bindings/gemm_swiglu.cpp`, commit (chunk 17/21, include the gate log), push, then update the PR body.

---

### Task 18: `ops/reductions.cuh` (chunk 18, compile gate)

**Files:** Create `$EPI/ops/reductions.cuh`. Throwaway: `$G/tu/c18.cpp`.

- [ ] **Step 1: Import**

```bash
git -C $WT show $PR:$P90/ops/reductions.cuh > $EPI/ops/reductions.cuh
```

- [ ] **Step 2: Review session (owner checkpoint).** Talking points:
  - Group `grp = col·4 + wc` owns two **non-adjacent** 32-column chunks (`wc` and `wc+4`) of each 256-column block. The comment says "REG_BLOCK_N columns", which reads as contiguous.
  - #90's own test only checked `partials.sum(0)`, so it never verified the per-group layout. Our gate checks it.
- [ ] **Step 3: Apply the agreed edits.**
- [ ] **Step 4: Gate**

`$G/tu/c18.cpp`:

```cpp
#include "reductions.cuh"
struct PartialsG { gl<float,-1,-1,-1,-1> partials; };
using Accum = rt_fl<HALF_REG_BLOCK_M, HALF_REG_BLOCK_N, col_l, rt_16x16_s>[2][2];
__global__ void gate_partials(const PartialsG g, int row, int col, int wr, int wc) {
    Accum C;
    for (auto& r : C) for (auto& t : r) zero(t);
    partial_row_sum_sq(g, C, row, col, wr, wc);
}
```

Run: `$G/compile_gate.sh device $G/tu/c18.cpp`. Expected: `GATE PASS`.

- [ ] **Step 5: Final diff** to the owner.
- [ ] **Step 6: On "approve": commit and push.** `git -C $WT add kernels/cdna4/gemm/bf16fp32/epilogue/ops/reductions.cuh`, commit (chunk 18/21), push, then update the PR body.

---

### Task 19: `gemm_base.cuh` + partials-only branch (chunk 19, compile gate)

**Files:** Modify `$EPI/gemm_base.cuh` (`launch`). Throwaway: `$G/tu/c19.cpp`.

- [ ] **Step 1: Starting point.** In `launch`, wrap the output-shape check in a guard for epilogues without `c`:

```cpp
    if constexpr (requires { g.c; }) {                          // partials-only epilogues have no c
        int expect_cols = N;
        if constexpr (requires { Epilogue::out_cols(N); }) expect_cols = Epilogue::out_cols(N);
        if (g.c.rows() != M || g.c.cols() != expect_cols)
            throw std::runtime_error("GEMM: output c shape mismatch (need c=[M, out_cols(N)])");
    }
```

Use the chunk-15 code as landed, if review changed it.

- [ ] **Step 2: Review session (owner checkpoint).** Talking points:
  - Same silent-trait risk as chunk 15. A globals struct with a misspelled `c` field skips the check entirely.
  - Who validates `partials`? In #90, the binding's `dispatch` does it, not `launch`.
- [ ] **Step 3: Apply the agreed edits.**
- [ ] **Step 4: Gate**

`$G/tu/c19.cpp`:

```cpp
#include "gemm_base.cuh"
#include "reductions.cuh"
struct PartialsOnlyG { _gl_A a; _gl_B b; gl<float,-1,-1,-1,-1> partials; };
struct PartialsOnly {
    template<typename G, typename A>
    static __device__ inline void apply(const G& g, A& C, int r, int c, int wr, int wc) { partial_row_sum_sq(g, C, r, c, wr, wc); }
};
template void launch<PartialsOnly, PartialsOnlyG>(PartialsOnlyG, hipStream_t);
template void launch<NoOpEpilogue, gemm_args_base>(gemm_args_base, hipStream_t);
```

```bash
$G/compile_gate.sh full $G/tu/c19.cpp && $G/compile_gate.sh full $G/tu/c15.cpp && $G/compile_gate.sh device $G/tu/c04.cpp
```

Expected: `GATE PASS` three times. c04 must still show `VGPRs: 210`, 0 spills.

- [ ] **Step 5: Final diff** (`git -C $WT diff`) to the owner.
- [ ] **Step 6: On "approve": commit and push.** `git -C $WT add kernels/cdna4/gemm/bf16fp32/epilogue/gemm_base.cuh`, commit (chunk 19/21, subject `epilogue: gemm_base.cuh: partials-only launch (chunk 19/21)`), push, then update the PR body.

---

### Task 20: `epilogues/partialrms.cuh` (chunk 20, compile gate)

**Files:** Create `$EPI/epilogues/partialrms.cuh`. Throwaway: `$G/tu/c20.cpp`.

- [ ] **Step 1: Import**

```bash
git -C $WT show $PR:$P90/epilogues/partialrms.cuh > $EPI/epilogues/partialrms.cuh
```

- [ ] **Step 2: Review session (owner checkpoint).** Talking point: the comment mentions an "aux kernel", which doesn't exist until after Phase 3. Decide whether to describe the output instead.
- [ ] **Step 3: Apply the agreed edits.**
- [ ] **Step 4: Gate**

`$G/tu/c20.cpp`:

```cpp
#include "gemm_base.cuh"
#include "partialrms.cuh"
template __global__ void gemm_kernel<PartialRMSEpilogue, PartialRMSGlobals>(const PartialRMSGlobals, int, int, int);
template void launch<PartialRMSEpilogue, PartialRMSGlobals>(PartialRMSGlobals, hipStream_t);
```

```bash
$G/compile_gate.sh device $G/tu/c20.cpp && $G/compile_gate.sh full $G/tu/c20.cpp
```

Expected: `GATE PASS` twice, with 0 spills.

- [ ] **Step 5: Final diff** to the owner.
- [ ] **Step 6: On "approve": commit and push.** `git -C $WT add kernels/cdna4/gemm/bf16fp32/epilogue/epilogues/partialrms.cuh`, commit (chunk 20/21), push, then update the PR body.

---

### Task 21: `bindings/gemm_partialrms.cpp` (chunk 21, gfx950 gate: partialrms)

**Files:** Create `$EPI/bindings/gemm_partialrms.cpp`.

- [ ] **Step 1: Import**

```bash
git -C $WT show $PR:$P90/bindings/gemm_partialrms.cpp > $EPI/bindings/gemm_partialrms.cpp
```

- [ ] **Step 2: Review session (owner checkpoint).** Talking points:
  - `dispatch` validates the `partials` shape before `launch`. Good, but its error text names `REG_BLOCK_N` and not the number 64.
  - `m.doc()` is set after `hkstream::bind`, the opposite order from the other bindings.
- [ ] **Step 3: Apply the agreed edits.**
- [ ] **Step 4: Gate**

```bash
cd $EPI && make KERNEL=partialrms && make KERNEL=partialrms clean
$G/austin_run.sh partialrms 2>&1 | tee $S/gate-c21.log
```

Expected: four `partialrms <shape>: ... PASS` lines (the `[N/64, M]` per-group layout is checked exactly), two `reject ...: PASS` lines, and `GATE PASS`.

- [ ] **Step 5: Final diff + gate log** to the owner.
- [ ] **Step 6: On "approve": commit and push.** `git -C $WT add kernels/cdna4/gemm/bf16fp32/epilogue/bindings/gemm_partialrms.cpp`, commit (chunk 21/21, include the gate log), push, then update the PR body.

---

### Task 22: Phase 1 close-out

- [ ] **Step 1: Full re-gate of the landed tree.** Re-run every compile gate and all five gfx950 gates against the final HEAD. This catches regressions from later edits to shared files.

```bash
for t in c01 c03 c04 c08 c09 c11 c12 c14 c16 c18 c20; do $G/compile_gate.sh device $G/tu/$t.cpp || break; done
for t in c04 c09 c12 c15 c16 c19 c20; do $G/compile_gate.sh full $G/tu/$t.cpp || break; done
for k in noop silu scale swiglu partialrms; do $G/austin_run.sh $k 2>&1 | tee $S/final-$k.log | tail -1; done
```

Expected: every line `GATE PASS`.

- [ ] **Step 2: Update the PR body.** All 21 boxes are ticked, and a "Phase 1 complete @ <sha>" line is added with the five final gate results.
- [ ] **Step 3: Vault.** `/save` a page on the Phase 1 outcome, the #90 deviations, and any #90 bugs found, then run `bash ~/rockKE/vault/vault-sync.sh`.
- [ ] **Step 4: Start Phase 2.** Begin a new brainstorming session for the correctness test suite, with the five landed kernels and the gate script as input.
