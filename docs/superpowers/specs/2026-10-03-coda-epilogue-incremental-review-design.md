# CODA epilogue: incremental bottom-up review of PR #90

Date: 2026-10-03 · Owner: avirgoel · Status: approved design, pending spec review

## Problem

[HazyResearch/HipKittens#90](https://github.com/HazyResearch/HipKittens/pull/90) ports CODA's
GEMM-epilogue fusion (forward pass only) onto HipKittens' bf16→fp32 GEMM for gfx950. It adds 52
files and has 93 commits: a templated GEMM mainloop, 13 epilogues, an aux reduction kernel, the
`hk.matmul` API, a registry, a correctness suite, and benchmarks. It is too large to review as one
unit, and its test and benchmark machinery is convoluted.

## Goal

Rebuild #90 on the fork, one reviewed file at a time, into a PR that supersedes #90 upstream.
Each file lands only after the owner reviews it and it passes a gate.

## Phases

1. **Phase 1 (this spec):** C++ building blocks plus one epilogue per output mode: `noop`, `silu`,
   `scale` (store), `swiglu` (dim-reducing), `partialrms` (partials-only). No Python from #90.
2. **Phase 2 (separate spec):** a correctness test suite designed from scratch. It covers every
   Phase 1 kernel and replaces the throwaway gate scripts. #90's `test_all.py`,
   `epilogue_testlib.py`, and registry are reference material, not a starting point.
3. **Phase 3 (separate spec):** a performance benchmark suite designed from scratch. It must be
   robust (cold cache, repeats, same-node `torch.compile` baseline) but small.

After Phase 3, the remaining epilogues, `aux_reduce.cuh`, the chains, `hk.py`, and `README.md` land
one by one, each using the Phase 2 and Phase 3 tools. Each follows the same review loop.

## Facts this design rests on

- Fork `main` was fast-forwarded to upstream `main` `be1c9184` on 2026-10-03 (`gh repo sync`). The
  owner granted this as a one-time exception to the no-push-to-main rule, for mirroring upstream
  only.
- #90 is reviewed from the ref `fork/users/avirgoel/upstream-forward-epilogue` @ `b41ff33f`. That is
  the PR head, and #90's base is `be1c9184`.
- #90 places files under `kernels/gemm/bf16fp32/epilogue/`. Upstream has since reorganized into
  `kernels/cdna{3,4,5}/` with a shared `kernels/common.mk`. The base GEMM that `gemm_base.cuh`
  copies is now `kernels/cdna4/gemm/bf16fp32/256_256_64_32_with16x32.cpp`.
- This login node (MI210, gfx90a, HIP 7.1.25424, AMD clang 20) can compile for gfx950. Verified:
  a full device compile of #90's `gemm_kernel<NoOpEpilogue, gemm_args_base>` gave 210 VGPRs,
  0 AGPRs, 0 spills. A host build of `bindings/gemm_noop.cpp` into a pybind `.so` also succeeded.
- gfx950 hardware is AUSTIN MI355X, reached with `srun --constraint=AUSTIN
  --gres=gpu:gfx950-mi355x:1` and a torch-bearing pytorch `.sqsh` image from `/cluster/images`
  (vault `runbook-austin-gpu-verify`; the kreb images lack torch). Source is staged on beegfs
  `/scratch`. The runbook warns `/scratch` was node-local on AUSTIN in one October trial, so the
  plan probes it before the first GPU gate.

## Section 1: location and mechanics

- **Location:** `kernels/cdna4/gemm/bf16fp32/epilogue/`.
- **Build:** the `Makefile` sets local knobs and then includes `../../../../common.mk`, following
  `kernels/cdna4/gemm/bf16fp32/Makefile`.
- **Branch:** `users/avirgoel/coda-epilogue-reviewed`, cut from `fork/main` (`be1c9184`).
- **Worktree:** `~/rockKE/coda/HipKittens/.worktrees/coda-epilogue-reviewed`. It is a cone sparse
  checkout of `include/`, `kernels/cdna4/gemm/bf16fp32/`, and `docs/`, because home-directory
  quota is tight. Commits still carry the full tree.
- **PR:** one draft PR on `AviralGoelAMD/HipKittens`, from the branch into `main`. It stays a draft
  until Phases 1–3 are complete.
- **Commits:** one commit per approved chunk. The message records:
  - the #90 source path @ `b41ff33f`
  - each change made in review, with its reason
  - the gate command and its output
- **This spec** is the branch's first commit. It is excluded when the upstream PR is assembled,
  because the upstream PR is built fresh at migration time.
- **Untouched:** fork PR #1 (`users/avirgoel/ck/epilogue-fusion`) and #90 itself.

## Section 2: Phase 1 chunk order

**Rule: a file lands with only the code that a landed kernel uses.** Helpers that only later
epilogues need wait for those epilogues. For example, `apply_inv_rms` and `apply_gamma` from
`ops/vec.cuh` wait for `rmsnorm_scale`. Each `launch<>` output-mode branch lands with its first user.

| # | File (under `epilogue/`) | Content that lands | Gate |
|---|---|---|---|
| 1 | `epilogue_args.cuh` | constants, `gemm_args_base`, gl typedefs | compile |
| 2 | `stream.cuh` | per-module launch stream (review decides whether it stays in Phase 1) | compile |
| 3 | `ops/base.cuh` | `subtile_coords`, `block_coords`, `store_C` | compile |
| 4 | `gemm_base.cuh` | `NoOpEpilogue`, `gemm_kernel`, store-only `launch` | compile |
| 5 | `bindings/gemm_noop.cpp` | noop binding | compile (`.so` via direct `hipcc`) |
| 6 | `Makefile` | `common.mk`-based build | **gfx950: noop** |
| 7 | `.gitignore` | build artifacts only | none |
| 8 | `ops/activations.cuh` | `fast_silu`, `silu_op` | compile |
| 9 | `epilogues/silu.cuh` | `SiluEpilogue` | compile |
| 10 | `bindings/gemm_silu.cpp` | silu binding | **gfx950: silu** |
| 11 | `ops/vec.cuh` | `apply_scale` only | compile |
| 12 | `epilogues/scale.cuh` | `ScaleGlobals`, `ScaleEpilogue` | compile |
| 13 | `bindings/gemm_scale.cpp` | scale binding | **gfx950: scale** |
| 14 | `ops/base.cuh` | add `store_swiglu` | compile |
| 15 | `gemm_base.cuh` | add the `out_cols` (dim-reducing) branch to `launch` | compile |
| 16 | `epilogues/swiglu.cuh` | `SwigluEpilogue` | compile |
| 17 | `bindings/gemm_swiglu.cpp` | swiglu binding | **gfx950: swiglu** |
| 18 | `ops/reductions.cuh` | `partial_row_sum_sq` | compile |
| 19 | `gemm_base.cuh` | add the partials-only (no `c`) branch to `launch` | compile |
| 20 | `epilogues/partialrms.cuh` | `PartialRMSGlobals`, `PartialRMSEpilogue` | compile |
| 21 | `bindings/gemm_partialrms.cpp` | partialrms binding | **gfx950: partialrms** |

If the chunk 2 review drops `stream.cuh` from Phase 1, `launch` uses the null stream and bindings
omit `hkstream::bind` until Phase 3 needs graph capture.

Chunk 4 is reviewed as a diff against upstream `256_256_64_32_with16x32.cpp` @ `be1c9184`. It may
span several sessions but lands as one commit. The review must check whether upstream base-GEMM
changes made after #90 copied the mainloop (for example #88's int64 index-overflow fix) are
present in the copy.

## Section 3: gate definitions

### Compile gate (this node)

- A throwaway `.cpp` file (never committed) includes the chunk's file. It explicitly instantiates
  every landed kernel template and `static_assert`s key constants. Explicit instantiation is
  required because `-fsyntax-only` only partly checks uninstantiated template bodies.
- Command:
  `hipcc -x hip -std=c++20 -O3 -DKITTENS_CDNA4 --offload-arch=gfx950 --cuda-device-only -c -Rpass-analysis=kernel-resource-usage`.
- Include path: upstream `include/` at `be1c9184`, from the review worktree. The main checkout
  is not used, because it sits on another branch.
- Chunks that touch a kernel record VGPR, AGPR, and spill counts in the commit message. Chunk 4
  must show 0 spills.
- Chunk 5 instead builds the binding to a full pybind `.so` with `hipcc -shared -fPIC`.

### gfx950 gate (AUSTIN MI355X)

- Build with the chunk's `Makefile` inside the pytorch `.sqsh` image, from source staged on
  `/scratch`.
- A throwaway Python script (never committed) checks:
  1. **Accuracy:** output vs an fp32 `torch` reference computed from the same bf16 inputs, with
     `torch.testing.assert_close(rtol=1e-2, atol=1e-2)`. Shapes (M×N×K):
     `256×256×128`, `4096×4096×4096`, `8192×2048×4096`, and `4096×4096×128` (K at the
     128-element minimum). Dim-reducing and partials-only epilogues compare their own output
     shapes: `[M, N/2]` and `[N/64, M]` partials.
  2. **Rejection:** `M % 256 != 0` and `K % 128 != 0` must raise.
  3. **noop only (chunk 6):** `tk_noop` output is bit-identical to upstream's untouched base GEMM,
     built from `kernels/cdna4/gemm/bf16fp32/` on the same inputs.
- The commit message records the node, GPU, image, command, and printed results.
- AUSTIN's ROCm (7.13 image) differs from this node's (7.1). If a chunk passes the local compile
  gate but fails to build on AUSTIN, the chunk is blocked.

## Section 4: review loop and failure handling

Per chunk. Nothing is committed without the owner's explicit approval.

1. **Present:** #90's file (or, for chunk 4, the diff against upstream), a plain-English
   walkthrough, and findings tagged bug / dead code / stale-or-duplicate comment / naming /
   missing check.
2. **Cross-question:** the owner challenges; answers come from the code or a small experiment.
3. **Agree edits:** the owner decides every disagreement.
4. **Apply** in the review worktree.
5. **Gate** per Section 3.
6. **Final diff:** show the landing file and the gate output.
7. **Approve → commit → push** to `fork/users/avirgoel/coda-epilogue-reviewed`, then tick the
   chunk in the PR description.

The PR description tracks a checklist of the 21 Phase 1 chunks and a **Deviations from #90**
table (file, change, reason).

| Situation | Handling |
|---|---|
| Gate fails | The chunk does not land. Fix and re-gate. If an earlier file is at fault, handle it as the next row. |
| An already-landed file is wrong | New fix commit, reviewed as its own chunk. Pushed history is never rewritten. |
| A review edit breaks a later #90 file | Allowed. Log it in the deviations table and adapt the later file at its turn. |
| A bug that also affects #90 as-is | Log it in the deviations table and in the vault. The owner decides whether to comment on #90. |
| No AUSTIN gfx950 allocation | The gfx950 chunk waits. Later compile-gated chunks may be reviewed but do not land ahead of it. |

## Done criteria

Phase 1 is done when chunk 21 passes its gfx950 gate and is pushed.

## Out of scope

Phase 2 and Phase 3 designs; epilogues beyond the five above; `aux_reduce.cuh`; chains; `hk.py`;
`README.md`; upstream PR mechanics; any change to #90.
