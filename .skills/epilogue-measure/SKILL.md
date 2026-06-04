---
name: epilogue-measure
description: Use when testing or measuring a fused bf16→fp32 GEMM epilogue in HipKittens (kernels/gemm/bf16fp32/epilogue) — verifying correctness of a new/changed epilogue kernel, quantifying the fusion payoff (fused vs unfused, HBM bytes saved), checking "does my epilogue compute the right thing" or "was fusing worth it" on gfx950. Not for whole-GEMM throughput vs torch/rocBLAS, and not for the base GEMM's own bf16 accuracy (that is test_python.py).
---

# epilogue-measure

## Overview

Two registry-driven tools (in `kernels/gemm/bf16fp32/epilogue/util/`) for any fused epilogue:

- **`test_epilogue.py <k>`** — correctness. (a) the epilogue at its identity param is
  **bit-exact** to the no-op GEMM output; (b) its output matches a torch reference applied to
  the no-op baseline across a param sweep — comparing vs the no-op baseline (not an fp32
  matmul) **isolates the epilogue from the GEMM's bf16 error**.
- **`fusion_win.py --kernels <k>`** — payoff. Runs fused (one kernel) vs unfused (base GEMM
  writes D to HBM, then a torch epilogue reads D), reports wall-clock speedup and **HBM bytes
  saved**. Bytes saved is the shape-independent truth; speedup is ~1x compute-bound, >1x
  memory-bound (skinny K).

Both pull each kernel's definition from one place — `util/epilogue_testlib.py` (the `EPILOGUES`
registry) — so the test's correctness oracle and the bench's unfused stage are the SAME math.

## When to use / not

- Use to test a new/changed epilogue (RMSNorm, residual, SwiGLU, …) or to quantify its fusion win.
- NOT for GEMM-vs-torch/rocBLAS throughput, and NOT to validate the base GEMM's bf16 accuracy
  (that is `test_python.py`). NOT for comparing different epilogues against each other (meaningless).

## Quick reference

Each tool needs the base GEMM `tk_kernel`, the `tk_noop` baseline, and the fused `tk_<k>` built.
**REQUIRED SUB-SKILL:** use `kreb` (arch `gfx950`) to build + run; the runs need a real GPU.

In one gfx950 kreb job, from `kernels/gemm/bf16fp32/epilogue/`:

```bash
( cd .. && make GPU_TARGET=CDNA4 )                      # base GEMM -> tk_kernel*.so
make KERNEL=noop  MODULE=tk_noop  GPU_TARGET=CDNA4      # baseline  -> tk_noop*.so
make KERNEL=<k>   MODULE=tk_<k>   GPU_TARGET=CDNA4      # fused     -> tk_<k>*.so   (e.g. scale)
cp ../tk_kernel.cpython*.so .                           # base .so next to the kernels

python3 util/test_epilogue.py <k>                       # correctness -> ALL PASSED / SOME FAILED
python3 util/fusion_win.py --kernels <k> --json results_fusion.json   # payoff table + manifest
```

Run from the epilogue dir (where the `.so` are built); the scripts add it to `sys.path` so the
`tk_*` modules import. `--help` on `fusion_win.py` for `--shapes`/`--iters`.

## Adding a kernel

Add ONE entry to `util/epilogue_testlib.py::EPILOGUES`: `module` (`tk_<k>`), `args(m,n,k)`,
`ref(D,out,*args)` (the torch definition — oracle AND unfused stage), `identity` (param where
it == noop, or `None`), `sweep`, `label`, `hbm_passes` (D transfers fusion skips). Then both
`test_epilogue.py <k>` and `fusion_win.py --kernels <k>` work — no new files.

## Interpreting

| signal | meaning |
|---|---|
| `bit_exact=True` at identity | the epilogue path adds zero numerical noise (strongest correctness) |
| isolate sweep `PASS` | the epilogue math matches the reference under project tol (rtol=1e-2/atol=1e-1) |
| `HBM saved` | bytes the unfused D round-trip costs and fusion skips — **the real win** |
| `speedup` | shape-dependent: ~1x compute-bound, >1x memory-bound |

## Common mistakes

- Reporting **TFLOPS** as the fusion win — wrong metric; fusion saves bytes, not flops.
- Comparing the epilogue against an **fp32 matmul** instead of the no-op baseline — re-tests the
  GEMM's bf16 error (amplified by the param) and false-fails correct kernels.
- **Host cross-compile** to skip the GPU — fails for these kernels (`gl<float,1,1,1,1>` ctor) and
  the runs need a GPU anyway; use `kreb gfx950`.
- Forgetting to build/copy **`tk_kernel`** (base) or **`tk_noop`** — both tools need them.
- Reading ~1x **wall-clock** at a compute-bound shape as "no win" — check `HBM saved` and a
  memory-bound (skinny-K) shape.
