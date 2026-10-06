# Epilogue benchmarks (gfx950)

Benchmarks for the fused GEMM-epilogue kernels in this directory, against PyTorch eager and
`torch.compile`. Every script checks its outputs against an fp32 PyTorch reference before timing,
prints a table, and exits non-zero if any row fails.

| Script | Question |
|---|---|
| `ab_noop_vs_base.py` | Does the epilogue chassis (`tk_noop`) cost anything versus upstream's original GEMM? |
| `bench_epilogues.py` | How much time and HBM traffic does each fused epilogue save versus eager PyTorch? |
| `bench_layer.py --phases` | Per phase of one forward layer: HK vs eager vs `torch.compile`. |
| `bench_layer.py --full` | The whole layer: HK vs eager vs `torch.compile`. |

## Build

From this directory's parent (the epilogue directory). Building needs a ROCm `hipcc` that targets gfx950
but no GPU; running needs a gfx950 GPU and PyTorch.

```bash
for m in noop silu scale residual_add rmsnorm_scale swiglu rmsnorm_swiglu rope rmsnorm_rope \
         partialrms residual_rms rms_reduce; do make KERNEL=$m; done
make -C ..                                  # upstream base GEMM, for ab_noop_vs_base.py
bench/build_attn.sh 32 8 128 2048 4096 8192 # attention: one build per sequence length (H, H_KV, Dh, M...)
```

Each module takes about 6 s from a local disk. Build on a local disk: on our cluster the same module
took 6 s from `/tmp` and 269 s from a network filesystem.

## Run

```bash
python3 bench/ab_noop_vs_base.py
python3 bench/bench_epilogues.py
python3 bench/bench_layer.py --phases --full
```

Common flags: `--config pr1`, `--M 2048,4096`, `--iters 50`, `--warmup 10`, `--json out.json`,
`--markdown`. `torch.compile` autotuning can pick different kernels in different processes, so run each
script in 3 fresh processes and report the spread, not the best run.

## Config

`pr1`: d = 4096, d_ff = 11008, H = 32, H_KV = 8, Dh = 128, batch 1, causal, M ∈ {2048, 4096, 8192}.
All configs live in `CONFIGS` in `common.py`.

## How timing works

- **Cold cache.** Before every timed call, the timer overwrites a buffer twice the last-level cache
  (2 × 256 MiB on gfx950), so each call reads its inputs from HBM. GPU events bracket only the call.
- **Median of 50 calls** after 10 warm-up calls; JSON also records p10 and p90.
- **Outside the timer:** input creation, weight preparation (`hk.prepare`: gamma folding, column
  permutation, transpose), and `torch.compile` compilation and autotuning.
- **One compile mode:** `torch.compile(mode="max-autotune-no-cudagraphs", dynamic=False)`. CUDA graphs
  are not used: both sides are limited by kernel time, not launch overhead.
- Eager and compiled PyTorch are separate columns; there is no best-of number.

## Columns in `bench_epilogues.py`

- **HK:** one fused HipKittens call.
- **eager:** `torch.matmul` (hipBLASLt) followed by the epilogue written as lean bf16 eager PyTorch.
- **fusion-only:** HK's `noop` GEMM followed by the same eager PyTorch. HK and fusion-only share one
  GEMM, so `fusion-only/HK` is the speedup from fusion alone; `eager/HK` also includes the GEMM difference.
- **HBM saved:** an analytical estimate, not a hardware measurement. Each epilogue has one formula in
  `bench_epilogues.py` (`saved=` in `make_case`): the bytes of every [M, N] or [M, K] intermediate the
  eager code writes to HBM and reads back, counted once for the write and once for the read. Vectors
  ([M] or [N]) are ignored. It ignores cache: a small intermediate may stay in the 256 MiB Infinity Cache,
  so the real saving can be lower. With bf16 sizes `MN = M·N·2` and `MK = M·K·2` bytes:

  | Epilogue | Eager intermediates | Saved |
  |---|---|---|
  | noop | none | 0 |
  | silu, scale, residual_add | D | 2·MN |
  | rmsnorm_scale | D, D·r | 4·MN |
  | rmsnorm | x·r, x·r·g | 4·MK |
  | swiglu | D, silu(gate) (half width) | 3·MN |
  | rmsnorm_swiglu | rmsnorm + swiglu | 4·MK + 3·MN |
  | rope | D, six half-width products and sums | 8·MN |
  | rmsnorm_rope | rmsnorm + rope | 4·MK + 8·MN |
  | inv_rms | D | 2·MN − HK's fp32 partials, 2·(N/64)·M·4 |
  | residual_rms | D, second read of h | 3·MN − HK's fp32 partials |

## Layer phases and boundary rules

| # | Phase | HK | PyTorch |
|---|---|---|---|
| 1 | qkv | `rmsnorm_rope` (Wq, Wk) + `rmsnorm` (Wv): 3 launches | `rms_norm`, three matmuls, RoPE on q and k |
| 2 | attention | `gqa_causal` on views of q, k, v | transposes, then SDPA |
| 3 | out_proj | `residual_rms(o, Wo, x)`: 2 launches | `x + o @ Wo` |
| 4 | gate_up | `rmsnorm_swiglu`: 1 launch | `rms_norm`, matmul, `silu(gate) * up` |
| 5 | down | `residual_rms(g, Wd, h)`: 2 launches | `h + g @ Wd`, then 1/rms for the next layer |

- HK splits RMSNorm across phases: the producing GEMM emits r = 1/rms, the consuming GEMM applies it.
  PyTorch normalizes in the consumer. Each side does each RMS reduction once per layer, so phase *sums*
  compare fairly, but a single phase does not compare op-for-op.
- HK's layer input `r_attn` is computed once outside timing (a previous layer would emit it).
- Each phase is timed on intermediates captured from a full run of the same implementation.
- `--phases` compiles each PyTorch phase separately; `--full` compiles the whole layer as one function.
  `compile full - phases` shows what cross-phase fusion buys PyTorch.
- **Attention baseline:** the script uses the strongest fused SDPA backend available (flash, then
  memory-efficient), with native GQA if that backend supports it, else KV expanded to all heads. The math
  backend is disabled. The chosen backend is printed and saved in the JSON.

## Results

Measured on AMD Instinct MI355X (`gfx950:sramecc+:xnack-`, GPU UUID `63343132-3431-3165-3334-353539393564`, host `smci355-ccs-aus-n14-05`), HIP
7.13.99004, torch `2.11.0+rocm7.13.0rc2`, commit `c8186697`, config `pr1`. Tables are process 1 of 3 fresh
processes; the spread lines give the min–max of each headline ratio over all 3. SDPA baseline: flash
attention with native GQA. Commits after `c8186697` changed only docstrings, `--help` text, and the
build commands (`build_attn.sh`); no timed or checked code changed.

### Refactor neutrality

| M | K | N | noop ms | base ms | noop/base | bitwise | status |
|---|---|---|---|---|---|---|---|
| 2048 | 4096 | 4096 | 0.094 | 0.094 | 1.000 | yes | PASS |
| 2048 | 4096 | 1024 | 0.085 | 0.085 | 0.995 | yes | PASS |
| 2048 | 4096 | 22016 | 0.290 | 0.289 | 1.001 | yes | PASS |
| 2048 | 11008 | 4096 | 0.272 | 0.272 | 0.999 | yes | PASS |
| 4096 | 4096 | 4096 | 0.108 | 0.107 | 1.002 | yes | PASS |
| 4096 | 4096 | 1024 | 0.093 | 0.093 | 1.000 | yes | PASS |
| 4096 | 4096 | 22016 | 0.564 | 0.565 | 0.999 | yes | PASS |
| 4096 | 11008 | 4096 | 0.283 | 0.284 | 0.997 | yes | PASS |
| 8192 | 4096 | 4096 | 0.203 | 0.204 | 0.999 | yes | PASS |
| 8192 | 4096 | 1024 | 0.099 | 0.099 | 1.000 | yes | PASS |
| 8192 | 4096 | 22016 | 1.083 | 1.084 | 0.999 | yes | PASS |
| 8192 | 11008 | 4096 | 0.539 | 0.540 | 0.999 | yes | PASS |
| 8192 | 8192 | 8192 | 0.748 | 0.747 | 1.001 | yes | PASS |

Spread: `noop/base` 0.994–1.004 over all shapes and processes; every output bitwise identical.

### Fused epilogues vs eager

| epilogue | GEMM | M | N | K | HK ms | eager ms | fusion-only ms | eager/HK | fusion-only/HK | HBM saved MB | rel err | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noop | Wq | 2048 | 4096 | 4096 | 0.095 | 0.067 | 0.093 | 0.70 | 0.97 | 0.0 | 3.3e-03 | PASS |
| noop | Wk | 2048 | 1024 | 4096 | 0.086 | 0.039 | 0.086 | 0.46 | 1.00 | 0.0 | 3.3e-03 | PASS |
| noop | Wo | 2048 | 4096 | 4096 | 0.097 | 0.066 | 0.097 | 0.68 | 1.00 | 0.0 | 3.3e-03 | PASS |
| noop | Wgu | 2048 | 22016 | 4096 | 0.297 | 0.286 | 0.296 | 0.96 | 1.00 | 0.0 | 3.3e-03 | PASS |
| noop | Wd | 2048 | 4096 | 11008 | 0.264 | 0.184 | 0.265 | 0.70 | 1.00 | 0.0 | 3.3e-03 | PASS |
| rope | Wq | 2048 | 4096 | 4096 | 0.101 | 0.130 | 0.160 | 1.29 | 1.59 | 134.2 | 1.7e-03 | PASS |
| rope | Wk | 2048 | 1024 | 4096 | 0.090 | 0.071 | 0.116 | 0.79 | 1.30 | 33.6 | 1.7e-03 | PASS |
| rmsnorm_rope | Wq | 2048 | 4096 | 4096 | 0.101 | 0.155 | 0.177 | 1.54 | 1.76 | 201.3 | 2.3e-03 | PASS |
| rmsnorm_rope | Wk | 2048 | 1024 | 4096 | 0.090 | 0.094 | 0.137 | 1.04 | 1.51 | 100.7 | 2.3e-03 | PASS |
| scale | Wv | 2048 | 1024 | 4096 | 0.090 | 0.043 | 0.088 | 0.48 | 0.98 | 8.4 | 3.3e-03 | PASS |
| rmsnorm_scale | Wv | 2048 | 1024 | 4096 | 0.087 | 0.051 | 0.096 | 0.59 | 1.11 | 16.8 | 3.3e-03 | PASS |
| rmsnorm | Wv | 2048 | 1024 | 4096 | 0.087 | 0.065 | 0.107 | 0.74 | 1.23 | 67.1 | 3.8e-03 | PASS |
| residual_add | Wo | 2048 | 4096 | 4096 | 0.105 | 0.076 | 0.107 | 0.73 | 1.02 | 33.6 | 3.3e-03 | PASS |
| residual_add | Wd | 2048 | 4096 | 11008 | 0.273 | 0.193 | 0.275 | 0.71 | 1.01 | 33.6 | 3.3e-03 | PASS |
| residual_rms | Wo | 2048 | 4096 | 4096 | 0.112 | 0.093 | 0.123 | 0.83 | 1.10 | 49.3 | 3.3e-03 | PASS |
| residual_rms | Wd | 2048 | 4096 | 11008 | 0.279 | 0.212 | 0.291 | 0.76 | 1.04 | 49.3 | 3.3e-03 | PASS |
| inv_rms | Wo | 2048 | 4096 | 4096 | 0.102 | 0.082 | 0.113 | 0.81 | 1.11 | 32.5 | 5.0e-08 | PASS |
| inv_rms | Wd | 2048 | 4096 | 11008 | 0.269 | 0.201 | 0.280 | 0.75 | 1.04 | 32.5 | 5.9e-08 | PASS |
| silu | Wgu | 2048 | 22016 | 4096 | 0.315 | 0.322 | 0.344 | 1.02 | 1.09 | 180.4 | 3.3e-03 | PASS |
| swiglu | Wgu | 2048 | 22016 | 4096 | 0.294 | 0.341 | 0.360 | 1.16 | 1.22 | 270.5 | 3.3e-03 | PASS |
| rmsnorm_swiglu | Wgu | 2048 | 22016 | 4096 | 0.299 | 0.363 | 0.380 | 1.22 | 1.27 | 337.6 | 4.2e-03 | PASS |
| noop | Wq | 4096 | 4096 | 4096 | 0.115 | 0.105 | 0.108 | 0.91 | 0.94 | 0.0 | 3.3e-03 | PASS |
| noop | Wk | 4096 | 1024 | 4096 | 0.089 | 0.049 | 0.089 | 0.55 | 1.00 | 0.0 | 3.3e-03 | PASS |
| noop | Wo | 4096 | 4096 | 4096 | 0.108 | 0.104 | 0.108 | 0.96 | 1.00 | 0.0 | 3.3e-03 | PASS |
| noop | Wgu | 4096 | 22016 | 4096 | 0.583 | 0.606 | 0.581 | 1.04 | 1.00 | 0.0 | 3.3e-03 | PASS |
| noop | Wd | 4096 | 4096 | 11008 | 0.277 | 0.261 | 0.276 | 0.94 | 0.99 | 0.0 | 3.3e-03 | PASS |
| rope | Wq | 4096 | 4096 | 4096 | 0.117 | 0.217 | 0.214 | 1.86 | 1.84 | 268.4 | 1.7e-03 | PASS |
| rope | Wk | 4096 | 1024 | 4096 | 0.094 | 0.089 | 0.128 | 0.95 | 1.36 | 67.1 | 1.7e-03 | PASS |
| rmsnorm_rope | Wq | 4096 | 4096 | 4096 | 0.118 | 0.250 | 0.249 | 2.11 | 2.11 | 402.7 | 2.3e-03 | PASS |
| rmsnorm_rope | Wk | 4096 | 1024 | 4096 | 0.095 | 0.127 | 0.163 | 1.33 | 1.72 | 201.3 | 2.3e-03 | PASS |
| scale | Wv | 4096 | 1024 | 4096 | 0.093 | 0.053 | 0.092 | 0.57 | 1.00 | 16.8 | 3.3e-03 | PASS |
| rmsnorm_scale | Wv | 4096 | 1024 | 4096 | 0.089 | 0.063 | 0.102 | 0.71 | 1.14 | 33.6 | 3.3e-03 | PASS |
| rmsnorm | Wv | 4096 | 1024 | 4096 | 0.090 | 0.087 | 0.123 | 0.96 | 1.36 | 134.2 | 3.8e-03 | PASS |
| residual_add | Wo | 4096 | 4096 | 4096 | 0.119 | 0.132 | 0.126 | 1.11 | 1.06 | 67.1 | 3.3e-03 | PASS |
| residual_add | Wd | 4096 | 4096 | 11008 | 0.290 | 0.279 | 0.297 | 0.96 | 1.02 | 67.1 | 3.3e-03 | PASS |
| residual_rms | Wo | 4096 | 4096 | 4096 | 0.126 | 0.150 | 0.144 | 1.19 | 1.15 | 98.6 | 3.3e-03 | PASS |
| residual_rms | Wd | 4096 | 4096 | 11008 | 0.298 | 0.299 | 0.316 | 1.00 | 1.06 | 98.6 | 3.3e-03 | PASS |
| inv_rms | Wo | 4096 | 4096 | 4096 | 0.111 | 0.127 | 0.126 | 1.14 | 1.14 | 65.0 | 4.8e-08 | PASS |
| inv_rms | Wd | 4096 | 4096 | 11008 | 0.285 | 0.280 | 0.297 | 0.98 | 1.04 | 65.0 | 5.8e-08 | PASS |
| silu | Wgu | 4096 | 22016 | 4096 | 0.596 | 0.674 | 0.656 | 1.13 | 1.10 | 360.7 | 3.3e-03 | PASS |
| swiglu | Wgu | 4096 | 22016 | 4096 | 0.565 | 0.701 | 0.676 | 1.24 | 1.20 | 541.1 | 3.3e-03 | PASS |
| rmsnorm_swiglu | Wgu | 4096 | 22016 | 4096 | 0.561 | 0.737 | 0.707 | 1.32 | 1.26 | 675.3 | 4.2e-03 | PASS |
| noop | Wq | 8192 | 4096 | 4096 | 0.201 | 0.204 | 0.204 | 1.01 | 1.01 | 0.0 | 3.3e-03 | PASS |
| noop | Wk | 8192 | 1024 | 4096 | 0.096 | 0.071 | 0.095 | 0.74 | 0.99 | 0.0 | 3.3e-03 | PASS |
| noop | Wo | 8192 | 4096 | 4096 | 0.202 | 0.204 | 0.203 | 1.01 | 1.00 | 0.0 | 3.3e-03 | PASS |
| noop | Wgu | 8192 | 22016 | 4096 | 1.076 | 1.023 | 1.071 | 0.95 | 0.99 | 0.0 | 3.3e-03 | PASS |
| noop | Wd | 8192 | 4096 | 11008 | 0.541 | 0.497 | 0.541 | 0.92 | 1.00 | 0.0 | 3.3e-03 | PASS |
| rope | Wq | 8192 | 4096 | 4096 | 0.229 | 0.411 | 0.410 | 1.79 | 1.79 | 536.9 | 1.7e-03 | PASS |
| rope | Wk | 8192 | 1024 | 4096 | 0.102 | 0.134 | 0.157 | 1.31 | 1.54 | 134.2 | 1.7e-03 | PASS |
| rmsnorm_rope | Wq | 8192 | 4096 | 4096 | 0.232 | 0.464 | 0.465 | 2.00 | 2.01 | 805.3 | 2.3e-03 | PASS |
| rmsnorm_rope | Wk | 8192 | 1024 | 4096 | 0.102 | 0.196 | 0.213 | 1.91 | 2.08 | 402.7 | 2.3e-03 | PASS |
| scale | Wv | 8192 | 1024 | 4096 | 0.099 | 0.077 | 0.101 | 0.78 | 1.02 | 33.6 | 3.3e-03 | PASS |
| rmsnorm_scale | Wv | 8192 | 1024 | 4096 | 0.097 | 0.093 | 0.116 | 0.96 | 1.20 | 67.1 | 3.3e-03 | PASS |
| rmsnorm | Wv | 8192 | 1024 | 4096 | 0.097 | 0.135 | 0.151 | 1.39 | 1.56 | 268.4 | 3.8e-03 | PASS |
| residual_add | Wo | 8192 | 4096 | 4096 | 0.226 | 0.238 | 0.238 | 1.05 | 1.05 | 134.2 | 3.3e-03 | PASS |
| residual_add | Wd | 8192 | 4096 | 11008 | 0.558 | 0.534 | 0.572 | 0.96 | 1.02 | 134.2 | 3.3e-03 | PASS |
| residual_rms | Wo | 8192 | 4096 | 4096 | 0.239 | 0.261 | 0.262 | 1.09 | 1.10 | 197.1 | 3.3e-03 | PASS |
| residual_rms | Wd | 8192 | 4096 | 11008 | 0.572 | 0.562 | 0.597 | 0.98 | 1.04 | 197.1 | 3.3e-03 | PASS |
| inv_rms | Wo | 8192 | 4096 | 4096 | 0.208 | 0.225 | 0.224 | 1.08 | 1.08 | 130.0 | 4.9e-08 | PASS |
| inv_rms | Wd | 8192 | 4096 | 11008 | 0.544 | 0.526 | 0.564 | 0.97 | 1.04 | 130.0 | 5.7e-08 | PASS |
| silu | Wgu | 8192 | 22016 | 4096 | 1.108 | 1.168 | 1.236 | 1.05 | 1.11 | 721.4 | 3.3e-03 | PASS |
| swiglu | Wgu | 8192 | 22016 | 4096 | 1.032 | 1.246 | 1.291 | 1.21 | 1.25 | 1082.1 | 3.3e-03 | PASS |
| rmsnorm_swiglu | Wgu | 8192 | 22016 | 4096 | 1.039 | 1.293 | 1.348 | 1.24 | 1.30 | 1350.6 | 4.2e-03 | PASS |

Spread: median min–max width 0.016 for `eager/HK` and 0.007 for `fusion-only/HK`; widest row
`rmsnorm` on Wv at M=8192 (`eager/HK` 1.32–1.42, `fusion-only/HK` 1.49–1.60).

### Layer phases

| M | phase | HK launches | HK ms | eager ms | compile ms | eager/HK | compile/HK | rel err | status |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | qkv | 3 | 0.258 | 0.256 | 0.151 | 0.99 | 0.58 | 2.3e-03 | PASS |
| 2048 | attention | 1 | 0.063 | 0.154 | 0.152 | 2.44 | 2.41 | 2.5e-03 | PASS |
| 2048 | out_proj | 2 | 0.113 | 0.077 | 0.081 | 0.68 | 0.72 | 3.3e-03 | PASS |
| 2048 | gate_up | 1 | 0.308 | 0.356 | 0.324 | 1.16 | 1.05 | 9.6e-03 | PASS |
| 2048 | down | 2 | 0.282 | 0.213 | 0.222 | 0.76 | 0.79 | 3.3e-03 | PASS |
| 2048 | sum | 9 | 1.024 | 1.056 | 0.929 | 1.03 | 0.91 | - | PASS |
| 4096 | qkv | 3 | 0.278 | 0.376 | 0.234 | 1.35 | 0.84 | 2.3e-03 | PASS |
| 4096 | attention | 1 | 0.171 | 0.450 | 0.443 | 2.62 | 2.58 | 2.6e-03 | PASS |
| 4096 | out_proj | 2 | 0.131 | 0.133 | 0.126 | 1.02 | 0.97 | 3.3e-03 | PASS |
| 4096 | gate_up | 1 | 0.573 | 0.727 | 0.668 | 1.27 | 1.17 | 9.6e-03 | PASS |
| 4096 | down | 2 | 0.320 | 0.300 | 0.294 | 0.94 | 0.92 | 3.3e-03 | PASS |
| 4096 | sum | 9 | 1.472 | 1.985 | 1.765 | 1.35 | 1.20 | - | PASS |
| 8192 | qkv | 3 | 0.395 | 0.642 | 0.400 | 1.63 | 1.01 | 2.3e-03 | PASS |
| 8192 | attention | 1 | 0.551 | 1.296 | 1.296 | 2.35 | 2.35 | 2.6e-03 | PASS |
| 8192 | out_proj | 2 | 0.247 | 0.238 | 0.243 | 0.96 | 0.98 | 3.3e-03 | PASS |
| 8192 | gate_up | 1 | 1.045 | 1.294 | 1.152 | 1.24 | 1.10 | 9.6e-03 | PASS |
| 8192 | down | 2 | 0.587 | 0.571 | 0.569 | 0.97 | 0.97 | 3.3e-03 | PASS |
| 8192 | sum | 9 | 2.824 | 4.041 | 3.661 | 1.43 | 1.30 | - | PASS |

Spread of the `sum` rows: `compile/HK` 0.90–0.91 (M=2048), 1.19–1.20 (M=4096), 1.30–1.30 (M=8192);
`eager/HK` 1.03–1.04, 1.34–1.35, 1.43–1.43.

### Full layer

| M | HK ms | eager ms | compile ms | eager/HK | compile/HK | compile full - phases ms | rel HK | rel compile | status |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | 0.927 | 1.014 | 0.882 | 1.09 | 0.95 | -0.047 | 8.4e-03 | 3.3e-03 | PASS |
| 4096 | 1.388 | 1.914 | 1.729 | 1.38 | 1.25 | -0.036 | 8.4e-03 | 3.2e-03 | PASS |
| 8192 | 2.724 | 4.054 | 3.780 | 1.49 | 1.39 | +0.120 | 8.4e-03 | 3.2e-03 | PASS |

Spread: `compile/HK` 0.94–0.95 (M=2048), 1.24–1.25 (M=4096), 1.39–1.40 (M=8192); `eager/HK` 1.09–1.10,
1.37–1.38, 1.49–1.50; `compile full - phases` −0.053..−0.047, −0.045..−0.036, +0.109..+0.120 ms.
