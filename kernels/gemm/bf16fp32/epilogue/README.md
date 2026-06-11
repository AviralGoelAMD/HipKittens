# bf16→fp32 GEMM epilogues

One hand-scheduled GEMM mainloop (`gemm_base.cuh`, `micro_tk`) made generic over a compile-time
**epilogue**: a small struct whose `apply()` runs between the mainloop and the stores, fusing
elementwise / normalization / reduction work into the GEMM so the [M,N] intermediate never
round-trips HBM. No virtual calls — each epilogue is a separate template instantiation.

## How to add an epilogue

A new epilogue is **one new file** in `bindings/` plus **one entry** in `util/epilogue_testlib.py`.
Touch nothing else (not `gemm_base.cuh`, not `epilogue_args.cuh`, not other bindings).

1. **Globals struct** — the kernel inputs, in bind order: `a, b, c` first, then this epilogue's
   own `gl` fields, then a trailing `hipStream_t stream`. (No extra inputs? reuse `gemm_args_base`.)
   ```cpp
   struct MyGlobals {
       _gl_A a; _gl_B b; _gl_C c;
       gl<bf16,-1,-1,-1,-1> my_input;   // your per-op tensor(s)
       hipStream_t stream;
   };
   ```
2. **Epilogue struct** — a static `apply` that transforms the accumulator and stores it. Read your
   inputs off `g` (e.g. `g.my_input`); reuse the ops in `ops/` (`apply_inv_rms`, `apply_gamma`,
   `residual_add`, `partial_row_sum_sq`, `silu_op`, …) and end with `store_C` (or your own store).
   ```cpp
   struct MyEpilogue {
       template<typename G, typename Accum>
       static __device__ inline void apply(const G& g, Accum& C, int row,int col,int wr,int wc){
           /* transform C using g.my_input */
           store_C(g, C, row, col, wr, wc);
       }
   };
   ```
   The `apply` signature is enforced at compile time by a `static_assert` in `gemm_kernel`; a
   malformed epilogue gives *"Epilogue must define: static apply(...)"*, not a template-spew.
3. **Dispatch + binding** — one line each; `launch` does the shape preconditions + checked
   HIP launch. Bind base + your own fields only (no positional dummy slots).
   ```cpp
   void dispatch(MyGlobals g) { launch<MyEpilogue, MyGlobals>(g); }
   PYBIND11_MODULE(TK_MODULE_NAME, m) {
       py::bind_function<dispatch>(m, "dispatch",
           &MyGlobals::a, &MyGlobals::b, &MyGlobals::c, &MyGlobals::my_input);
   }
   ```
4. **Registry entry** — add `"myop": {module, args, ref, identity, sweep, label, hbm_passes}` to
   `EPILOGUES` in `util/epilogue_testlib.py`. It is then automatically correctness-tested by
   the unified suite (`python3 util/test_all.py`) and benched by `util/bench.py` — no new script.
5. **Build + test (gfx950):** `util/build.sh myop` then `python3 util/test_all.py`.

## Invariants enforced

- **Shapes:** `M, N` multiples of `BLOCK_SIZE` (256); `K` a multiple of 128 — `launch_micro`
  throws otherwise (a bad `K` would silently corrupt results). Tiling relations are
  `static_assert`-checked in `epilogue_args.cuh`.
- **HIP errors:** every HIP call in `launch_micro` is wrapped in HK's `CHECK_CUDA_ERROR`
  (`pyutils/util.cuh`), which reports `file:line` + the HIP error string.
- **Layout:** the tile↔global coordinate fan-out is a single helper, `block_coords` in
  `ops/epilogue_base.cuh`; `store_C` / `residual_add` / `save_tile` all derive from it.

## Testing

- `python3 util/test_all.py` — the single correctness suite: base GEMM, every registry epilogue
  (identity + sweep), the multi-output kernels (partialrms/residual_rms/aux), the fused chain, and
  math invariants. Deterministic (seeded); each case prints PASS/FAIL.
- `python3 util/bench.py` — the benchmark: HK fused vs torch (and Triton for the chain), cold-cache
  median + the HBM bytes the fusion saves.
- Tolerance: bf16 outputs vs an fp32 reference at `rtol=1e-2, atol=1e-1`; the chain uses a normwise
  relative error (`2e-2`). Runners seed `torch.manual_seed(0)`.
- `util/epilogue_testlib.py` is the **registry only** (definitions); `test_all.py` (correctness) and
  `bench.py` (performance) both import it. Do not merge a correctness assertion and a timing loop
  into one script.

## Conventions
- **Naming:** epilogue structs `<Operation>Epilogue`; their launch args `<Operation>Globals`; module
  / binding file / registry key all agree (`tk_<op>` / `gemm_<op>.cpp` / `"<op>"`). No paper-index
  names (k4/k5) or vestigial `micro_*`.
- **Constants:** no magic numbers in logic — tiling, `RMS_EPS`, and `SUBTILES_PER_DIM` are
  `constexpr` in `epilogue_args.cuh`.
- **Comments:** production voice — explain the *why* (axis/layout invariants); no dev-log tags
  (`[Cx]`, Stage/Task) or paper-kernel labels.
