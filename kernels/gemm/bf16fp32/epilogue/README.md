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
   The `apply` signature is enforced at compile time by a `static_assert` in `micro_tk`; a
   malformed epilogue gives *"Epilogue must define: static apply(...)"*, not a template-spew.
3. **Dispatch + binding** — one line each; `launch_micro` does the shape preconditions + checked
   HIP launch. Bind base + your own fields only (no positional dummy slots).
   ```cpp
   void dispatch_micro(MyGlobals g) { launch_micro<MyEpilogue, MyGlobals>(g); }
   PYBIND11_MODULE(TK_MODULE_NAME, m) {
       py::bind_function<dispatch_micro>(m, "dispatch_micro",
           &MyGlobals::a, &MyGlobals::b, &MyGlobals::c, &MyGlobals::my_input);
   }
   ```
4. **Registry entry** — add `"myop": {module, args, ref, identity, sweep, label, hbm_passes}` to
   `EPILOGUES` in `util/epilogue_testlib.py`. It is then automatically correctness-tested by
   `util/test_epilogue.py myop` and benched by `util/fusion_win.py` — no new script.
5. **Build + test (gfx950):** `util/build.sh myop` then `python3 util/test_epilogue.py myop`.

## Invariants enforced

- **Shapes:** `M, N` multiples of `BLOCK_SIZE` (256); `K` a multiple of 128 — `launch_micro`
  throws otherwise (a bad `K` would silently corrupt results). Tiling relations are
  `static_assert`-checked in `epilogue_args.cuh`.
- **HIP errors:** every HIP call in `launch_micro` is wrapped in HK's `CHECK_CUDA_ERROR`
  (`pyutils/util.cuh`), which reports `file:line` + the HIP error string.
- **Layout:** the tile↔global coordinate fan-out is a single helper, `block_coords` in
  `ops/epilogue_base.cuh`; `store_C` / `residual_add` / `save_tile` all derive from it.

## Testing

- `util/test_gemm.py` — base GEMM vs fp32 ground truth.
- `util/test_epilogue.py <name>` — registry-driven per-epilogue correctness (identity bit-exact +
  param sweep), isolated against the no-op GEMM baseline.
- `util/test_stage2.py`, `util/test_block.py` — the multi-kernel money-pattern chain.
- Tolerance: bf16 outputs compared to an fp32 reference at `rtol=1e-2, atol=1e-1`; runners seed
  `torch.manual_seed(0)` so results are reproducible.
- `util/epilogue_testlib.py` is the **registry only** (definitions); correctness (`test_*`) and
  performance (`fusion_win.py`) are separate runners that both import it. Do not merge a
  correctness assertion and a timing loop into one script.

<!-- The naming convention, magic-number policy, and comment style are finalized in later passes. -->
