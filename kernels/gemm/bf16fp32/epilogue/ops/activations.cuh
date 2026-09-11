#pragma once
#include <type_traits>
#include "epilogue_args.cuh"
using namespace kittens;

// Activation epilogue ops. Defined LOCALLY here, not in core base_ops.cuh
// (design principle: silu lives in the epilogue, not the core tile library).
//
// silu_op: SiLU / swish, in-place on a register accumulator tile.
//   silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Building block for SwiGLU: out = silu(gate) * value.
//
// -------------------------------------------------------------------------------------------
// TWO IMPLEMENTATIONS, selected at compile time by -DCODA_OPT_FAST_SILU.
//
// BASELINE (default): the tile maps mul/exp/add/div.
//   `div` is base_ops::div -> `a/b` on float. The build has no -ffast-math, so that is an IEEE
//   correctly-rounded divide, which clang expands into the v_div_scale_f32 / v_rcp_f32 /
//   v_div_fmas_f32 / v_div_fixup_f32 sequence -- roughly 13 instructions per element for the
//   whole silu, most of them the divide.
//
// FAST (-DCODA_OPT_FAST_SILU): the same function, evaluated as
//   silu(x) = x * rcp(1 + exp2(-x * log2(e)))
// which is FIVE hardware instructions per element:
//   v_mul (by -log2 e) -> v_exp_f32 -> v_add (+1) -> v_rcp_f32 -> v_mul
//
// WHY IT WINS -- and it is NOT the reason you would guess. MEASURED on gfx950 (clang 23):
//
//   tk_silu  baseline : 256 VGPRs, 12 VGPR spills, 52 B/lane scratch
//   tk_silu  fast     : 256 VGPRs, 20 VGPR spills, 84 B/lane scratch   <-- MORE spilling
//
// and it is still 1.13x faster at (2048,1024,512). So the win is **instruction count**, not
// register pressure: ~8 fewer VALU ops per element beats the extra spill traffic. The allocator
// simply spent some of the freed pressure elsewhere.
//
// Note this is NOT a register-pressure win, which is what you would expect and what I first
// assumed. Removing ~8 VALU ops per element simply outweighs the extra spill traffic. So neither
// "fewer instructions" nor "fewer spills" is the right thing to optimize on its own; only the
// balance decides, and only measurement settles it.
//
// ACCURACY -- measured on gfx950, not asserted. util/silu_ab.py runs identical inputs through
// both builds (they share a bit-identical GEMM, so the delta is silu alone) and scores each
// against a float64 oracle; util/silu_precision_probe.cpp does the same for the two formulas
// in isolation, including an exhaustive sweep of all 65280 finite bf16-representable inputs.
//
//   normwise relative error vs float64, 4096^3 : fast 4.061e-03   original 4.061e-03
//   max per-element relative error             : fast 3.053e-02   original 3.053e-02
//   exhaustive bf16 sweep, max ULP vs float64  : identical for both; 4 of 65280 outputs differ
//
// Both sit on bf16's own rounding floor (~3.9e-3), so at tensor level the builds are
// indistinguishable. 0.34% of individual bf16 values do differ, and the split matters:
//   * 99.95% are FLUSH-TO-ZERO. For very negative x, exp(-x) is enormous and v_rcp_f32 of it
//     underflows, so this path returns exactly 0 where the exact answer is a denormal. The
//     largest magnitude ever discarded is 1.02e-36 -- it vanishes in the first bf16 add.
//   * the rest are ordinary rounding flips, splitting 14 fast-closer to 12 original-closer
//     (a wash), and NONE exceeds one bf16 step in magnitude (worst 0.92 steps).
//
// CAVEAT worth carrying: flush-to-zero is harmless HERE because silu feeds a multiply (SwiGLU)
// and then a GEMM. A consumer that DIVIDED by a silu output would turn 1e-36 into inf. That is
// a property of the downstream graph, and it is true of any rcp-based activation.
// -------------------------------------------------------------------------------------------

#ifdef CODA_OPT_FAST_SILU
namespace coda_ops {
// sigmoid via the hardware reciprocal: rcp(1 + exp2(-x*log2e)). Local to the epilogue.
struct fast_silu {
    static constexpr float NEG_LOG2E = -1.4426950408889634f;
    template<typename T> static __device__ inline T op(const T &x);
};
template<> __device__ inline float fast_silu::op<float>(const float &x) {
    return x * __builtin_amdgcn_rcpf(1.0f + __builtin_amdgcn_exp2f(x * NEG_LOG2E));
}
template<> __device__ inline float2 fast_silu::op<float2>(const float2 &x) {
    return float2{ fast_silu::op<float>(x.x), fast_silu::op<float>(x.y) };
}
}  // namespace coda_ops
#endif

template<typename T>
__device__ inline void silu_op(T& x) {
#ifdef CODA_OPT_FAST_SILU
    unary_map<coda_ops::fast_silu, T>(x, x);   // x * rcp(1 + exp2(-x*log2e))
#else
    T t;
    mul(t, x, -1.0f);     // t = -x
    exp(t, t);            // t = exp(-x)
    add(t, t,  1.0f);     // t = 1 + exp(-x)
    div(x, x, t);         // x = x / (1 + exp(-x)) = silu(x)
#endif
}
