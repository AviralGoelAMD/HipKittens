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
// Evaluated as x * rcp(1 + exp2(-x*log2e)): 5 instructions rather than ~13, since base_ops::div
// is an IEEE divide (no -ffast-math) and v_exp_f32 is natively base 2, so log2(e) folds into the
// negation. ~1 ULP, far under bf16's 2^-7 step. Very negative x flushes to 0 (v_rcp_f32
// underflows) instead of a ~1e-36 denormal -- harmless downstream of a multiply and a GEMM.
namespace coda_ops {
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

template<typename T>
__device__ inline void silu_op(T& x) {
    unary_map<coda_ops::fast_silu, T>(x, x);
}
