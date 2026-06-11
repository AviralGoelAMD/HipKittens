#pragma once
#include <type_traits>
#include "epilogue_args.cuh"
using namespace kittens;

// Activation epilogue ops. Defined LOCALLY here, not in core base_ops.cuh
// (design principle: silu lives in the epilogue, not the core tile library).
//
// silu_op: SiLU / swish, in-place on a register accumulator tile.
//   silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Implemented with the register-tile maps (mul/exp/add/div); one temp tile of the
// same type. Building block for SwiGLU: out = silu(gate) * value.
template<typename T>
__device__ inline void silu_op(T& x) {
    T t;
    mul(t, x, -1.0f);     // t = -x
    exp(t, t);            // t = exp(-x)
    add(t, t,  1.0f);     // t = 1 + exp(-x)
    div(x, x, t);         // x = x / (1 + exp(-x)) = silu(x)
}
