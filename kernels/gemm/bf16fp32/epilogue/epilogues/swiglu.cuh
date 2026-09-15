#pragma once
#include "base.cuh"          // store_swiglu, gl typedefs (via epilogue_args.cuh)
#include "activations.cuh"   // silu_op
using namespace kittens;

// SwiGLU (dim-reducing):  out = silu(gate) * value, where gate = C[*][0], value = C[*][1]
// (made register-co-resident by the one-time gate_up weight permutation: the pair lands 128 cols
// apart in a single lane -> silu(gate)*value is register-local).
// Reuses gemm_args_base (no extra inputs); output c is [M, d_ff], b is the [2*d_ff, K] weight.
struct SwigluEpilogue {
    static constexpr int out_cols(int n) { return n / 2; }   // dim-reducing: c is [M, d_ff] = N/2
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row,int col,int wr,int wc){
        silu_op(C[0][0]); silu_op(C[1][0]);          // gate <- silu(gate)
        mul(C[0][0], C[0][0], C[0][1]);              // gate * value  (register-local co-resident pair)
        mul(C[1][0], C[1][0], C[1][1]);
        store_swiglu(g, C, row, col, wr, wc);        // half-width store -> [M, d_ff]
    }
};
