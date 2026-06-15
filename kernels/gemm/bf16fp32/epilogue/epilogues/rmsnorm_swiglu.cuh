#pragma once
#include "base.cuh"          // store_swiglu
#include "vec.cuh"           // apply_inv_rms
#include "activations.cuh"   // silu_op
using namespace kittens;

// RMS -> SwiGLU (dim-reducing):  out = silu(gate) * value  where  [gate | value] = r * (A @ B).
// `r` is the precomputed per-row inv-rms of the pre-norm activation; the norm's per-d_model gamma is
// folded (host-side) into B's rows, and B's columns are gate_up-permuted, so gate = C[*][0] and
// value = C[*][1] are register-co-resident (128 cols apart, one lane). r post-scales the GEMM output
// (rmsnorm(X,gamma)@B == r * (X @ gamma-folded B)), then SwiGLU runs register-local.
struct RmsnormSwigluGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<bf16,-1,-1,-1,-1> r;   // per-row inv_rms, [1,1,1,M] (M on the last axis)
    hipStream_t stream;
};
struct RmsnormSwigluEpilogue {
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row,int col,int wr,int wc){
        apply_inv_rms(g, C, row,col,wr,wc);   // r * (A@B), per-row (col_vec, mul_row)
        silu_op(C[0][0]); silu_op(C[1][0]);   // gate <- silu(gate)
        mul(C[0][0], C[0][0], C[0][1]);       // gate * value  (register-local co-resident pair)
        mul(C[1][0], C[1][0], C[1][1]);
        store_swiglu(g, C, row,col,wr,wc);    // half-width store -> [M, d_ff]
    }
};
