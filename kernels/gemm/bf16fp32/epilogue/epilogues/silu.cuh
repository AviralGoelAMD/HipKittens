#pragma once
#include "base.cuh"        // store_C, gemm_args_base (via epilogue_args.cuh)
#include "activations.cuh" // silu_op

// SiLU activation epilogue (dim-preserving):  out = silu(A@B). No extra inputs -> gemm_args_base.
struct SiluEpilogue {
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row,int col,int wr,int wc){
        silu_op(C[0][0]); silu_op(C[0][1]);   // x <- silu(x), register-only
        silu_op(C[1][0]); silu_op(C[1][1]);
        store_C(g, C, row, col, wr, wc);
    }
};
