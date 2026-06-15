#pragma once
#include "base.cuh"      // store_C, gl typedefs (via epilogue_args.cuh)
#include "rotary.cuh"    // apply_rope
using namespace kittens;

// out = RoPE(A@B), interleaved pairs (2k, 2k+1), fused onto the GEMM epilogue.
//
// The caller pre-permutes the projection weight B AND cos_sin with rope_perm so each pair
// co-resides in one lane (C[r][0], C[r][1]); the rotation is register-only (apply_rope). The result
// is stored in permuted column order via the default store_C -- valid because the consumer is
// attention's Q*K^T, which is invariant to a shared feature permutation of Q and K (V is natural).
// cos_sin is the rope_perm'd [1,1,M,N] interleaved [cos,sin] tensor.
struct RopeGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<bf16,-1,-1,-1,-1> cos_sin;
    hipStream_t stream;
};
struct RopeEpilogue {
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row,int col,int wr,int wc){
        apply_rope(g, C, row,col,wr,wc);
        store_C(g, C, row,col,wr,wc);
    }
};
