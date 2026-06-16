#pragma once
#include "base.cuh"      // store_C
#include "vec.cuh"       // apply_inv_rms
#include "rotary.cuh"    // apply_rope
using namespace kittens;

// RMS -> RoPE (dim-preserving):  out = RoPE( r * (A @ B) ), interleaved pairs (2k, 2k+1).
// `r` is the precomputed per-row inv-rms of the pre-norm activation; the norm's per-d_model gamma is
// folded (host-side) into B's rows, and B's columns + cos_sin are rope_perm'd so each pair co-resides
// in one lane. apply_inv_rms post-scales the GEMM output (rmsnorm(X,gamma)@B == r * (X @ gamma-folded
// B)), then apply_rope rotates register-local (r commutes with the rotation). Stored in permuted
// column order (store_C) -- valid because attention Q*K^T is invariant to a shared Q/K feature perm.
struct RmsnormRopeGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<bf16,-1,-1,-1,-1> r;         // per-row inv_rms, [1,1,1,M] (M on the last axis)
    gl<bf16,-1,-1,-1,-1> cos_sin;   // rope_perm'd interleaved [cos,sin], [1,1,M,N]
    hipStream_t stream;
};
struct RmsnormRopeEpilogue {
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row,int col,int wr,int wc){
        apply_inv_rms(g, C, row,col,wr,wc);   // r * (A@B), per-row (col_vec, mul_row)
        apply_rope(g, C, row,col,wr,wc);      // rotate co-resident interleaved pairs
        store_C(g, C, row,col,wr,wc);         // dim-preserving store (permuted column order)
    }
};
