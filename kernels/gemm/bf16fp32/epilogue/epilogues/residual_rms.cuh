#pragma once
#include "base.cuh"        // store_C
#include "tile.cuh"        // residual_add, save_tile
#include "reductions.cuh"  // partial_row_sum_sq
#include "vec.cuh"         // apply_gamma

// Residual + RMS-partials GEMM: h1 = A@B + residual; save h1 (for the BACKWARD pass); emit
// Sigma(h1^2) partials (for the aux 1/rms); store c = h1*gamma. The aux kernel turns partials
// -> r = 1/rms; the RMSNorm-scale GEMM reads c (gamma already folded in), post-applying r.
struct ResidualRMSPartialsGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<bf16,-1,-1,-1,-1>  residual;   // [1,1,M,N] skip connection
    gl<bf16,-1,-1,-1,-1>  gamma;      // [1,1,1,N] per-feature gamma
    gl<float,-1,-1,-1,-1> partials;   // [1,1,N/REG_BLOCK_N,M] per-(group,row) Sigma(h1^2)
    gl<bf16,-1,-1,-1,-1>  save;       // [1,1,M,N] saved h1
    hipStream_t stream;
};
struct ResidualRMSPartialsEpilogue {
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row,int col,int wr,int wc){
        residual_add(g, C, row,col,wr,wc);        // C = A@B + residual = h1
        save_tile(g, C, row,col,wr,wc);           // persist h1 -> g.save
        partial_row_sum_sq(g, C, row,col,wr,wc);  // Sigma(h1^2) per (row, col group) -> g.partials
        apply_gamma(g, C, row,col,wr,wc);         // C = h1 * gamma
        store_C(g, C, row,col,wr,wc);             // -> g.c
    }
};
