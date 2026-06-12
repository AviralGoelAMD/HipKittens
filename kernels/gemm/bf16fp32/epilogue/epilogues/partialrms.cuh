#pragma once
#include "base.cuh"        // gl typedefs (via epilogue_args.cuh)
#include "reductions.cuh"  // partial_row_sum_sq

// Per-(row, REG_BLOCK_N-col group) partial sum-of-squares for RMSNorm. Emits ONLY `partials`
// (no C store); the aux kernel sums the groups per row -> Sigma(x^2) -> 1/rms.
struct PartialRMSGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<float,-1,-1,-1,-1> partials;   // [1,1,N/REG_BLOCK_N,M]
    hipStream_t stream;
};
struct PartialRMSEpilogue {
    template<typename G, typename Accum>
    static __device__ inline void apply(const G& g, Accum& C, int row,int col,int wr,int wc){
        partial_row_sum_sq(g, C, row,col,wr,wc);   // store partials only; no store_C
    }
};
