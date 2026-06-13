#pragma once
#include "base.cuh"   // store_C, gl typedefs (via epilogue_args.cuh)
#include "vec.cuh"    // apply_scale

// Per-output scalar scale:  out = alpha * (A@B).
struct ScaleGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<float,1,1,1,1> alpha{nullptr,nullptr,nullptr,nullptr,nullptr};
    hipStream_t stream;
};
struct ScaleEpilogue {
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row,int col,int wr,int wc){
        apply_scale(g, C);                       // alpha * C  (tile x 1-elem-gl scalar)
        store_C(g, C, row, col, wr, wc);         // epilogue owns the store
    }
};
