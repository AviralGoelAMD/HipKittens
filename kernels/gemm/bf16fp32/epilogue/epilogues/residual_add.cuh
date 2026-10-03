#pragma once
#include <type_traits>
#include "base.cuh"   // block_coords
#include "tile.cuh"   // residual_add (still used by residual_rms)

// out = (A@B) + residual   (the [M,N] skip connection), fused onto the GEMM epilogue so the
// intermediate never round-trips HBM.
struct ResidualAddGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<bf16,-1,-1,-1,-1> residual;   // [1,1,M,N] skip connection
};
struct ResidualAddEpilogue {
    // Load, add and store ONE sub-tile at a time rather than adding into all four and storing
    // afterwards: storing early ends each sub-tile's live range instead of holding all 128
    // accumulator VGPRs plus the residual temp to the end (256 VGPRs, 16 spills). The store of
    // sub-tile n and the load of n+1 are independent, so the memory pipeline stays fed.
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row,int col,int wr,int wc){
        using Tile = std::remove_all_extents_t<Accum>;
        subtile_coords co = block_coords(row,col,wr,wc);
        #pragma unroll
        for (int i = 0; i < SUBTILES_PER_DIM; ++i) {
            #pragma unroll
            for (int j = 0; j < SUBTILES_PER_DIM; ++j) {
                Tile t;
                load(t, g.residual, {0,0,co.m[i],co.n[j]});
                add(C[i][j], C[i][j], t);
                store(g.c, C[i][j], {0,0,co.m[i],co.n[j]});
            }
        }
    }
};
