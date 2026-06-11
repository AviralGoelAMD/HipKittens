#pragma once
#include "epilogue_args.cuh"
using namespace kittens;
// Coordinate helper: the four {row,col} subtile coordinates an 8-warp block's accumulator
// fans out to. Single source for the tile<->global mapping; store_C / residual_add / save_tile
// all derive their coords from it, so a layout change is one edit and they cannot drift.
struct subtile_coords { int m[2]; int n[2]; };
__device__ inline subtile_coords block_coords(int row,int col,int wr,int wc){
    return { {(row*2)*WARPS_M+wr, (row*2)*WARPS_M+WARPS_M+wr},
             {col*2*WARPS_N+wc,  col*2*WARPS_N+WARPS_N+wc} };
}
// the four default full-tile stores; reused by every dim-preserving epilogue
template<typename G, typename Accum>
__device__ inline void store_C(const G& g, const Accum& C, int row,int col,int wr,int wc){
    subtile_coords co = block_coords(row,col,wr,wc);
    store(g.c, C[0][0], {0,0,co.m[0],co.n[0]});
    store(g.c, C[0][1], {0,0,co.m[0],co.n[1]});
    store(g.c, C[1][0], {0,0,co.m[1],co.n[0]});
    store(g.c, C[1][1], {0,0,co.m[1],co.n[1]});
}
// Identity epilogue (store only). Lives here, NOT in bindings/gemm_noop.cpp like the other
// epilogue structs, because it is gemm_kernel's DEFAULT Epilogue
// (gemm_base.cuh: template<typename Epilogue = NoOpEpilogue>), which every binding includes --
// moving it would leave that default referencing an undefined type and break every binding.
struct NoOpEpilogue {
    template<typename G, typename Accum>
    static __device__ inline void apply(const G& g, Accum& C, int row,int col,int wr,int wc){
        store_C(g, C, row, col, wr, wc);
    }
};
