#pragma once
#include "epilogue_args.cuh"
using namespace kittens;
// Coordinate helper: the four {row,col} subtile coordinates an 8-warp block's accumulator
// fans out to. Single source for the tile<->global mapping; every store/load/reduce op in ops/
// derives its coords from it, so a layout change is one edit and they cannot drift.
// The two row / two col sub-tile origins of the FIXED 2x2 fan-out (length-2, indexed [0]/[1] by every consumer).
struct subtile_coords { int m[SUBTILES_PER_DIM]; int n[SUBTILES_PER_DIM]; };
__device__ inline subtile_coords block_coords(int row,int col,int wr,int wc){
    return { {(row*SUBTILES_PER_DIM)*WARPS_M+wr, (row*SUBTILES_PER_DIM)*WARPS_M+WARPS_M+wr},
             {col*SUBTILES_PER_DIM*WARPS_N+wc,   col*SUBTILES_PER_DIM*WARPS_N+WARPS_N+wc} };
}
// the four default full-tile stores; reused by every dim-preserving epilogue
template<typename Globals, typename Accum>
__device__ inline void store_C(const Globals& g, const Accum& C, int row,int col,int wr,int wc){
    subtile_coords co = block_coords(row,col,wr,wc);
    store(g.c, C[0][0], {0,0,co.m[0],co.n[0]});
    store(g.c, C[0][1], {0,0,co.m[0],co.n[1]});
    store(g.c, C[1][0], {0,0,co.m[1],co.n[0]});
    store(g.c, C[1][1], {0,0,co.m[1],co.n[1]});
}
// Half-width store for the dim-reducing SwiGLU epilogue. The reduced result lives in the gate
// sub-tiles C[*][0] (block-col co.n[0]); the partner value sub-tile C[*][1] (co.n[0]+128) is
// already consumed. Output is [M, d_ff]: the gate sub-tile at the gate half of its 256-block maps
// to output col b*128+c (natural feature order). In HALF_REG_BLOCK_N sub-tile units:
template<typename Globals, typename Accum>
__device__ inline void store_swiglu(const Globals& g, const Accum& C, int row,int col,int wr,int wc){
    constexpr int NSUB_BLOCK = BLOCK_SIZE / HALF_REG_BLOCK_N;        // 8 sub-tiles span one 256 block
    constexpr int NSUB_HALF  = (BLOCK_SIZE / 2) / HALF_REG_BLOCK_N;  // 4 sub-tiles span the 128 gate half
    subtile_coords co = block_coords(row,col,wr,wc);
    int o0 = (co.n[0] / NSUB_BLOCK) * NSUB_HALF + (co.n[0] % NSUB_BLOCK);
    store(g.c, C[0][0], {0,0,co.m[0], o0});
    store(g.c, C[1][0], {0,0,co.m[1], o0});
}
