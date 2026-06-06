#pragma once
#include <type_traits>
#include "epilogue_args.cuh"
using namespace kittens;

// Tile (rank-2) epilogue ops on the col_l accumulator. Load coords MIRROR store_C exactly
// (the same {batch,depth,row_tile,col_tile} the accumulator subtiles store to), so the loaded
// tile lines up element-for-element with each subtile.
//
// residual_add: C += residual, where `residual` is a full [M,N] bf16 tile (the skip
// connection). Each subtile is loaded from g.residual with store_C's coords, converting
// bf16 -> fp32 on the way in (col_layout load, global_to_register.cuh:134), then added.
template<typename G, typename Accum>
__device__ inline void residual_add(const G& g, Accum& C, int row,int col,int wr,int wc){
    using Tile = std::remove_all_extents_t<Accum>;     // rt_fl<64,32,col_l,rt_16x16_s>
    Tile t;
    load(t, g.residual, {0,0,(row*2)*WARPS_M+wr,         col*2*WARPS_N+wc});         add(C[0][0], C[0][0], t);
    load(t, g.residual, {0,0,(row*2)*WARPS_M+wr,         col*2*WARPS_N+WARPS_N+wc}); add(C[0][1], C[0][1], t);
    load(t, g.residual, {0,0,(row*2)*WARPS_M+WARPS_M+wr, col*2*WARPS_N+wc});         add(C[1][0], C[1][0], t);
    load(t, g.residual, {0,0,(row*2)*WARPS_M+WARPS_M+wr, col*2*WARPS_N+WARPS_N+wc}); add(C[1][1], C[1][1], t);
}
