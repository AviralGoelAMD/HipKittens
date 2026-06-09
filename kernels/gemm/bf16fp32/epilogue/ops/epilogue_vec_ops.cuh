#pragma once
#include <type_traits>
#include "epilogue_args.cuh"
using namespace kittens;

// Vector-broadcast epilogue ops on the col_l accumulator tiles ([C2]).
//
// per-ROW scale (1/rms): one value per row -> a COLUMN-shaped vector (col_vec) + mul_row.
// per-COL scale (gamma): one value per col -> a ROW-shaped vector    (row_vec) + mul_col.
// (The name is the vector's SHAPE, not what it scales — per-row uses col_vec. This is [C2].)
//
// Load coords MIRROR store_C exactly so r[i]/gamma[j] line up with the rows/cols each
// register subtile actually holds.

// per-row 1/rms : r is gl[1,1,1,M] (M on the LAST axis, like scaled_matmul's scale_a); col_vec +
// mul_row. Single-index tile coord (the canonical register-vector layout: data on the last axis).
template<typename G, typename Accum>
__device__ inline void apply_inv_rms(const G& g, Accum& C, int row,int col,int wr,int wc){
    using Tile = std::remove_all_extents_t<Accum>;       // rt_fl<64,32,col_l,rt_16x16_s>
    using CV   = typename Tile::col_vec;                 // length = nrows (64)
    CV cv0, cv1;
    load(cv0, g.r, {(row*2)*WARPS_M+wr});         // r for rows of C[0][*]
    load(cv1, g.r, {(row*2)*WARPS_M+WARPS_M+wr}); // r for rows of C[1][*]
    mul_row(C[0][0], C[0][0], cv0); mul_row(C[0][1], C[0][1], cv0);
    mul_row(C[1][0], C[1][0], cv1); mul_row(C[1][1], C[1][1], cv1);
}

// per-feature gamma : gamma is gl[1,1,1,N]; row_vec (one value per col), applied with mul_col
template<typename G, typename Accum>
__device__ inline void apply_gamma(const G& g, Accum& C, int row,int col,int wr,int wc){
    using Tile = std::remove_all_extents_t<Accum>;
    using RV   = typename Tile::row_vec;                 // length = ncols (32)
    RV g0, g1;
    load(g0, g.gamma, {0,0,0, col*2*WARPS_N+wc});        // cols of C[*][0]
    load(g1, g.gamma, {0,0,0, col*2*WARPS_N+WARPS_N+wc});// cols of C[*][1]
    mul_col(C[0][0], C[0][0], g0); mul_col(C[1][0], C[1][0], g0);
    mul_col(C[0][1], C[0][1], g1); mul_col(C[1][1], C[1][1], g1);
}

// scalar alpha : alpha is a 1-elem gl ([C9]); broadcast-multiply every subtile. Coordinate-free
// (a pure scalar, no store_C-mirrored load), so it takes no row/col/wr/wc.
template<typename G, typename Accum>
__device__ inline void apply_scale(const G& g, Accum& C){
    float a = g.alpha[{0,0,0,0}];
    mul(C[0][0], C[0][0], a); mul(C[0][1], C[0][1], a);   // tile x scalar (maps.cuh:571)
    mul(C[1][0], C[1][0], a); mul(C[1][1], C[1][1], a);
}
