#pragma once
#include <type_traits>
#include "base.cuh"    // block_coords, subtile_coords
using namespace kittens;

// Vector-broadcast epilogue ops on the col_l accumulator tiles.
//
// per-ROW scale (1/rms): one value per row -> a COLUMN-shaped vector (col_vec) + mul_row.
// per-COL scale (gamma): one value per col -> a ROW-shaped vector    (row_vec) + mul_col.
// (The name is the vector's SHAPE, not what it scales -- a per-row value uses a col_vec.)
// Coordinates come from block_coords so r[i]/gamma[j] line up with the rows/cols each register
// sub-tile holds.
// Load-coord forms differ by vector kind: col_vec loads a single last-axis index ({co.m[0]});
// row_vec loads full coords ({0,0,0,co.n[0]}, N last). Both correct -- not a typo.

// per-row 1/rms: r is gl[1,1,1,M] (M on the last axis); col_vec + mul_row.
template<typename Globals, typename Accum>
__device__ inline void apply_inv_rms(const Globals& g, Accum& C, int row,int col,int wr,int wc){
    using Tile = std::remove_all_extents_t<Accum>;
    using CV   = typename Tile::col_vec;                 // one value per row
    subtile_coords co = block_coords(row,col,wr,wc);
    CV cv0, cv1;
    load(cv0, g.r, {co.m[0]});                           // r for the rows of C[0][*]
    load(cv1, g.r, {co.m[1]});                           // r for the rows of C[1][*]
    mul_row(C[0][0], C[0][0], cv0); mul_row(C[0][1], C[0][1], cv0);
    mul_row(C[1][0], C[1][0], cv1); mul_row(C[1][1], C[1][1], cv1);
}

// per-feature gamma: gamma is gl[1,1,1,N]; row_vec (one value per col) + mul_col.
template<typename Globals, typename Accum>
__device__ inline void apply_gamma(const Globals& g, Accum& C, int row,int col,int wr,int wc){
    using Tile = std::remove_all_extents_t<Accum>;
    using RV   = typename Tile::row_vec;                 // one value per col
    subtile_coords co = block_coords(row,col,wr,wc);
    RV g0, g1;
    load(g0, g.gamma, {0,0,0, co.n[0]});                 // cols of C[*][0]
    load(g1, g.gamma, {0,0,0, co.n[1]});                 // cols of C[*][1]
    mul_col(C[0][0], C[0][0], g0); mul_col(C[1][0], C[1][0], g0);
    mul_col(C[0][1], C[0][1], g1); mul_col(C[1][1], C[1][1], g1);
}

// scalar alpha: a 1-element gl, broadcast-multiplied into every sub-tile. Coordinate-free
// (a pure scalar, no block_coords load), so it takes no row/col/wr/wc.
template<typename Globals, typename Accum>
__device__ inline void apply_scale(const Globals& g, Accum& C){
    float a = g.alpha[{0,0,0,0}];
    mul(C[0][0], C[0][0], a); mul(C[0][1], C[0][1], a);
    mul(C[1][0], C[1][0], a); mul(C[1][1], C[1][1], a);
}
