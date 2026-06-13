#pragma once
#include <type_traits>
#include "base.cuh"    // block_coords, subtile_coords
using namespace kittens;

// Tile (rank-2) epilogue ops on the col_l accumulator. Coordinates come from block_coords (the
// single tile->global mapping), so a loaded/stored tile lines up element-for-element with each
// accumulator sub-tile.

// residual_add: C += residual, where `residual` is a full [M,N] bf16 skip connection. Each
// sub-tile is loaded with the accumulator's own coords, converting bf16 -> fp32 on the way in.
template<typename Globals, typename Accum>
__device__ inline void residual_add(const Globals& g, Accum& C, int row,int col,int wr,int wc){
    using Tile = std::remove_all_extents_t<Accum>;
    subtile_coords co = block_coords(row,col,wr,wc);
    Tile t;
    load(t, g.residual, {0,0,co.m[0],co.n[0]}); add(C[0][0], C[0][0], t);
    load(t, g.residual, {0,0,co.m[0],co.n[1]}); add(C[0][1], C[0][1], t);
    load(t, g.residual, {0,0,co.m[1],co.n[0]}); add(C[1][0], C[1][0], t);
    load(t, g.residual, {0,0,co.m[1],co.n[1]}); add(C[1][1], C[1][1], t);
}

// save_tile: persist the current accumulator to g.save (HBM) for a later kernel. The residual-RMS
// GEMM saves h1 = A@B + residual so the RMSNorm-scale GEMM can normalize it once the aux kernel
// has produced 1/rms (not known until this kernel + aux finish). C is read-only here; the kernel
// keeps transforming C in registers afterward.
template<typename Globals, typename Accum>
__device__ inline void save_tile(const Globals& g, const Accum& C, int row,int col,int wr,int wc){
    subtile_coords co = block_coords(row,col,wr,wc);
    store(g.save, C[0][0], {0,0,co.m[0],co.n[0]});
    store(g.save, C[0][1], {0,0,co.m[0],co.n[1]});
    store(g.save, C[1][0], {0,0,co.m[1],co.n[0]});
    store(g.save, C[1][1], {0,0,co.m[1],co.n[1]});
}
