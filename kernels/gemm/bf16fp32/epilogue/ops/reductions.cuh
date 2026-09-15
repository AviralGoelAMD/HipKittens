#pragma once
#include <type_traits>
#include "base.cuh"    // block_coords
using namespace kittens;

// Per-row partial sum-of-squares for RMSNorm, split across warp-columns.
//
// A row's full Sigma(x^2) over all N features is spread across WARPS_N warp-columns; row_sum
// reduces only the calling warp's own REG_BLOCK_N columns (intra-warp shuffles, no cross-warp),
// so each warp emits a PARTIAL per (row, column-group). The aux kernel sums the N/REG_BLOCK_N
// groups per row -> full Sigma(x^2) -> 1/rms.
//
// Axis: a per-row vector (col_vec) anchors on the gl's LAST axis, so `partials` is shaped
// [1, 1, N/REG_BLOCK_N, M] -- the column-group index on axis 2, the row (M) on the last axis --
// and the col_vec stores along M. (Shaping it [1,1,M,N/REG_BLOCK_N] would reduce the wrong axis.)
template<typename Globals, typename Accum>
__device__ inline void partial_row_sum_sq(const Globals& g, const Accum& C, int row,int col,int wr,int wc){
    using Tile = std::remove_all_extents_t<Accum>;
    using CV   = typename Tile::col_vec;        // one value per row
    Tile sq; CV p0, p1;
    mul(sq, C[0][0], C[0][0]); row_sum(p0, sq);        // this warp's first half of its columns (reset)
    mul(sq, C[0][1], C[0][1]); row_sum(p0, sq, p0);    // + second half -> p0 = sum over this warp's columns
    mul(sq, C[1][0], C[1][0]); row_sum(p1, sq);
    mul(sq, C[1][1], C[1][1]); row_sum(p1, sq, p1);
    const int grp = col * WARPS_N + wc;                // column-group index (not col)
    subtile_coords co = block_coords(row,col,wr,wc);
    store(g.partials, p0, {0, 0, grp, co.m[0]});
    store(g.partials, p1, {0, 0, grp, co.m[1]});
}
