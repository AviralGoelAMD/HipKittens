#pragma once
#include <type_traits>
#include "epilogue_args.cuh"
using namespace kittens;

// Per-row partial sum-of-squares for RMSNorm, split across warp-columns ([C1]).
//
// A row's full Sigma(x^2) over all N features is spread across WARPS_N=4 warp-columns; row_sum
// reduces ONLY the calling warp's own 64 columns (intra-warp shuffles, no cross-warp). So each
// warp emits a PARTIAL per (row, 64-col group). The aux kernel (Task 2.4) sums the N/64 groups
// per row -> Sigma(x^2) -> 1/rms.
//
// AXIS ([C12d]; mirrors store_C's row coords AND the L_vec precedent,
// kernels/attn/gqa/kernel.cpp:573 `store(g.L_vec, norm_vec, {b,h,0,tile})`): a per-row vector
// (col_vec) anchors on the gl's LAST axis. Therefore `partials` is shaped [1,1, N/64, M] -- the
// 64-col-group index on axis 2, the row (M) on the LAST axis -- and the col_vec stores along M.
// (NOT [1,1,M,N/64] with the row-tile in the 3rd coord: that is the K5-class axis bug.)
// This col_vec->global store axis is gfx950-verified (test_partialrms, test_k4).
template<typename G, typename Accum>
__device__ inline void partial_row_sum_sq(const G& g, const Accum& C, int row,int col,int wr,int wc){
    using Tile = std::remove_all_extents_t<Accum>;
    using CV   = typename Tile::col_vec;        // one value per row
    Tile sq; CV p0, p1;
    mul(sq, C[0][0], C[0][0]); row_sum(p0, sq);        // this warp's first 32 cols (reset)
    mul(sq, C[0][1], C[0][1]); row_sum(p0, sq, p0);    // + next 32  -> p0 = sum over this warp's 64 cols
    mul(sq, C[1][0], C[1][0]); row_sum(p1, sq);
    mul(sq, C[1][1], C[1][1]); row_sum(p1, sq, p1);
    const int grp = col * WARPS_N + wc;                // 64-col group index, NOT col ([C1])
    store(g.partials, p0, {0, 0, grp, (row*2)*WARPS_M + wr});
    store(g.partials, p1, {0, 0, grp, (row*2)*WARPS_M + WARPS_M + wr});
}
