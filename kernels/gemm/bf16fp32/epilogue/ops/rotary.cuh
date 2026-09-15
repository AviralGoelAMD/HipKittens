#pragma once
#include <type_traits>
#include "base.cuh"    // block_coords, subtile_coords, gl typedefs
using namespace kittens;

// apply_rope_store: interleaved RoPE on the col_l accumulator, register-only, then stored.
//
// The projection weight B and `cos_sin` are pre-permuted (rope_perm) so each interleaved pair
// (2k, 2k+1) lands 128 columns apart -> the same (warp,lane) owns both in its two col-sub-tiles:
// C[r][0] = even member (x), C[r][1] = odd member (y). cos_sin is permuted identically, so loading
// a tile at the even-member coords gives cos_k and at the odd-member coords gives sin_k, aligned
// element-for-element with C[r][0]/C[r][1] (no lane math). The rotation:
//   O[2k]   =  x*cos_k + y*sin_k        (-> C[r][0])
//   O[2k+1] = -x*sin_k + y*cos_k        (-> C[r][1])
//
// The rotation also STORES each row sub-tile as soon as it is rotated, rather than rotating both
// and letting the caller store afterwards. Deferring the stores keeps both halves of C live
// across the whole rotation on top of the four temps below -- 256 VGPRs, which spilled (93 on
// rmsnorm_rope, 12 on rope). Storing early ends the live range and frees 64 VGPRs. y's products
// are taken first so x can be overwritten in place, which also avoids a full-tile copy.
template<typename Globals, typename Accum>
__device__ inline void apply_rope_store(const Globals& g, Accum& C, int row,int col,int wr,int wc){
    using Tile = std::remove_all_extents_t<Accum>;
    subtile_coords co = block_coords(row,col,wr,wc);
    static_assert(SUBTILES_PER_DIM == 2, "hardwired rt_fl[2][2] fan-out");
    Tile cos_t, sin_t, t0, t1;
    #pragma unroll
    for (int i = 0; i < SUBTILES_PER_DIM; ++i) {
        load(cos_t, g.cos_sin, {0,0,co.m[i],co.n[0]});   // cos_k, aligned with C[i][0] (even)
        load(sin_t, g.cos_sin, {0,0,co.m[i],co.n[1]});   // sin_k, aligned with C[i][1] (odd)
        mul(t0, C[i][1], sin_t);                      // y*sin
        mul(t1, C[i][1], cos_t);                      // y*cos  -- y now dead
        mul(C[i][1], C[i][0], sin_t);
        sub(C[i][1], t1, C[i][1]);                    // O_odd  = y*cos - x*sin
        mul(C[i][0], C[i][0], cos_t);
        add(C[i][0], C[i][0], t0);                    // O_even = x*cos + y*sin
        store(g.c, C[i][0], {0,0,co.m[i],co.n[0]});   // store now: releases this sub-tile
        store(g.c, C[i][1], {0,0,co.m[i],co.n[1]});
    }
}
