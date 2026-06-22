#pragma once
#include <type_traits>
#include "base.cuh"    // block_coords, subtile_coords, gl typedefs
using namespace kittens;

// apply_rope: interleaved RoPE on the col_l accumulator, register-only.
//
// The projection weight B and `cos_sin` are pre-permuted (rope_perm) so each interleaved pair
// (2k, 2k+1) lands 128 columns apart -> the same (warp,lane) owns both in its two col-sub-tiles:
// C[r][0] = even member (x), C[r][1] = odd member (y). cos_sin is permuted identically, so loading
// a tile at the even-member coords gives cos_k and at the odd-member coords gives sin_k, aligned
// element-for-element with C[r][0]/C[r][1] (no lane math). The rotation:
//   O[2k]   =  x*cos_k + y*sin_k        (-> C[r][0])
//   O[2k+1] = -x*sin_k + y*cos_k        (-> C[r][1])
template<typename Globals, typename Accum>
__device__ inline void apply_rope(const Globals& g, Accum& C, int row,int col,int wr,int wc){
    using Tile = std::remove_all_extents_t<Accum>;
    subtile_coords co = block_coords(row,col,wr,wc);
    // Fixed 2x2 fan-out: the two row sub-tiles, written explicitly (like store_C / residual_add).
    // NOT generic over SUBTILES_PER_DIM -- the accumulator is a hardwired rt_fl[2][2].
    Tile cos_t, sin_t, oe, tmp;
    // --- row sub-tile 0 ---
    load(cos_t, g.cos_sin, {0,0,co.m[0],co.n[0]});   // cos_k, aligned with C[0][0] (even)
    load(sin_t, g.cos_sin, {0,0,co.m[0],co.n[1]});   // sin_k, aligned with C[0][1] (odd)
    mul(oe, C[0][0], cos_t); mul(tmp, C[0][1], sin_t); add(oe, oe, tmp);        // O_even = x*cos + y*sin
    mul(tmp, C[0][1], cos_t); mul(C[0][1], C[0][0], sin_t); sub(C[0][1], tmp, C[0][1]);  // O_odd = y*cos - x*sin
    copy(C[0][0], oe);
    // --- row sub-tile 1 ---
    load(cos_t, g.cos_sin, {0,0,co.m[1],co.n[0]});
    load(sin_t, g.cos_sin, {0,0,co.m[1],co.n[1]});
    mul(oe, C[1][0], cos_t); mul(tmp, C[1][1], sin_t); add(oe, oe, tmp);
    mul(tmp, C[1][1], cos_t); mul(C[1][1], C[1][0], sin_t); sub(C[1][1], tmp, C[1][1]);
    copy(C[1][0], oe);
}
