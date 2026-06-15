#pragma once
#include <type_traits>
#include "base.cuh"    // block_coords, subtile_coords, SUBTILES_PER_DIM, gl typedefs
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
    #pragma unroll
    for (int r = 0; r < SUBTILES_PER_DIM; ++r) {
        Tile cos_t, sin_t, oe, tmp;
        load(cos_t, g.cos_sin, {0,0,co.m[r],co.n[0]});   // cos_k, aligned with C[r][0] (even)
        load(sin_t, g.cos_sin, {0,0,co.m[r],co.n[1]});   // sin_k, aligned with C[r][1] (odd)
        // O_even = x*cos + y*sin  (compute into oe; C[r][*] still hold the originals x,y)
        mul(oe,  C[r][0], cos_t);
        mul(tmp, C[r][1], sin_t);
        add(oe,  oe, tmp);
        // O_odd = y*cos - x*sin  (consume y, then x; C[r][0] still holds x for the last term)
        mul(tmp,    C[r][1], cos_t);     // tmp = y*cos
        mul(C[r][1], C[r][0], sin_t);    // C[r][1] = x*sin
        sub(C[r][1], tmp, C[r][1]);      // C[r][1] = y*cos - x*sin = O_odd
        mul(C[r][0], oe, 1.0f);          // C[r][0] = O_even
    }
}
