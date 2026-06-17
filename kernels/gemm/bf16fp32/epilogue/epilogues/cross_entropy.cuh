#pragma once
#include <type_traits>   // std::remove_all_extents_t
#include "base.cuh"      // block_coords, subtile_coords
#include "reductions.cuh"
#include "vec.cuh"        // apply_inv_rms (per-row 1/rms prepend, mirrors rmsnorm_swiglu)

// K3 forward fused cross-entropy epilogue. The GEMM produces logits = h @ W_vocab in the fp32
// accumulator; this epilogue emits ONLY per-(row, REG_BLOCK_N-col group) reductions -- the full
// [M, vocab] logits are NEVER stored to HBM (there is no `c` operand). Two partials per
// (group,row), both shaped [1,1,N/REG_BLOCK_N,M] (group on axis 2, row M on the last axis, matching
// partial_row_sum_sq):
//   max_buf[g,r]    = max over this warp's 64 cols of logits[r,c]          (online-softmax base)
//   sumexp_buf[g,r] = sum over this warp's 64 cols of exp(logits[r,c]-max) (online-softmax sum)
// The aux kernel (cross_entropy_reduce) combines the N/REG_BLOCK_N groups per row with a max
// correction -> loss[r] = logsumexp(logits[r,:]) - logits[r, label[r]].
//
// The target logit logits[r, label[r]] is NOT gathered here. Doing so was an O(vocab) per-column
// mask (mask[r,c] = [c == label[r]], then row_sum(logits * mask)) to extract one element per row.
// The aux kernel instead computes it directly as the O(K) dot <h[row,:], Wt[label[r],:]> --
// O(K), not O(vocab), with no in-epilogue per-column label compare.
struct CrossEntropyGlobals {
    _gl_A a;                              // [M,K] bf16  (h)
    _gl_B b;                              // [N,K] bf16  (W_vocab transposed -> [vocab, d_model])
    gl<float,-1,-1,-1,-1> max_buf;       // [1,1,N/REG_BLOCK_N,M]  per-(group,row) softmax max
    gl<float,-1,-1,-1,-1> sumexp_buf;    // [1,1,N/REG_BLOCK_N,M]  per-(group,row) sum exp(logit-max)
    hipStream_t stream;
};

struct PartialLseEpilogue {
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row,int col,int wr,int wc){
        using Tile = std::remove_all_extents_t<Accum>;   // rt_fl<...,col_l>
        using CV   = typename Tile::col_vec;             // one value per row
        subtile_coords co = block_coords(row,col,wr,wc);
        const int grp = col * WARPS_N + wc;              // column-group index (matches partial_row_sum_sq)

        // (a) per-row max over this warp's 64 cols.
        CV m0, m1;
        row_max(m0, C[0][0]); row_max(m0, C[0][1], m0);
        row_max(m1, C[1][0]); row_max(m1, C[1][1], m1);

        // (b) tmp = exp(C - max); s = row_sum(tmp) over this warp's 64 cols.
        Tile t; CV s0, s1;
        sub_row(t, C[0][0], m0); exp(t, t); row_sum(s0, t);
        sub_row(t, C[0][1], m0); exp(t, t); row_sum(s0, t, s0);
        sub_row(t, C[1][0], m1); exp(t, t); row_sum(s1, t);
        sub_row(t, C[1][1], m1); exp(t, t); row_sum(s1, t, s1);

        // (c) store softmax partials only. The target logit is NOT gathered here -- that was an
        // O(vocab) mask over every column to extract one element per row. The aux kernel instead
        // computes it directly as the O(K) dot <h[row], W[:, label[row]]> (== logits[row,label[row]]).
        store(g.max_buf,    m0, {0, 0, grp, co.m[0]});
        store(g.max_buf,    m1, {0, 0, grp, co.m[1]});
        store(g.sumexp_buf, s0, {0, 0, grp, co.m[0]});
        store(g.sumexp_buf, s1, {0, 0, grp, co.m[1]});
    }
};

// K8: RMS -> forward cross-entropy. logits = rmsnorm(h,gamma) @ W_lm == r * (h @ gamma-folded W_lm).
// The per-d_model gamma folds into W_lm's ROWS host-side; `r` is the precomputed per-row inv-rms
// (bf16). apply_inv_rms scales the fp32 accumulator C by r BEFORE the max/sumexp, so the softmax
// partials are r-scaled (correct); the aux kernel re-applies r to the directly-computed target dot.
// Same partials-only contract as CrossEntropyGlobals -- no `c` operand -- plus `r` ([1,1,1,M]).
struct CrossEntropyRmsGlobals {
    _gl_A a;                              // [M,K] bf16  (h)
    _gl_B b;                              // [N,K] bf16  (gamma-folded W_lm transposed -> [vocab, d_model])
    gl<float,-1,-1,-1,-1> max_buf;       // [1,1,N/REG_BLOCK_N,M]  per-(group,row) softmax max
    gl<float,-1,-1,-1,-1> sumexp_buf;    // [1,1,N/REG_BLOCK_N,M]  per-(group,row) sum exp(logit-max)
    gl<bf16,-1,-1,-1,-1>  r;             // [1,1,1,M]  per-row inv-rms (M on the last axis)
    hipStream_t stream;
};

// apply_inv_rms (r * (A@B), per-row) THEN the EXACT PartialLseEpilogue body (max/exp/sum partials),
// reused verbatim -- the r-scaled C feeds the softmax.
struct PartialLseRmsEpilogue {
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row,int col,int wr,int wc){
        apply_inv_rms(g, C, row,col,wr,wc);          // scale by per-row r before the max/sumexp partials
        PartialLseEpilogue::apply(g, C, row,col,wr,wc);
    }
};
