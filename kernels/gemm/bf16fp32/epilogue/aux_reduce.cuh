#pragma once
#include "epilogue_args.cuh"   // REG_BLOCK_N = per-warp-column group width (cols per (row,group) partial)
#include <cfloat>              // FLT_MAX (empty-lane sentinel in cross_entropy_reduce)
using namespace kittens;

// Auxiliary RMS reduce: turn the per-(column-group, row) partials into the per-row inverse RMS.
// One thread per row sums the N/REG_BLOCK_N group-partials into the full Sigma(x^2), then
// r[row] = rsqrt(Sigma / N + RMS_EPS). Standalone kernel (not a gemm_kernel epilogue) -> its own
// tiny globals struct.
//
// partials: [1, 1, N/REG_BLOCK_N, M] (groups x rows; row = last axis).   r: [1, 1, 1, M] (1/rms).
struct aux_globals {
    gl<float,-1,-1,-1,-1> partials;   // input  [1,1,N/REG_BLOCK_N,M]
    gl<bf16,-1,-1,-1,-1>  r;          // output [1,1,1,M]
    hipStream_t stream;
};

__global__ void rms_reduce(const gl<float,-1,-1,-1,-1> partials, gl<bf16,-1,-1,-1,-1> r) {
    constexpr int WF = kittens::WARP_THREADS;                // wavefront width (per arch: 64 CDNA4)
    const int lane = threadIdx.x & (WF - 1);
    const int row  = (blockIdx.x * blockDim.x + threadIdx.x) / WF;
    const int M = r.cols();
    if (row >= M) return;                                    // whole wavefront shares one row -> uniform
    const int groups = partials.rows();                      // = N / REG_BLOCK_N
    const int N = groups * REG_BLOCK_N;                       // full feature dim (REG_BLOCK_N cols/group)
    // 64 lanes split this row's group-partials, then a shuffle-sum (was 1 thread/row, serial: low
    // occupancy / latency-bound, same pathology as the old cross_entropy_reduce).
    float s = 0.f;
    for (int g = lane; g < groups; g += WF) s += partials[{0, 0, g, row}];
    for (int off = WF / 2; off > 0; off >>= 1) s += __shfl_down(s, off);
    if (lane == 0) r.raw_ptr[row] = (bf16)(rsqrtf(s / (float)N + RMS_EPS));
}

// Auxiliary cross-entropy reduce (warp-per-row): combine the N/REG_BLOCK_N per-(group,row) softmax
// partials into logsumexp AND compute the target logit directly as the O(K) dot
// <h[row,:], Wt[label[row],:]> (== logits[row,label[row]]) -- replacing the old O(vocab) mask-gather
// the GEMM epilogue used to do over every column. One wavefront (64 lanes) per row:
//   logsumexp = combine_g (max_buf[g,row], sumexp_buf[g,row])      (associative online softmax)
//   target    = sum_k h[row,k] * Wt[label[row],k]                  (* r[row] for the K8 RMS path)
//   loss[row] = logsumexp - target.
// Partials are [1,1,N/REG_BLOCK_N,M]; a=h [M,K]; b=Wt [N,K]; labels/r/loss are [1,1,1,M].
struct ce_aux_globals {                 // K3 (cross-entropy)
    gl<float,-1,-1,-1,-1> max_buf;
    gl<float,-1,-1,-1,-1> sumexp_buf;
    _gl_A a;                            // h  [M,K]
    _gl_B b;                            // Wt [N,K] (W_vocab transposed)
    gl<float,-1,-1,-1,-1> labels;       // [1,1,1,M]
    gl<float,-1,-1,-1,-1> loss;         // [1,1,1,M]
    hipStream_t stream;
};
struct ce_aux_rms_globals {             // K8 (RMS -> cross-entropy): + per-row inv-rms r
    gl<float,-1,-1,-1,-1> max_buf;
    gl<float,-1,-1,-1,-1> sumexp_buf;
    _gl_A a;
    _gl_B b;
    gl<float,-1,-1,-1,-1> labels;
    gl<bf16,-1,-1,-1,-1>  r;            // [1,1,1,M] per-row inv-rms (target *= r[row])
    gl<float,-1,-1,-1,-1> loss;
    hipStream_t stream;
};

template<bool RMS>
__global__ void cross_entropy_reduce(const gl<float,-1,-1,-1,-1> max_buf,
                                     const gl<float,-1,-1,-1,-1> sumexp_buf,
                                     const _gl_A a, const _gl_B b,
                                     const gl<float,-1,-1,-1,-1> labels,
                                     const bf16* __restrict__ r_ptr,
                                     gl<float,-1,-1,-1,-1> loss) {
    constexpr int WF = kittens::WARP_THREADS;                    // wavefront width (per arch: 64 CDNA4)
    const int lane = threadIdx.x & (WF - 1);
    const int row  = (blockIdx.x * blockDim.x + threadIdx.x) / WF;
    const int M = loss.cols();
    if (row >= M) return;                                        // whole wavefront shares one row -> uniform
    const int groups = max_buf.rows();                           // = N / REG_BLOCK_N
    // (1) online-softmax combine over this row's groups. -FLT_MAX (not -inf) sentinel: empty lanes
    // (groups<64) combine as exp(m-nm)=exp(0)=1 instead of exp(-inf - -inf)=NaN poisoning the reduce.
    float m = -FLT_MAX, s = 0.f;
    for (int g = lane; g < groups; g += WF) {
        const float mg = max_buf[{0, 0, g, row}];
        const float nm = fmaxf(m, mg);
        s = s * __expf(m - nm) + sumexp_buf[{0, 0, g, row}] * __expf(mg - nm);
        m = nm;
    }
    // (2) target logit = <h[row,:], Wt[label,:]> (O(K), not O(vocab)); lanes split K, both rows contiguous.
    const int   K     = a.cols();
    const int   label = (int)labels[{0, 0, 0, row}];
    const bf16* hp = a.raw_ptr + (size_t)row   * K;
    const bf16* wp = b.raw_ptr + (size_t)label * K;
    float t = 0.f;
    for (int k = lane; k < K; k += WF) t += __bfloat162float(hp[k]) * __bfloat162float(wp[k]);
    // (3) reduce across the 64 lanes: associative online-softmax combine + plain sum for the target.
    for (int off = WF / 2; off > 0; off >>= 1) {
        const float om = __shfl_down(m, off);
        const float os = __shfl_down(s, off);
        const float ot = __shfl_down(t, off);
        const float nm = fmaxf(m, om);
        s = s * __expf(m - nm) + os * __expf(om - nm);
        m = nm;
        t += ot;
    }
    if (lane == 0) {
        if constexpr (RMS) t *= __bfloat162float(r_ptr[row]);    // K8: target = r[row] * <h, Wt[label]>
        loss.raw_ptr[row] = (m + logf(s)) - t;
    }
}
