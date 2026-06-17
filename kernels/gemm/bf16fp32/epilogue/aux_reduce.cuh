#pragma once
#include "epilogue_args.cuh"   // REG_BLOCK_N = per-warp-column group width (cols per (row,group) partial)
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
