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
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int M = r.cols();
    if (row >= M) return;
    int groups = partials.rows();        // = N / REG_BLOCK_N
    int N = groups * REG_BLOCK_N;        // full feature dim (REG_BLOCK_N cols folded into each group)
    float s = 0.f;
    for (int g = 0; g < groups; ++g) s += partials[{0, 0, g, row}];   // stride-aware: no contiguity assumption
    r.raw_ptr[row] = (bf16)(rsqrtf(s / (float)N + RMS_EPS));
}
