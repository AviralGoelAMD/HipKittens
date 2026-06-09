#pragma once
#include "epilogue_args.cuh"   // REG_BLOCK_N = per-warp-column group width (cols per (row,group) partial)
using namespace kittens;

// Auxiliary RMS reduce (Stage 2 Task 2.4): turn K4's per-(64-col group, row) partials into the
// per-row inverse RMS. One thread per row sums the N/64 group-partials ([C1]) into the full
// Sigma(x^2), then r[row] = rsqrt(Sigma / N + eps). eps = 1e-5 (matches layer_norm.py).
// Standalone kernel (not a micro_tk epilogue) -> its own tiny globals struct.
//
// partials: [1,1, N/64, M] (groups x rows; row = last axis, [C12d]).   r: [1,1, 1, M] (1/rms).
struct aux_globals {
    gl<float,-1,-1,-1,-1> partials;   // input  [1,1,N/64,M]
    gl<bf16,-1,-1,-1,-1>  r;          // output [1,1,1,M]
    hipStream_t stream;
};

__global__ void rms_reduce(const gl<float,-1,-1,-1,-1> partials, gl<bf16,-1,-1,-1,-1> r) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int M = r.cols();
    if (row >= M) return;
    int groups = partials.rows();        // = N / REG_BLOCK_N
    int N = groups * REG_BLOCK_N;        // full feature dim (REG_BLOCK_N cols folded into each group)
    int stride = partials.cols();        // = M  (partials laid out [groups, M] row-major)
    float s = 0.f;
    for (int g = 0; g < groups; ++g) s += partials.raw_ptr[g * stride + row];
    constexpr float eps = 1e-5f;
    r.raw_ptr[row] = (bf16)(rsqrtf(s / (float)N + eps));
}
