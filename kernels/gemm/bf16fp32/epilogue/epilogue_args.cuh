#pragma once
#include "kittens.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE       = 256;  
constexpr int HALF_BLOCK_SIZE  = BLOCK_SIZE / 2;
constexpr int K_STEP           = 64;
constexpr int WARPS_M          = 2;
constexpr int WARPS_N          = 4;
constexpr int REG_BLOCK_M      = BLOCK_SIZE / WARPS_M;
constexpr int REG_BLOCK_N      = BLOCK_SIZE / WARPS_N;
constexpr int HALF_REG_BLOCK_M = REG_BLOCK_M / 2;
constexpr int HALF_REG_BLOCK_N = REG_BLOCK_N / 2;
constexpr int DOT_SLICE        = 32;

#define NUM_WARPS (WARPS_M * WARPS_N)
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    gl<float,1,1,1,1> alpha{nullptr,nullptr,nullptr,nullptr,nullptr};  // null default so bindings that omit alpha still aggregate-init
    gl<bf16,-1,-1,-1,-1> r{nullptr,1,1,1,1};      // K5: per-row inv_rms [1,1,1,M] (last axis); null default so bindings that omit it still aggregate-init
    gl<bf16,-1,-1,-1,-1> gamma{nullptr,1,1,1,1};  // K5: per-feature gamma [1,1,1,N]; null default
    gl<bf16,-1,-1,-1,-1> residual{nullptr,1,1,1,1};  // Stage 2: skip connection [1,1,M,N]; null default
    gl<float,-1,-1,-1,-1> partials{nullptr,1,1,1,1};  // Stage 2: per-(group,row) RMS partials [1,1,N/64,M] (row=M LAST axis, [C12d]); null default
    gl<bf16,-1,-1,-1,-1> save{nullptr,1,1,1,1};  // Stage 2 (K4): saved h1 = A@B+residual [1,1,M,N] for K5; null default
    hipStream_t stream;
    int M = a.rows();
    int N = c.cols();
    int K = a.cols();
    dim3 grid()  { return dim3((N / BLOCK_SIZE) * (M / BLOCK_SIZE)); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; } 
};
