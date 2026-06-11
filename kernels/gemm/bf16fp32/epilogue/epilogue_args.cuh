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

constexpr int K_ALIGN = 2 * K_STEP;   // base GEMM requires K to be a multiple of 128 (two K-steps)
static_assert(BLOCK_SIZE % WARPS_M == 0, "BLOCK_SIZE must be divisible by WARPS_M");
static_assert(BLOCK_SIZE % WARPS_N == 0, "BLOCK_SIZE must be divisible by WARPS_N");
static_assert(REG_BLOCK_M * WARPS_M == BLOCK_SIZE, "REG_BLOCK_M * WARPS_M must equal BLOCK_SIZE");
static_assert(REG_BLOCK_N * WARPS_N == BLOCK_SIZE, "REG_BLOCK_N * WARPS_N must equal BLOCK_SIZE");
static_assert(REG_BLOCK_M % 2 == 0, "REG_BLOCK_M must be even (two row sub-tiles)");
static_assert(REG_BLOCK_N % 2 == 0, "REG_BLOCK_N must be even (two col sub-tiles)");

#define NUM_WARPS (WARPS_M * WARPS_N)
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

// Default launch args for an epilogue with no extra inputs (noop, silu): the GEMM operands +
// stream. An epilogue WITH inputs declares its own flat globals struct that starts with
// {a,b,c} (so the positional pybind binding lists them first), then its own gl fields, then a
// trailing `stream`. M/N/K and the launch geometry are derived in launch(), not stored.
struct gemm_args_base {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    hipStream_t stream;
};
