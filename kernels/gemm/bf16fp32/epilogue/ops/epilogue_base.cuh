#pragma once
#include "epilogue_args.cuh"
using namespace kittens;
// the four default full-tile stores; reused by every dim-preserving epilogue
template<typename G, typename Accum>
__device__ inline void store_C(const G& g, const Accum& C, int row,int col,int wr,int wc){
    store(g.c, C[0][0], {0,0,(row*2)*WARPS_M+wr,         col*2*WARPS_N+wc});
    store(g.c, C[0][1], {0,0,(row*2)*WARPS_M+wr,         col*2*WARPS_N+WARPS_N+wc});
    store(g.c, C[1][0], {0,0,(row*2)*WARPS_M+WARPS_M+wr, col*2*WARPS_N+wc});
    store(g.c, C[1][1], {0,0,(row*2)*WARPS_M+WARPS_M+wr, col*2*WARPS_N+WARPS_N+wc});
}
// Identity epilogue (store only). Lives here, NOT in bindings/gemm_noop.cpp like the other
// epilogue structs, because it is micro_tk's DEFAULT Epilogue
// (gemm_base.cuh: template<typename Epilogue = NoOpEpilogue>), which every binding includes --
// moving it would leave that default referencing an undefined type and break every binding.
struct NoOpEpilogue {
    template<typename G, typename Accum>
    static __device__ inline void apply(const G& g, Accum& C, int row,int col,int wr,int wc){
        store_C(g, C, row, col, wr, wc);
    }
};
