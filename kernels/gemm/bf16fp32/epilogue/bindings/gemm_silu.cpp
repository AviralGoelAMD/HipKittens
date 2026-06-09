#include "gemm_base.cuh"
#include "epilogue_base.cuh"          // store_C
#include "epilogue_activations.cuh"   // silu_op
#include "pyutils/pyutils.cuh"

// Stage 3.1: SiLU activation epilogue (dim-preserving).  out = silu(A@B).
struct SiluEpilogue {
    template<typename G, typename Accum>
    static __device__ inline void apply(const G& g, Accum& C, int row,int col,int wr,int wc){
        silu_op(C[0][0]); silu_op(C[0][1]);   // x <- silu(x), register-only
        silu_op(C[1][0]); silu_op(C[1][1]);
        store_C(g, C, row, col, wr, wc);      // epilogue owns the store ([C7])
    }
};

void dispatch_micro(micro_globals g) {
    unsigned long mem = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk<SiluEpilogue>, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    micro_tk<SiluEpilogue><<<g.grid(), g.block(), mem, g.stream>>>(g, g.M, g.N, g.K);
}
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk silu activation epilogue (Stage 3.1)";
    py::bind_function<dispatch_micro>(m, "dispatch_micro",
        &micro_globals::a, &micro_globals::b, &micro_globals::c);
}
