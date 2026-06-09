#include "gemm_base.cuh"
#include "epilogue_vec_ops.cuh"   // apply_scale
#include "pyutils/pyutils.cuh"

struct ScaleEpilogue {
    template<typename G, typename Accum>
    static __device__ inline void apply(const G& g, Accum& C, int row,int col,int wr,int wc){
        apply_scale(g, C);                            // alpha * C  (tile x 1-elem-gl scalar, [C9])
        store_C(g, C, row, col, wr, wc);              // epilogue owns the store ([C7])
    }
};

void dispatch_micro(micro_globals g) {
    unsigned long mem = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk<ScaleEpilogue>, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    micro_tk<ScaleEpilogue><<<g.grid(), g.block(), mem, g.stream>>>(g, g.M, g.N, g.K);
}
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk_kernel scale epilogue";
    py::bind_function<dispatch_micro>(m, "dispatch_micro",
        &micro_globals::a, &micro_globals::b, &micro_globals::c, &micro_globals::alpha);
}
