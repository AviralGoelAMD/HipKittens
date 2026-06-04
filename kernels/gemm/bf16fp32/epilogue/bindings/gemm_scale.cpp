#include "gemm_base.cuh"
#include "pyutils/pyutils.cuh"

struct ScaleEpilogue {
    template<typename G, typename Accum>
    static __device__ inline void apply(const G& g, Accum& C, int row,int col,int wr,int wc){
        float a = g.alpha[{0,0,0,0}];                 // 1-elem gl scalar ([C9])
        #pragma unroll
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) mul(C[i][j], C[i][j], a);   // tile x scalar (maps.cuh:571)
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
