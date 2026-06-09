#include "gemm_base.cuh"
#include "epilogue_tile_ops.cuh"
#include "pyutils/pyutils.cuh"

// Stage 2 Task 2.1 (first brick of the money pattern):
//   out = (A@B) + residual      residual is the [M,N] bf16 skip connection.
// Standalone residual add, fused onto the GEMM epilogue (D never round-trips HBM).
struct ResAddEpilogue {
    template<typename G, typename Accum>
    static __device__ inline void apply(const G& g, Accum& C, int row,int col,int wr,int wc){
        residual_add(g, C, row,col,wr,wc);   // load residual tile, add into the accumulator
        store_C(g, C, row,col,wr,wc);        // epilogue owns the store ([C7])
    }
};

void dispatch_micro(micro_globals g) {
    unsigned long mem = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk<ResAddEpilogue>, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    micro_tk<ResAddEpilogue><<<g.grid(), g.block(), mem, g.stream>>>(g, g.M, g.N, g.K);
}
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk_kernel residual-add epilogue";
    // Aggregate-init is POSITIONAL by declaration order ([C12b], pyutils.cuh:56/68): `residual`
    // sits after alpha/r/gamma, so bind those slots too and let the caller pass throwaway
    // alpha/r/gamma that ResAddEpilogue ignores. (Per-kernel globals structs would remove the
    // dummies -- the flagged tech-debt as Stage-2+ fields pile up.)
    py::bind_function<dispatch_micro>(m, "dispatch_micro",
        &micro_globals::a, &micro_globals::b, &micro_globals::c,
        &micro_globals::alpha, &micro_globals::r, &micro_globals::gamma, &micro_globals::residual);
}
