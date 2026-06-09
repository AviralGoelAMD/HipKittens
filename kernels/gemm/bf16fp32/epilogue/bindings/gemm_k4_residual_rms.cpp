#include "gemm_base.cuh"
#include "epilogue_base.cuh"        // store_C
#include "epilogue_tile_ops.cuh"    // residual_add, save_tile
#include "epilogue_reductions.cuh"  // partial_row_sum_sq
#include "epilogue_vec_ops.cuh"     // apply_gamma
#include "pyutils/pyutils.cuh"

// Stage 2 Task 2.3 — K4, the money-pattern GEMM:
//   h1 = A@B + residual; save h1 (for K5); emit Sigma(h1^2) partials (for the aux 1/rms);
//   apply gamma; store h1*gamma. The aux kernel turns partials -> r=1/rms; K5 then reads `save`.
struct K4_ResidualRMS {
    template<typename G, typename Accum>
    static __device__ inline void apply(const G& g, Accum& C, int row,int col,int wr,int wc){
        residual_add(g, C, row,col,wr,wc);        // C = A@B + residual = h1
        save_tile(g, C, row,col,wr,wc);           // persist h1 -> g.save (consumed by K5)
        partial_row_sum_sq(g, C, row,col,wr,wc);  // Sigma(h1^2) per (row, 64-col group) -> g.partials
        apply_gamma(g, C, row,col,wr,wc);         // C = h1 * gamma
        store_C(g, C, row,col,wr,wc);             // -> g.c
    }
};

void dispatch_micro(micro_globals g) {
    unsigned long mem = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk<K4_ResidualRMS>, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    micro_tk<K4_ResidualRMS><<<g.grid(), g.block(), mem, g.stream>>>(g, g.M, g.N, g.K);
}
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk_kernel K4: residual + partial-RMS + gamma epilogue";
    // positional aggregate-init ([C12b]); K4 uses gamma/residual/partials/save -> dummy alpha,r to reach them.
    py::bind_function<dispatch_micro>(m, "dispatch_micro",
        &micro_globals::a, &micro_globals::b, &micro_globals::c,
        &micro_globals::alpha, &micro_globals::r, &micro_globals::gamma,
        &micro_globals::residual, &micro_globals::partials, &micro_globals::save);
}
