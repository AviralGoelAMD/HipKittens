#include "gemm_base.cuh"
#include "epilogue_reductions.cuh"
#include "pyutils/pyutils.cuh"

// Stage 2 Task 2.2: per-(row, 64-col group) partial sum-of-squares for RMSNorm ([C1]).
// Emits ONLY the `partials` buffer (no C store) -- the aux kernel (Task 2.4) sums the N/64
// groups per row -> Sigma(x^2) -> 1/rms. Test kernel for the cross-warp-column partial path.
struct PartialRMSEpilogue {
    template<typename G, typename Accum>
    static __device__ inline void apply(const G& g, Accum& C, int row,int col,int wr,int wc){
        partial_row_sum_sq(g, C, row,col,wr,wc);   // store partials only; no store_C ([C7])
    }
};

void dispatch_micro(micro_globals g) {
    unsigned long mem = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk<PartialRMSEpilogue>, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    micro_tk<PartialRMSEpilogue><<<g.grid(), g.block(), mem, g.stream>>>(g, g.M, g.N, g.K);
}
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk_kernel partial RMS sum-of-squares epilogue";
    // POSITIONAL aggregate-init ([C12b]): `partials` is the 5th extra field, so bind every slot up
    // to it; the caller passes throwaway alpha/r/gamma/residual (ignored) to reach partials.
    // FUTURE: a python arg-packer fills these dummies by name (plan backlog) -- decided to keep the
    // dummy chain for now and add the helper later.
    py::bind_function<dispatch_micro>(m, "dispatch_micro",
        &micro_globals::a, &micro_globals::b, &micro_globals::c,
        &micro_globals::alpha, &micro_globals::r, &micro_globals::gamma,
        &micro_globals::residual, &micro_globals::partials);
}
