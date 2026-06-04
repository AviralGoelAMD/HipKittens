#include "gemm_base.cuh"
#include "epilogue_vec_ops.cuh"
#include "pyutils/pyutils.cuh"

// K5 (Stage 1): RMSNorm scaling with a PRECOMPUTED per-row inv_rms `r` and per-feature `gamma`.
//   out = (A@B) * r[:,None] * gamma[None,:]
// Validates the col_l vector-broadcast path both axes ([C2]); the real RMS reduction is Stage 2.
struct K5_RMSScale_Epilogue {
    template<typename G, typename Accum>
    static __device__ inline void apply(const G& g, Accum& C, int row,int col,int wr,int wc){
        apply_inv_rms(g, C, row,col,wr,wc);   // per-row 1/rms  (col_vec, mul_row)
        apply_gamma  (g, C, row,col,wr,wc);   // per-feature gamma (row_vec, mul_col)
        store_C(g, C, row,col,wr,wc);         // epilogue owns the store ([C7])
    }
};

void dispatch_micro(micro_globals g) {
    unsigned long mem = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk<K5_RMSScale_Epilogue>, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    micro_tk<K5_RMSScale_Epilogue><<<g.grid(), g.block(), mem, g.stream>>>(g, g.M, g.N, g.K);
}
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk_kernel K5 RMSNorm-scale epilogue";
    // bind_function is POSITIONAL (pyutils.cuh:68): args fill micro_globals members in
    // declaration order. alpha sits at slot 3, so K5 must bind it too (a throwaway slot) to
    // reach r/gamma at slots 4/5 -> caller passes a dummy alpha that this epilogue ignores.
    py::bind_function<dispatch_micro>(m, "dispatch_micro",
        &micro_globals::a, &micro_globals::b, &micro_globals::c,
        &micro_globals::alpha, &micro_globals::r, &micro_globals::gamma);
}
