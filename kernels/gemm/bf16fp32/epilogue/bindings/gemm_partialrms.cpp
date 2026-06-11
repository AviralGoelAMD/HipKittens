#include "gemm_base.cuh"
#include "epilogue_reductions.cuh"   // partial_row_sum_sq
#include "pyutils/pyutils.cuh"

// Per-(row, REG_BLOCK_N-col group) partial sum-of-squares for RMSNorm. Emits ONLY `partials`
// (no C store); the aux kernel sums the groups per row -> Sigma(x^2) -> 1/rms.
struct PartialRMSGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<float,-1,-1,-1,-1> partials;   // [1,1,N/REG_BLOCK_N,M]
    hipStream_t stream;
};
struct PartialRMSEpilogue {
    template<typename G, typename Accum>
    static __device__ inline void apply(const G& g, Accum& C, int row,int col,int wr,int wc){
        partial_row_sum_sq(g, C, row,col,wr,wc);   // store partials only; no store_C
    }
};

void dispatch(PartialRMSGlobals g) { launch<PartialRMSEpilogue, PartialRMSGlobals>(g); }
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk_kernel partial RMS sum-of-squares epilogue";
    py::bind_function<dispatch>(m, "dispatch",
        &PartialRMSGlobals::a, &PartialRMSGlobals::b, &PartialRMSGlobals::c, &PartialRMSGlobals::partials);
}
