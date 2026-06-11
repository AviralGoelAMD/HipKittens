#include "gemm_base.cuh"
#include "epilogue_tile_ops.cuh"    // residual_add, save_tile
#include "epilogue_reductions.cuh"  // partial_row_sum_sq
#include "epilogue_vec_ops.cuh"     // apply_gamma
#include "pyutils/pyutils.cuh"

// Residual + RMS-partials GEMM: h1 = A@B + residual; save h1 (for the second GEMM); emit
// Sigma(h1^2) partials (for the aux 1/rms); apply gamma; store h1*gamma. The aux kernel turns
// partials -> r = 1/rms; the RMSNorm-scale GEMM then reads `save`.
struct ResidualRMSPartialsGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<bf16,-1,-1,-1,-1>  residual;   // [1,1,M,N] skip connection
    gl<bf16,-1,-1,-1,-1>  gamma;      // [1,1,1,N] per-feature gamma
    gl<float,-1,-1,-1,-1> partials;   // [1,1,N/REG_BLOCK_N,M] per-(group,row) Sigma(h1^2)
    gl<bf16,-1,-1,-1,-1>  save;       // [1,1,M,N] saved h1
    hipStream_t stream;
};
struct ResidualRMSPartialsEpilogue {
    template<typename G, typename Accum>
    static __device__ inline void apply(const G& g, Accum& C, int row,int col,int wr,int wc){
        residual_add(g, C, row,col,wr,wc);        // C = A@B + residual = h1
        save_tile(g, C, row,col,wr,wc);           // persist h1 -> g.save
        partial_row_sum_sq(g, C, row,col,wr,wc);  // Sigma(h1^2) per (row, col group) -> g.partials
        apply_gamma(g, C, row,col,wr,wc);         // C = h1 * gamma
        store_C(g, C, row,col,wr,wc);             // -> g.c
    }
};

void dispatch(ResidualRMSPartialsGlobals g) { launch<ResidualRMSPartialsEpilogue, ResidualRMSPartialsGlobals>(g); }
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk residual + partial-RMS + gamma epilogue";
    py::bind_function<dispatch>(m, "dispatch",
        &ResidualRMSPartialsGlobals::a, &ResidualRMSPartialsGlobals::b, &ResidualRMSPartialsGlobals::c,
        &ResidualRMSPartialsGlobals::residual, &ResidualRMSPartialsGlobals::gamma,
        &ResidualRMSPartialsGlobals::partials, &ResidualRMSPartialsGlobals::save);
}
