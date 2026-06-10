#include "gemm_base.cuh"
#include "epilogue_tile_ops.cuh"    // residual_add, save_tile
#include "epilogue_reductions.cuh"  // partial_row_sum_sq
#include "epilogue_vec_ops.cuh"     // apply_gamma
#include "pyutils/pyutils.cuh"

// K4 money-pattern GEMM: h1 = A@B + residual; save h1 (for K5); emit Sigma(h1^2) partials
// (for the aux 1/rms); apply gamma; store h1*gamma. The aux kernel turns partials -> r=1/rms;
// K5 then reads `save`.
struct K4Globals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<bf16,-1,-1,-1,-1>  residual;   // [1,1,M,N] skip connection
    gl<bf16,-1,-1,-1,-1>  gamma;      // [1,1,1,N] per-feature gamma
    gl<float,-1,-1,-1,-1> partials;   // [1,1,N/REG_BLOCK_N,M] per-(group,row) Sigma(h1^2)
    gl<bf16,-1,-1,-1,-1>  save;       // [1,1,M,N] saved h1 for K5
    hipStream_t stream;
};
struct K4_ResidualRMS {
    template<typename G, typename Accum>
    static __device__ inline void apply(const G& g, Accum& C, int row,int col,int wr,int wc){
        residual_add(g, C, row,col,wr,wc);        // C = A@B + residual = h1
        save_tile(g, C, row,col,wr,wc);           // persist h1 -> g.save (consumed by K5)
        partial_row_sum_sq(g, C, row,col,wr,wc);  // Sigma(h1^2) per (row, col group) -> g.partials
        apply_gamma(g, C, row,col,wr,wc);         // C = h1 * gamma
        store_C(g, C, row,col,wr,wc);             // -> g.c
    }
};

void dispatch_micro(K4Globals g) { launch_micro<K4_ResidualRMS, K4Globals>(g); }
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk_kernel K4: residual + partial-RMS + gamma epilogue";
    py::bind_function<dispatch_micro>(m, "dispatch_micro",
        &K4Globals::a, &K4Globals::b, &K4Globals::c,
        &K4Globals::residual, &K4Globals::gamma, &K4Globals::partials, &K4Globals::save);
}
