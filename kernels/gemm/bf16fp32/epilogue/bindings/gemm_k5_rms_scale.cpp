#include "gemm_base.cuh"
#include "epilogue_vec_ops.cuh"   // apply_inv_rms, apply_gamma
#include "pyutils/pyutils.cuh"

// RMSNorm scaling with a precomputed per-row inv_rms `r` and per-feature `gamma`:
//   out = (A@B) * r[:,None] * gamma[None,:]
struct K5Globals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<bf16,-1,-1,-1,-1> r;       // per-row inv_rms, [1,1,1,M] (M on the last axis)
    gl<bf16,-1,-1,-1,-1> gamma;   // per-feature gamma, [1,1,1,N]
    hipStream_t stream;
};
struct K5_RMSScale_Epilogue {
    template<typename G, typename Accum>
    static __device__ inline void apply(const G& g, Accum& C, int row,int col,int wr,int wc){
        apply_inv_rms(g, C, row,col,wr,wc);   // per-row 1/rms  (col_vec, mul_row)
        apply_gamma  (g, C, row,col,wr,wc);   // per-feature gamma (row_vec, mul_col)
        store_C(g, C, row,col,wr,wc);
    }
};

void dispatch_micro(K5Globals g) { launch_micro<K5_RMSScale_Epilogue, K5Globals>(g); }
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk_kernel K5 RMSNorm-scale epilogue";
    py::bind_function<dispatch_micro>(m, "dispatch_micro",
        &K5Globals::a, &K5Globals::b, &K5Globals::c, &K5Globals::r, &K5Globals::gamma);
}
