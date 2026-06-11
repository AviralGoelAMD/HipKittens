#include "gemm_base.cuh"
#include "epilogue_activations.cuh"   // silu_op
#include "pyutils/pyutils.cuh"

// SiLU activation epilogue (dim-preserving):  out = silu(A@B). No extra inputs -> gemm_args_base.
struct SiluEpilogue {
    template<typename G, typename Accum>
    static __device__ inline void apply(const G& g, Accum& C, int row,int col,int wr,int wc){
        silu_op(C[0][0]); silu_op(C[0][1]);   // x <- silu(x), register-only
        silu_op(C[1][0]); silu_op(C[1][1]);
        store_C(g, C, row, col, wr, wc);
    }
};

void dispatch(gemm_args_base g) { launch<SiluEpilogue, gemm_args_base>(g); }
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk silu activation epilogue";
    py::bind_function<dispatch>(m, "dispatch",
        &gemm_args_base::a, &gemm_args_base::b, &gemm_args_base::c);
}
