#include "gemm_base.cuh"
#include "epilogue_vec_ops.cuh"   // apply_scale
#include "pyutils/pyutils.cuh"

// Per-output scalar scale:  out = alpha * (A@B).
struct ScaleGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<float,1,1,1,1> alpha{nullptr,nullptr,nullptr,nullptr,nullptr};
    hipStream_t stream;
};
struct ScaleEpilogue {
    template<typename G, typename Accum>
    static __device__ inline void apply(const G& g, Accum& C, int row,int col,int wr,int wc){
        apply_scale(g, C);                            // alpha * C  (tile x 1-elem-gl scalar)
        store_C(g, C, row, col, wr, wc);              // epilogue owns the store
    }
};

void dispatch_micro(ScaleGlobals g) { launch_micro<ScaleEpilogue, ScaleGlobals>(g); }
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk_kernel scale epilogue";
    py::bind_function<dispatch_micro>(m, "dispatch_micro",
        &ScaleGlobals::a, &ScaleGlobals::b, &ScaleGlobals::c, &ScaleGlobals::alpha);
}
