#include "gemm_base.cuh"
#include "epilogue_tile_ops.cuh"   // residual_add
#include "pyutils/pyutils.cuh"

// out = (A@B) + residual   (the [M,N] skip connection), fused onto the GEMM epilogue so the
// intermediate never round-trips HBM.
struct ResidualAddGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<bf16,-1,-1,-1,-1> residual;   // [1,1,M,N] skip connection
    hipStream_t stream;
};
struct ResidualAddEpilogue {
    template<typename G, typename Accum>
    static __device__ inline void apply(const G& g, Accum& C, int row,int col,int wr,int wc){
        residual_add(g, C, row,col,wr,wc);   // load residual tile, add into the accumulator
        store_C(g, C, row,col,wr,wc);
    }
};

void dispatch(ResidualAddGlobals g) { launch<ResidualAddEpilogue, ResidualAddGlobals>(g); }
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk residual-add epilogue";
    py::bind_function<dispatch>(m, "dispatch",
        &ResidualAddGlobals::a, &ResidualAddGlobals::b, &ResidualAddGlobals::c, &ResidualAddGlobals::residual);
}
