#include "gemm_base.cuh"
#include "rmsnorm_swiglu.cuh"
#include "pyutils/pyutils.cuh"

void dispatch(RmsnormSwigluGlobals g) {
    if (g.r.rows() != 1 || g.r.cols() != g.c.rows())
        throw std::runtime_error("rmsnorm_swiglu: r must be [1,1,1,M] (r.cols()==c.rows())");
    launch<RmsnormSwigluEpilogue, RmsnormSwigluGlobals>(g);
}
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk RMS->SwiGLU epilogue: out = silu(gate)*value, gate|value = r*(A@B); dim-reducing [M,2*d_ff] -> [M,d_ff] (requires gate_up_perm'd + gamma-folded weight)";
    py::bind_function<dispatch>(m, "dispatch",
        &RmsnormSwigluGlobals::a, &RmsnormSwigluGlobals::b, &RmsnormSwigluGlobals::c,
        &RmsnormSwigluGlobals::r);
}
