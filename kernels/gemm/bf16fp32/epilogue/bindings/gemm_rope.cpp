#include "gemm_base.cuh"
#include "rope.cuh"
#include "pyutils/pyutils.cuh"

void dispatch(RopeGlobals g) {
    if (g.cos_sin.rows() != g.c.rows() || g.cos_sin.cols() != g.c.cols())
        throw std::runtime_error("rope: cos_sin must be [M,N] matching c");
    launch<RopeEpilogue, RopeGlobals>(g);
}
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk RoPE epilogue: out = RoPE(A@B), interleaved (requires rope_perm'd weight + cos_sin)";
    py::bind_function<dispatch>(m, "dispatch",
        &RopeGlobals::a, &RopeGlobals::b, &RopeGlobals::c, &RopeGlobals::cos_sin);
}
