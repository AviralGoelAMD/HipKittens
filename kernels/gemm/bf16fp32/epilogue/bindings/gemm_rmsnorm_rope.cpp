#include "gemm_base.cuh"
#include "rmsnorm_rope.cuh"
#include "pyutils/pyutils.cuh"

void dispatch(RmsnormRopeGlobals g) {
    if (g.r.rows() != 1 || g.r.cols() != g.c.rows())
        throw std::runtime_error("rmsnorm_rope: r must be [1,1,1,M] (r.cols()==c.rows())");
    if (g.cos_sin.rows() != g.c.rows() || g.cos_sin.cols() != g.c.cols())
        throw std::runtime_error("rmsnorm_rope: cos_sin must be [M,N] matching c");
    launch<RmsnormRopeEpilogue, RmsnormRopeGlobals>(g);
}
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk RMS->RoPE epilogue: out = RoPE(r * (A@B)), interleaved (requires rope_perm'd weight + cos_sin)";
    py::bind_function<dispatch>(m, "dispatch",
        &RmsnormRopeGlobals::a, &RmsnormRopeGlobals::b, &RmsnormRopeGlobals::c,
        &RmsnormRopeGlobals::r, &RmsnormRopeGlobals::cos_sin);
}
