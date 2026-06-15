#include "gemm_base.cuh"
#include "rmsnorm_scale.cuh"
#include "pyutils/pyutils.cuh"

void dispatch(RMSNormScaleGlobals g) {
    if (g.r.rows() != 1 || g.r.cols() != g.c.rows())
        throw std::runtime_error("rmsnorm_scale: r must be [1,1,1,M] (r.cols()==c.rows())");
    if (g.gamma.rows() != 1 || g.gamma.cols() != g.c.cols())
        throw std::runtime_error("rmsnorm_scale: gamma must be [1,1,1,N] (gamma.cols()==c.cols())");
    launch<RMSNormScaleEpilogue, RMSNormScaleGlobals>(g);
}
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk RMSNorm-scale epilogue: out = (A@B) * r * gamma";
    py::bind_function<dispatch>(m, "dispatch",
        &RMSNormScaleGlobals::a, &RMSNormScaleGlobals::b, &RMSNormScaleGlobals::c,
        &RMSNormScaleGlobals::r, &RMSNormScaleGlobals::gamma);
}
