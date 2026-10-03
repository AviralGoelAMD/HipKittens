#include "gemm_base.cuh"
#include "partialrms.cuh"
#include "pyutils/pyutils.cuh"

void dispatch(PartialRMSGlobals g) {
    if (g.partials.cols() != g.a.rows() || g.partials.rows() != g.b.rows() / REG_BLOCK_N)
        throw std::runtime_error("partialrms: partials must be [N/64, M] fp32 (N = b.rows(), M = a.rows())");
    launch<PartialRMSEpilogue, PartialRMSGlobals>(g);
}

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "GEMM + partial RMS: partials[g, m] = sum of (a @ b.T)[m, cols of group g]^2, "
              "a = [M,K], b = [N,K] (bf16), partials = [N/64, M] (fp32); no c is written";
    py::bind_function<dispatch>(m, "dispatch",
        &PartialRMSGlobals::a, &PartialRMSGlobals::b, &PartialRMSGlobals::partials);
}
