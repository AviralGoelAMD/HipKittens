#include "gemm_base.cuh"
#include "partialrms.cuh"
#include "pyutils/pyutils.cuh"

void dispatch(PartialRMSGlobals g) {
    if (g.partials.cols() != g.c.rows())
        throw std::runtime_error("partialrms: partials must be [1,1,N/REG_BLOCK_N,M] (partials.cols()==c.rows())");
    if (g.partials.rows() != g.c.cols() / REG_BLOCK_N)
        throw std::runtime_error("partialrms: partials.rows() must equal N/REG_BLOCK_N (group count)");
    launch<PartialRMSEpilogue, PartialRMSGlobals>(g);
}
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk partial RMS sum-of-squares epilogue";
    py::bind_function<dispatch>(m, "dispatch",
        &PartialRMSGlobals::a, &PartialRMSGlobals::b, &PartialRMSGlobals::c, &PartialRMSGlobals::partials);
}
