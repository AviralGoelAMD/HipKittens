#include "gemm_base.cuh"
#include "partialrms.cuh"
#include "pyutils/pyutils.cuh"

void dispatch(PartialRMSGlobals g) {
    if (g.partials.cols() != g.a.rows())
        throw std::runtime_error("partialrms: partials must be [1,1,N/REG_BLOCK_N,M] (partials.cols()==a.rows()==M)");
    if (g.partials.rows() != g.b.rows() / REG_BLOCK_N)
        throw std::runtime_error("partialrms: partials.rows() must equal N/REG_BLOCK_N (group count; N==b.rows())");
    launch<PartialRMSEpilogue, PartialRMSGlobals>(g);
}
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk partial-RMS epilogue: emits per-(row, N/REG_BLOCK_N col-group) Sigma((A@B)^2) partials ONLY (no C store); aux reduces the groups -> per-row 1/rms";
    py::bind_function<dispatch>(m, "dispatch",
        &PartialRMSGlobals::a, &PartialRMSGlobals::b, &PartialRMSGlobals::partials);
}
