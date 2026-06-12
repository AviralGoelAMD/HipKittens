#include "gemm_base.cuh"
#include "partialrms.cuh"
#include "pyutils/pyutils.cuh"

void dispatch(PartialRMSGlobals g) { launch<PartialRMSEpilogue, PartialRMSGlobals>(g); }
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk partial RMS sum-of-squares epilogue";
    py::bind_function<dispatch>(m, "dispatch",
        &PartialRMSGlobals::a, &PartialRMSGlobals::b, &PartialRMSGlobals::c, &PartialRMSGlobals::partials);
}
