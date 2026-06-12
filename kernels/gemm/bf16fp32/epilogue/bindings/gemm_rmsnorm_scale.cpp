#include "gemm_base.cuh"
#include "rmsnorm_scale.cuh"
#include "pyutils/pyutils.cuh"

void dispatch(RMSNormScaleGlobals g) { launch<RMSNormScaleEpilogue, RMSNormScaleGlobals>(g); }
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk RMSNorm-scale epilogue";
    py::bind_function<dispatch>(m, "dispatch",
        &RMSNormScaleGlobals::a, &RMSNormScaleGlobals::b, &RMSNormScaleGlobals::c,
        &RMSNormScaleGlobals::r, &RMSNormScaleGlobals::gamma);
}
