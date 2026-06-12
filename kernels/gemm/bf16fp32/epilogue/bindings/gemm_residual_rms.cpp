#include "gemm_base.cuh"
#include "residual_rms.cuh"
#include "pyutils/pyutils.cuh"

void dispatch(ResidualRMSPartialsGlobals g) { launch<ResidualRMSPartialsEpilogue, ResidualRMSPartialsGlobals>(g); }
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk residual + partial-RMS + gamma epilogue";
    py::bind_function<dispatch>(m, "dispatch",
        &ResidualRMSPartialsGlobals::a, &ResidualRMSPartialsGlobals::b, &ResidualRMSPartialsGlobals::c,
        &ResidualRMSPartialsGlobals::residual, &ResidualRMSPartialsGlobals::gamma,
        &ResidualRMSPartialsGlobals::partials, &ResidualRMSPartialsGlobals::save);
}
