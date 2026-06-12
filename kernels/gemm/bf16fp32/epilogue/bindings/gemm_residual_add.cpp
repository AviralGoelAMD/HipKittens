#include "gemm_base.cuh"
#include "residual_add.cuh"
#include "pyutils/pyutils.cuh"

void dispatch(ResidualAddGlobals g) { launch<ResidualAddEpilogue, ResidualAddGlobals>(g); }
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk residual-add epilogue";
    py::bind_function<dispatch>(m, "dispatch",
        &ResidualAddGlobals::a, &ResidualAddGlobals::b, &ResidualAddGlobals::c, &ResidualAddGlobals::residual);
}
