#include "gemm_base.cuh"
#include "residual_add.cuh"
#include "pyutils/pyutils.cuh"

void dispatch(ResidualAddGlobals g) {
    if (g.residual.rows() != g.c.rows() || g.residual.cols() != g.c.cols())
        throw std::runtime_error("residual_add: residual must be [M,N] matching c");
    launch<ResidualAddEpilogue, ResidualAddGlobals>(g);
}
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk residual-add epilogue: out = (A@B) + residual ([M,N] skip connection)";
    py::bind_function<dispatch>(m, "dispatch",
        &ResidualAddGlobals::a, &ResidualAddGlobals::b, &ResidualAddGlobals::c, &ResidualAddGlobals::residual);
}
