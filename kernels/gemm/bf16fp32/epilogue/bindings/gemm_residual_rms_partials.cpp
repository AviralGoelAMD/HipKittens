#include "gemm_base.cuh"
#include "residual_rms.cuh"
#include "pyutils/pyutils.cuh"

void dispatch(ResidualRMSPartialsGlobals g) {
    const int M = g.c.rows(), N = g.c.cols();
    if (g.residual.rows() != M || g.residual.cols() != N || g.save.rows() != M || g.save.cols() != N)
        throw std::runtime_error("residual_rms: residual and save must be [M,N] matching c");
    if (g.gamma.rows() != 1 || g.gamma.cols() != N)
        throw std::runtime_error("residual_rms: gamma must be [1,1,1,N]");
    if (g.partials.cols() != M)
        throw std::runtime_error("residual_rms: partials must be [1,1,N/REG_BLOCK_N,M] (partials.cols()==M)");
    if (g.partials.rows() != N / REG_BLOCK_N)
        throw std::runtime_error("residual_rms: partials.rows() must equal N/REG_BLOCK_N (group count)");
    launch<ResidualRMSPartialsEpilogue, ResidualRMSPartialsGlobals>(g);
}
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk residual-RMS-partials epilogue: h1 = A@B + residual -> save h1, emit Sigma(h1^2) partials, store h1*gamma";
    py::bind_function<dispatch>(m, "dispatch",
        &ResidualRMSPartialsGlobals::a, &ResidualRMSPartialsGlobals::b, &ResidualRMSPartialsGlobals::c,
        &ResidualRMSPartialsGlobals::residual, &ResidualRMSPartialsGlobals::gamma,
        &ResidualRMSPartialsGlobals::partials, &ResidualRMSPartialsGlobals::save);
}
