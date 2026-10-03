#include "gemm_base.cuh"
#include "stream.cuh"
#include "pyutils/pyutils.cuh"
void dispatch(gemm_args_base g) { launch<NoOpEpilogue, gemm_args_base>(g); }
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk noop epilogue: plain GEMM (store only)";
    hkstream::bind(m);
    py::bind_function<dispatch>(m, "dispatch",
        &gemm_args_base::a, &gemm_args_base::b, &gemm_args_base::c);
}