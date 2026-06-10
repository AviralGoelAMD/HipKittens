#include "gemm_base.cuh"
#include "pyutils/pyutils.cuh"
void dispatch_micro(gemm_args_base g) { launch_micro<NoOpEpilogue, gemm_args_base>(g); }
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk_kernel epilogue module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro",
        &gemm_args_base::a, &gemm_args_base::b, &gemm_args_base::c);
}
