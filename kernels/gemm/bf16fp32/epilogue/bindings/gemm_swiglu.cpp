#include "gemm_base.cuh"
#include "swiglu.cuh"
#include "pyutils/pyutils.cuh"

void dispatch(gemm_args_base g) { launch<SwigluEpilogue, gemm_args_base>(g); }
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk swiglu epilogue: out = silu(gate)*value, [M,2*d_ff] -> [M,d_ff]";
    py::bind_function<dispatch>(m, "dispatch",
        &gemm_args_base::a, &gemm_args_base::b, &gemm_args_base::c);
}
