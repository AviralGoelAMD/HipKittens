#include "gemm_base.cuh"
#include "silu.cuh"
#include "pyutils/pyutils.cuh"

void dispatch(gemm_args_base g) { launch<SiluEpilogue, gemm_args_base>(g); }

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "GEMM + SiLU: c = silu(a @ b.T), with a = [M,K], b = [N,K], c = [M,N] (bf16)";
    py::bind_function<dispatch>(m, "dispatch",
        &gemm_args_base::a, &gemm_args_base::b, &gemm_args_base::c);
}
