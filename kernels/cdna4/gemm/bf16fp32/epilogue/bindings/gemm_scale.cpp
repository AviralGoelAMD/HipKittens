#include "gemm_base.cuh"
#include "scale.cuh"
#include "pyutils/pyutils.cuh"

void dispatch(ScaleGlobals g) { launch<ScaleEpilogue, ScaleGlobals>(g); }

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "GEMM + scale: c = alpha * (a @ b.T), with a = [M,K], b = [N,K], c = [M,N] (bf16), "
              "alpha = 1-element fp32 CUDA tensor";
    py::bind_function<dispatch>(m, "dispatch",
        &ScaleGlobals::a, &ScaleGlobals::b, &ScaleGlobals::c, &ScaleGlobals::alpha);
}
