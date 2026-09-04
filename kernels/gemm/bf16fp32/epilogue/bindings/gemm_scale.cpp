#include "gemm_base.cuh"
#include "stream.cuh"
#include "scale.cuh"
#include "pyutils/pyutils.cuh"

void dispatch(ScaleGlobals g) { launch<ScaleEpilogue, ScaleGlobals>(g); }
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk scale epilogue: out = alpha * (A@B)";
    hkstream::bind(m);
    py::bind_function<dispatch>(m, "dispatch",
        &ScaleGlobals::a, &ScaleGlobals::b, &ScaleGlobals::c, &ScaleGlobals::alpha);
}
