#include "gemm_base.cuh"
#include "rope.cuh"
#include "pyutils/pyutils.cuh"

void dispatch(RopeGlobals g) { launch<RopeEpilogue, RopeGlobals>(g); }
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk RoPE epilogue (interleaved; permuted weight + cos_sin)";
    py::bind_function<dispatch>(m, "dispatch",
        &RopeGlobals::a, &RopeGlobals::b, &RopeGlobals::c, &RopeGlobals::cos_sin);
}
