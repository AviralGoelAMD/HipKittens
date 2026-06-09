#include "auxiliary_reduce.cuh"
#include "pyutils/pyutils.cuh"

// Stage 2 Task 2.4 binding: tk_aux_rms.reduce(partials, r) -> writes per-row 1/rms into r.
void dispatch_micro(aux_globals g) {
    int M = g.r.cols();
    constexpr int TPB = 256;
    rms_reduce<<<(M + TPB - 1) / TPB, TPB, 0, g.stream>>>(g.partials, g.r);
}
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk aux RMS reduce: per-(group,row) partials -> per-row 1/rms";
    py::bind_function<dispatch_micro>(m, "reduce", &aux_globals::partials, &aux_globals::r);
}
