#include "aux_reduce.cuh"
#include "pyutils/pyutils.cuh"
#include <stdexcept>

// tk_aux_rms.reduce(partials, r) -> writes per-row 1/rms into r.
void dispatch(aux_globals g) {
    // The exported reduce() interface has all-dynamic gl dims, so make_gl performs NO shape check:
    // a mis-shaped caller (e.g. partials as [1,1,M,N/64]) would silently reduce the wrong axis.
    //   Enforce the layout contract here:
    //   partials = [1,1, N/REG_BLOCK_N, M] (row=M on the LAST axis),  r = [1,1, 1, M].
    const int M = g.r.cols();
    if (g.r.rows() != 1)
        throw std::runtime_error("aux_rms.reduce: r must be [1,1,1,M] (r.rows() must be 1).");
    if (g.partials.cols() != M)
        throw std::runtime_error("aux_rms.reduce: partials must be [1,1,N/REG_BLOCK_N,M] with M on the last axis (partials.cols() must equal r.cols()).");
    if (g.partials.rows() < 1)
        throw std::runtime_error("aux_rms.reduce: partials must have at least one group (partials.rows() >= 1).");
    constexpr int TPB = 256;
    rms_reduce<<<(M + TPB - 1) / TPB, TPB, 0, g.stream>>>(g.partials, g.r);
}
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk aux RMS reduce: per-(group,row) partials -> per-row 1/rms";
    py::bind_function<dispatch>(m, "reduce", &aux_globals::partials, &aux_globals::r);
}
