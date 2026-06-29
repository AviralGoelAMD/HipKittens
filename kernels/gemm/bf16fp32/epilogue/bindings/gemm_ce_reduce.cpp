#include "aux_reduce.cuh"
#include "pyutils/pyutils.cuh"
#include "pyutils/util.cuh"
#include <stdexcept>

// tk_ce_reduce.reduce(max_buf, sumexp_buf, a, b, labels, valid_n, loss): warp-per-row combine of the
// per-(group,row) softmax partials into logsumexp PLUS the O(K) target dot <h[r], Wt[label[r]]>
// -> per-row loss. reduce_rms(..., r, loss) is the RMS variant (target *= per-row inv-rms r).
// a=h [M,K] bf16; b=Wt [N,K] bf16; partials=[1,1,N/REG_BLOCK_N,M]; labels/r/loss=[1,1,1,M].
void dispatch(ce_aux_globals g) {
    const int M = g.loss.cols();
    if (g.loss.rows() != 1)
        throw std::runtime_error("ce_reduce: loss must be [1,1,1,M] (loss.rows() must be 1).");
    if (g.max_buf.cols() != M || g.sumexp_buf.cols() != M)
        throw std::runtime_error("ce_reduce: max/sumexp buffers must be [1,1,N/REG_BLOCK_N,M] (cols()==M).");
    if (g.labels.cols() != M)
        throw std::runtime_error("ce_reduce: labels must be [1,1,1,M] (labels.cols()==M).");
    if (g.a.rows() != M || g.a.cols() != g.b.cols())
        throw std::runtime_error("ce_reduce: a (h) must be [M,K] and share K with b (Wt) [N,K].");
    const int groups = g.max_buf.rows();
    if (groups < 1 || g.sumexp_buf.rows() != groups)
        throw std::runtime_error("ce_reduce: max/sumexp buffers must share the same group count (rows() >= 1).");
    constexpr int TPB = 256;                                  // 4 wavefronts/block
    const long long total = (long long)M * kittens::WARP_THREADS;   // one wavefront per row
    cross_entropy_reduce<false><<<(int)((total + TPB - 1) / TPB), TPB, 0, g.stream>>>(
        g.max_buf, g.sumexp_buf, g.a, g.b, g.labels, g.valid_n.raw_ptr, nullptr, g.loss);
    CHECK_CUDA_ERROR(hipGetLastError());
}

void dispatch_rms(ce_aux_rms_globals g) {
    const int M = g.loss.cols();
    if (g.loss.rows() != 1)
        throw std::runtime_error("ce_reduce_rms: loss must be [1,1,1,M] (loss.rows() must be 1).");
    if (g.max_buf.cols() != M || g.sumexp_buf.cols() != M)
        throw std::runtime_error("ce_reduce_rms: max/sumexp buffers must be [1,1,N/REG_BLOCK_N,M] (cols()==M).");
    if (g.labels.cols() != M)
        throw std::runtime_error("ce_reduce_rms: labels must be [1,1,1,M] (labels.cols()==M).");
    if (g.r.rows() != 1 || g.r.cols() != M)
        throw std::runtime_error("ce_reduce_rms: r must be [1,1,1,M] (r.cols()==M).");
    if (g.a.rows() != M || g.a.cols() != g.b.cols())
        throw std::runtime_error("ce_reduce_rms: a (h) must be [M,K] and share K with b (Wt) [N,K].");
    const int groups = g.max_buf.rows();
    if (groups < 1 || g.sumexp_buf.rows() != groups)
        throw std::runtime_error("ce_reduce_rms: max/sumexp buffers must share the same group count (rows() >= 1).");
    constexpr int TPB = 256;
    const long long total = (long long)M * kittens::WARP_THREADS;   // one wavefront per row
    cross_entropy_reduce<true><<<(int)((total + TPB - 1) / TPB), TPB, 0, g.stream>>>(
        g.max_buf, g.sumexp_buf, g.a, g.b, g.labels, g.valid_n.raw_ptr, g.r.raw_ptr, g.loss);
    CHECK_CUDA_ERROR(hipGetLastError());
}

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk cross-entropy reduce: softmax partials + O(K) target dot -> per-row loss";
    py::bind_function<dispatch>(m, "reduce",
        &ce_aux_globals::max_buf, &ce_aux_globals::sumexp_buf,
        &ce_aux_globals::a, &ce_aux_globals::b, &ce_aux_globals::labels, &ce_aux_globals::valid_n, &ce_aux_globals::loss);
    py::bind_function<dispatch_rms>(m, "reduce_rms",
        &ce_aux_rms_globals::max_buf, &ce_aux_rms_globals::sumexp_buf,
        &ce_aux_rms_globals::a, &ce_aux_rms_globals::b, &ce_aux_rms_globals::labels,
        &ce_aux_rms_globals::valid_n, &ce_aux_rms_globals::r, &ce_aux_rms_globals::loss);
}
