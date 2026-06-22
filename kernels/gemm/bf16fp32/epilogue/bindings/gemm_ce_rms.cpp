#include "gemm_base.cuh"
#include "cross_entropy.cuh"
#include "pyutils/pyutils.cuh"

// tk_ce_rms.dispatch(a, b, max_buf, sumexp_buf, r): fused RMS->forward cross-entropy GEMM
// epilogue. r scales the fp32 accumulator per row (rmsnorm(h,gamma)@W_lm == r*(h @ gamma-folded
// W_lm)) BEFORE the softmax; emits per-(group,row) softmax partials only. The [M,vocab] logits are
// NEVER materialized; the aux kernel computes the (r-scaled) target logit as the O(K) dot.
void dispatch(CrossEntropyRmsGlobals g) {
    const int M = g.a.rows(), N = g.b.rows();
    if (g.r.rows() != 1 || g.r.cols() != M)
        throw std::runtime_error("ce_rms: r must be [1,1,1,M] (r.cols()==M)");
    if (g.max_buf.cols() != M || g.sumexp_buf.cols() != M)
        throw std::runtime_error("ce_rms: max/sumexp buffers must be [1,1,N/REG_BLOCK_N,M] (cols()==M)");
    if (g.max_buf.rows() != N / REG_BLOCK_N || g.sumexp_buf.rows() != N / REG_BLOCK_N)
        throw std::runtime_error("ce_rms: max/sumexp buffers must have N/REG_BLOCK_N groups (rows())");
    launch<PartialLseRmsEpilogue, CrossEntropyRmsGlobals>(g);
}

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk RMS->forward cross-entropy: r*(A@B) then per-(group,row) softmax partials only";
    py::bind_function<dispatch>(m, "dispatch",
        &CrossEntropyRmsGlobals::a, &CrossEntropyRmsGlobals::b,
        &CrossEntropyRmsGlobals::max_buf, &CrossEntropyRmsGlobals::sumexp_buf,
        &CrossEntropyRmsGlobals::r);
}
