#include "gemm_base.cuh"
#include "cross_entropy.cuh"
#include "pyutils/pyutils.cuh"

// tk_cross_entropy.dispatch(a, b, max_buf, sumexp_buf): fused forward cross-entropy GEMM epilogue.
// Emits per-(group,row) softmax partials only; the [M,vocab] logits are NEVER materialized
// (partials-only -> launch_partials_only, no `c`). The target logit is computed by the aux kernel
// as the O(K) dot <h, Wt[label]> (no in-epilogue O(vocab) gather).
void dispatch(CrossEntropyGlobals g) {
    const int M = g.a.rows(), N = g.b.rows();
    if (g.max_buf.cols() != M || g.sumexp_buf.cols() != M)
        throw std::runtime_error("cross_entropy: max/sumexp buffers must be [1,1,N/REG_BLOCK_N,M] (cols()==M)");
    if (g.max_buf.rows() != N / REG_BLOCK_N || g.sumexp_buf.rows() != N / REG_BLOCK_N)
        throw std::runtime_error("cross_entropy: max/sumexp buffers must have N/REG_BLOCK_N groups (rows())");
    launch<PartialLseEpilogue, CrossEntropyGlobals>(g);
}

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk forward fused cross-entropy: per-(group,row) softmax partials only";
    py::bind_function<dispatch>(m, "dispatch",
        &CrossEntropyGlobals::a, &CrossEntropyGlobals::b,
        &CrossEntropyGlobals::max_buf, &CrossEntropyGlobals::sumexp_buf);
}
