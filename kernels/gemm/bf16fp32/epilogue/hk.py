"""hk.py - a friendly Python API over the compiled GEMM-epilogue kernels.

The core op is a matmul; the epilogue is an optional modifier fused into it. Extra epilogue inputs
are passed BY NAME (validated against the registry), so a swapped `r`/`gamma` can't silently compute
the wrong thing.

    out  = hk.matmul(A, B)                                    # plain GEMM  (A @ B)
    out  = hk.matmul(A, B, epilogue="scale", alpha=0.5)       # 0.5 * (A @ B)
    out  = hk.matmul(A, B, epilogue="rmsnorm_scale", r=r, gamma=gamma)
    out  = hk.matmul(A, B, epilogue="residual_add", residual=res)
    out  = hk.matmul(A, B, epilogue="swiglu")                 # dim-reducing: [M,2*d_ff] -> [M,d_ff]
    out  = hk.matmul(A, B, epilogue="scale", alpha=0.5, sync=False)   # hot path: skip the sync
    hk.available()                                            # {epilogue name -> its keyword args}

Cross-entropy and the RMSNorm sublayer are DIFFERENT operation shapes (a per-row loss, and a
two-GEMM chain), so they are their own functions in this same module rather than `matmul` epilogues:

    loss = hk.cross_entropy(h, W_vocab, labels)              # loss[M]; logits never materialized
    loss = hk.cross_entropy(h, W_vocab, labels, gamma=g)    # RMS -> cross-entropy path
    out  = hk.rmsnorm_sublayer(X, W0, residual, gamma, W1)  # rmsnorm(X@W0+res, gamma) @ W1

Conveniences: B is passed normally (the wrapper transposes it for the kernel); scalars are plain
Python numbers (wrapped into the 1-element fp32 GPU tensor the kernel wants); tensors are moved to
CUDA / bf16 / contiguous; the output buffer is allocated and returned. `epilogue=None` is a plain
GEMM. `sync=False` skips the post-launch synchronize (for chaining several calls on a hot path).
`b_transposed=True` means B is already the kernel's [N,K] operand (transpose a static weight once
with hk.transpose(W)) -- skips the per-call copy. Shapes must satisfy M, N % 256 and K % 128.

Requires the compiled tk_*.so and util/ (for the registry) on the path -- run from this directory
or add it to sys.path / PYTHONPATH.
"""
import os, sys, importlib
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "util"))
from epilogue_testlib import EPILOGUES   # single source of truth: name -> {module, arg_names, ...}

DTYPE = torch.bfloat16


def _coerce(x):
    """A plain user value -> what the kernel expects:
    float/int -> a 1-element fp32 GPU tensor (e.g. a scalar alpha);
    tensor    -> CUDA, bf16, contiguous."""
    if isinstance(x, (int, float)):
        return torch.full((1,), float(x), dtype=torch.float32, device="cuda")
    return x.to(device="cuda", dtype=DTYPE).contiguous()


def _prep(A, B, b_transposed=False):
    """Cast A to bf16; get the kernel's transposed B operand; return (A, Bt, M, N, K).
    b_transposed=True: B is already that operand ([N,K]) -- skip the per-call copy."""
    A = _coerce(A)
    if b_transposed:
        Bt = _coerce(B); N, K2 = Bt.shape           # already the kernel operand
    else:
        B = _coerce(B); K2, N = B.shape
        Bt = B.t().contiguous()                     # kernel computes A @ Bt.t() == A @ B
    M, K = A.shape
    assert K == K2, f"inner dims disagree: A is {tuple(A.shape)}, B implies K={K2}"
    return A, Bt, M, N, K


def transpose(W):
    """Transpose+contiguous a static weight ONCE (CUDA/bf16), to reuse via matmul(..., b_transposed=True)."""
    return W.to(device="cuda", dtype=DTYPE).t().contiguous()


def available():
    """{epilogue name -> its extra keyword-arg names}, from the registry. `epilogue=None` (plain GEMM)
    takes none. Use it to see what each epilogue needs without reading the C++ binding, e.g.
    {'residual_add': ['residual'], 'rmsnorm_scale': ['r', 'gamma'], 'scale': ['alpha'], ...}."""
    return {name: list(spec.get("arg_names", [])) for name, spec in sorted(EPILOGUES.items())}


def matmul(A, B, epilogue=None, *, b_transposed=False, sync=True, **eargs):
    """GEMM `A @ B`, optionally with a fused `epilogue`.

    epilogue : registry name (see hk.available()); None -> plain GEMM.
    **eargs  : the epilogue's own inputs, passed BY NAME (e.g. epilogue="rmsnorm_scale", r=r,
               gamma=gamma). Validated against the registry -- a missing/misspelled/extra name
               raises, and there is no positional order to swap.
    sync     : True (default) synchronizes after the launch; False skips it (hot-path chaining).
    b_transposed : B is already the kernel's [N,K] operand (see hk.transpose) -> skip the copy.
    Returns the output tensor (out_shape handles dim-changing epilogues; swiglu halves N)."""
    name = "noop" if epilogue is None else epilogue
    try:
        spec = EPILOGUES[name]
    except KeyError:
        raise ValueError(f"unknown epilogue {epilogue!r}; available: {sorted(EPILOGUES)}")

    # validate the keyword epilogue args against the registry (names + arity)
    argn = list(spec.get("arg_names", []))
    missing = [n for n in argn if n not in eargs]
    unknown = [k for k in eargs if k not in argn]
    if missing or unknown:
        detail = "; ".join(([f"missing {missing}"] if missing else [])
                           + ([f"unexpected {unknown}"] if unknown else []))
        raise ValueError(f"epilogue {name!r} expects keyword args {argn}: {detail}")

    # dim-changing epilogues (swiglu): permute the gate_up weight's columns once so gate[j]/value[j]
    # land register-co-resident. b_transposed callers must have pre-permuted.
    if spec.get("weight_perm"):
        if b_transposed:
            raise ValueError(f"{name!r} needs the gate_up column permutation; pass b_transposed=False "
                             f"(hk.transpose only transposes, it does not permute).")
        B = _coerce(B)                                       # [d_model, weight width N]
        B = B[:, spec["weight_perm"](B.shape[1]).to(B.device)].contiguous()

    A, Bt, M, N, K = _prep(A, B, b_transposed)
    oh = spec.get("out_shape")
    out_rows, out_cols = oh(M, N, K) if oh else (M, N)
    C = torch.empty(out_rows, out_cols, dtype=DTYPE, device="cuda")

    mod = importlib.import_module(spec["module"])
    mod.dispatch(A, Bt, C, *[_coerce(eargs[n]) for n in argn])
    if sync:
        torch.cuda.synchronize()
    return C


def cross_entropy(h, W_vocab, labels, *, gamma=None, valid_n=None):
    """Fused forward cross-entropy: loss[M] = logsumexp(h @ W_vocab) - (h @ W_vocab)[labels]; the
    [M, vocab] logits are never materialized (a second aux kernel finishes the softmax + target dot).

    A separate op from matmul() -- it returns a per-row loss (not an [M,N] tile) and runs two kernels.
    gamma given -> the RMS -> cross-entropy path (rmsnorm(h, gamma) @ W_vocab). valid_n -> the real
    vocab when W_vocab is padded up to a 256 multiple (columns >= valid_n are masked out)."""
    from cross_entropy import make_ce, make_ce_rms   # util/ (lazy: only needs tk_cross_entropy/tk_ce_* when called)
    fwd = make_ce_rms(W_vocab, gamma, valid_n=valid_n) if gamma is not None \
        else make_ce(W_vocab, valid_n=valid_n)
    return fwd(h, labels)


def rmsnorm_sublayer(X, W0, residual, gamma, W1):
    """Fused RMSNorm sublayer: out = rmsnorm(X @ W0 + residual, gamma) @ W1. Two GEMMs + one aux
    reduce; the [M,N] intermediate never round-trips HBM. A separate op from matmul() (two GEMMs,
    not one). For a hot loop build once with util.block_chain.make_fused_rmsnorm_block(W0, W1)."""
    from block_chain import fused_rmsnorm_block   # util/ (lazy: needs tk_residual_rms_partials/tk_rms_reduce/tk_rmsnorm_scale)
    return fused_rmsnorm_block(X, W0, residual, gamma, W1)
