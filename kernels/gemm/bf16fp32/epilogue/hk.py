"""hk.py - a friendly Python API over the compiled GEMM-epilogue kernels.

Registry-driven: the set of epilogues, and each one's kernel module + argument order, come from
the EPILOGUES registry (the single source of truth shared with the test/bench runners). Adding a
new epilogue needs NO change to this file.

    out = hk.run("scale", A, B, 0.5)            # 0.5 * (A @ B)
    out = hk.run("rmsnorm_scale", A, B, r, gamma)
    out = hk.run("residual", A, B, residual)
    hk.available()                              # the epilogue names you can run

Conveniences: B is passed normally (the wrapper transposes it for the kernel); scalars are plain
Python floats (wrapped into the 1-element fp32 GPU tensor the kernel wants); tensors are moved to
CUDA / bf16 / contiguous; the output buffer is allocated and returned. Shapes must satisfy
M, N % 256 and K % 128 (the kernels enforce this and raise otherwise).

Requires the compiled tk_*.so and util/ (for the registry) on the path -- run from this directory
or add it to sys.path / PYTHONPATH.
"""
import os, sys, importlib
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "util"))
from epilogue_testlib import EPILOGUES   # single source of truth: name -> {module, args order, ...}

DTYPE = torch.bfloat16


def _coerce(x):
    """A plain user value -> what the kernel expects:
    float/int -> a 1-element fp32 GPU tensor (e.g. a scalar alpha);
    tensor    -> CUDA, bf16, contiguous."""
    if isinstance(x, (int, float)):
        return torch.full((1,), float(x), dtype=torch.float32, device="cuda")
    return x.to(device="cuda", dtype=DTYPE).contiguous()


def _prep(A, B):
    """Cast A,B to bf16; transpose B for the kernel; allocate the output C."""
    A = _coerce(A); B = _coerce(B)
    (M, K), (K2, N) = A.shape, B.shape
    assert K == K2, f"inner dims disagree: A is {tuple(A.shape)}, B is {tuple(B.shape)}"
    Bt = B.t().contiguous()                          # kernel computes A @ Bt.t() == A @ B
    C = torch.empty(M, N, dtype=DTYPE, device="cuda")
    return A, Bt, C


def available():
    """The epilogue names run() accepts (straight from the registry)."""
    return sorted(EPILOGUES)


def run(name, A, B, *extra):
    """GEMM + the named epilogue. `extra` are that epilogue's own inputs, in binding order
    (e.g. "scale" -> alpha; "rmsnorm_scale" -> r, gamma; "residual" -> residual). Returns the
    output tensor."""
    try:
        spec = EPILOGUES[name]
    except KeyError:
        raise ValueError(f"unknown epilogue '{name}'; available: {available()}")
    mod = importlib.import_module(spec["module"])
    A, Bt, C = _prep(A, B)
    mod.dispatch(A, Bt, C, *[_coerce(x) for x in extra])
    torch.cuda.synchronize()
    return C
