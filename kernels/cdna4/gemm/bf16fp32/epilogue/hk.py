"""hk: Python API over the compiled GEMM-epilogue kernels (the tk_<name> modules).

Prepare static weights once, then call:

    w  = hk.prepare(W, layout="swiglu", gamma=g)        # W: natural [K, N] bf16
    cs = hk.prepare_rope_table(cos_sin)                  # natural [M, N] bf16 [cos, sin] table
    y  = hk.matmul(x, w, epilogue="rmsnorm_swiglu", r=r)
    r  = hk.inv_rms(x, hk.prepare(W2))                   # [M] fp32 = 1 / rms(x @ W2)
    hk.available()                                       # {epilogue: [argument names]}

Every call launches on torch.cuda.current_stream() and does not synchronize. Inputs are never
converted: a wrong dtype, shape, device or layout raises. A scalar `alpha` may be a Python number.
Shapes: M and N multiples of 256, K a multiple of 128. The tk_<name> modules must be built
(make KERNEL=<name>) and importable.
"""
import importlib
from dataclasses import dataclass

import torch

BF16, FP32 = torch.bfloat16, torch.float32


@dataclass(frozen=True)
class PreparedWeight:
    """A weight in kernel form: [N, K] bf16, columns permuted for `layout`, gamma optionally folded."""
    tensor: torch.Tensor
    layout: str
    gamma_folded: bool


@dataclass(frozen=True)
class RopeTable:
    """An [M, N] bf16 interleaved [cos, sin] table, columns permuted like a rope-layout weight."""
    tensor: torch.Tensor


@dataclass(frozen=True)
class _Spec:
    module: str
    args: tuple = ()             # ((name, kind), ...); kind: scalar | row | col | tile | rope
    layout: str = "plain"
    gamma_folded: bool = False
    half_width: bool = False


_EPILOGUES = {
    "noop":           _Spec("tk_noop"),
    "silu":           _Spec("tk_silu"),
    "scale":          _Spec("tk_scale", (("alpha", "scalar"),)),
    "residual_add":   _Spec("tk_residual_add", (("residual", "tile"),)),
    "rmsnorm_scale":  _Spec("tk_rmsnorm_scale", (("r", "row"), ("gamma", "col"))),
    "swiglu":         _Spec("tk_swiglu", layout="swiglu", half_width=True),
    "rmsnorm_swiglu": _Spec("tk_rmsnorm_swiglu", (("r", "row"),), layout="swiglu", gamma_folded=True, half_width=True),
    "rope":           _Spec("tk_rope", (("cos_sin", "rope"),), layout="rope"),
    "rmsnorm_rope":   _Spec("tk_rmsnorm_rope", (("r", "row"), ("cos_sin", "rope")), layout="rope", gamma_folded=True),
}


def available():
    """{epilogue name: [argument names]} accepted by hk.matmul (epilogue=None means noop)."""
    return {name: [a for a, _ in spec.args] for name, spec in _EPILOGUES.items()}


def _require(t, name, dtype, shape, device=None):
    """Raise unless t is a contiguous GPU tensor of `dtype` and `shape` (None matches any extent),
    on `device` when given."""
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
    shape_ok = t.dim() == len(shape) and all(s is None or s == d for s, d in zip(shape, t.shape))
    if t.dtype != dtype or not t.is_cuda or not t.is_contiguous() or not shape_ok:
        want = tuple("*" if s is None else s for s in shape)
        raise ValueError(f"{name} must be a contiguous {dtype} GPU tensor of shape {want}; got {t.dtype}, "
                         f"{'GPU' if t.is_cuda else 'CPU'}, {'contiguous' if t.is_contiguous() else 'non-contiguous'}, "
                         f"shape {tuple(t.shape)}")
    if device is not None and t.device != device:
        raise ValueError(f"{name} must be on {device} (the same device as x), got {t.device}")


def _pair_slots(pairs, device):
    """Column of each pair's first member after permutation: pair j -> (j/128)*256 + j%128."""
    j = torch.arange(pairs, device=device)
    return (j // 128) * 256 + j % 128


def _swiglu_perm(n, device):
    """perm[new_col] = old_col: gate j (old col j) -> slot(j), value j (old col n/2 + j) -> slot(j) + 128."""
    slot = _pair_slots(n // 2, device)
    perm = torch.empty(n, dtype=torch.long, device=device)
    perm[slot] = torch.arange(n // 2, device=device)
    perm[slot + 128] = torch.arange(n // 2, n, device=device)
    return perm


def _rope_perm(n, device):
    """perm[new_col] = old_col: pair k's even col 2k -> slot(k), odd col 2k+1 -> slot(k) + 128."""
    slot = _pair_slots(n // 2, device)
    k = torch.arange(n // 2, device=device)
    perm = torch.empty(n, dtype=torch.long, device=device)
    perm[slot] = 2 * k
    perm[slot + 128] = 2 * k + 1
    return perm


def prepare(W, layout="plain", gamma=None):
    """Turn a natural weight W [K, N] (bf16) into the kernel operand, once.

    gamma (bf16 [K]) is folded into W's rows (rmsnorm(x, gamma) @ W == r * (x @ (gamma * W))), for the
    rmsnorm_swiglu / rmsnorm_rope epilogues. layout "swiglu" or "rope" permutes the columns so each pair
    sits in one thread; "plain" leaves them. The result is transposed to [N, K]."""
    if layout not in ("plain", "swiglu", "rope"):
        raise ValueError(f"layout must be 'plain', 'swiglu' or 'rope', got {layout!r}")
    _require(W, "W", BF16, (None, None))
    K, N = W.shape
    if K % 128 or N % 256:
        raise ValueError(f"W must be [K, N] with K % 128 == 0 and N % 256 == 0, got {tuple(W.shape)}")
    if gamma is not None:
        _require(gamma, "gamma", BF16, (K,), W.device)
        W = (W.float() * gamma.float()[:, None]).to(BF16)
    if layout == "swiglu":
        W = W[:, _swiglu_perm(N, W.device)]
    elif layout == "rope":
        W = W[:, _rope_perm(N, W.device)]
    return PreparedWeight(W.t().contiguous(), layout, gamma is not None)


def prepare_rope_table(cos_sin):
    """Permute a natural interleaved [cos, sin] table (bf16 [M, N]) the way rope weights are permuted, once."""
    _require(cos_sin, "cos_sin", BF16, (None, None))
    if cos_sin.shape[1] % 256:
        raise ValueError(f"cos_sin must have N % 256 == 0, got {tuple(cos_sin.shape)}")
    return RopeTable(cos_sin[:, _rope_perm(cos_sin.shape[1], cos_sin.device)].contiguous())


def _argument(name, kind, value, M, N, device):
    if kind == "scalar":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return torch.full((1,), float(value), dtype=FP32, device=device)
        _require(value, name, FP32, (1,), device)
        return value
    if kind == "rope":
        if not isinstance(value, RopeTable):
            raise TypeError(f"{name} must come from hk.prepare_rope_table(...), got {type(value).__name__}")
        _require(value.tensor, name, BF16, (M, N), device)
        return value.tensor
    dtype, shape = {"row": (FP32, (M,)), "col": (BF16, (N,)), "tile": (BF16, (M, N))}[kind]
    _require(value, name, dtype, shape, device)
    return value


def _check_weight(w, layout, gamma_folded, what):
    if not isinstance(w, PreparedWeight):
        raise TypeError(f"w must come from hk.prepare(...), got {type(w).__name__}")
    if w.layout != layout:
        raise ValueError(f"{what} needs a {layout!r} weight, got {w.layout!r} (use hk.prepare(W, layout={layout!r}))")
    if w.gamma_folded != gamma_folded:
        need = "gamma folded in (hk.prepare(W, ..., gamma=g))" if gamma_folded else "no gamma folded in"
        raise ValueError(f"{what} needs a weight with {need}")


def matmul(x, w, epilogue=None, **args):
    """GEMM x @ W with a fused epilogue; returns a new bf16 output ([M, N], or [M, N/2] for swiglu variants).

    x: bf16 [M, K]. w: from hk.prepare with the layout / gamma the epilogue needs. args: the epilogue's
    inputs by name (see hk.available()). Launches on torch.cuda.current_stream()."""
    name = "noop" if epilogue is None else epilogue
    spec = _EPILOGUES.get(name)
    if spec is None:
        raise ValueError(f"unknown epilogue {epilogue!r}; available: {sorted(_EPILOGUES)}")
    _check_weight(w, spec.layout, spec.gamma_folded, f"epilogue {name!r}")
    expected = [a for a, _ in spec.args]
    if sorted(args) != sorted(expected):
        raise TypeError(f"epilogue {name!r} takes arguments {expected}, got {sorted(args)}")
    N, K = w.tensor.shape
    _require(x, "x", BF16, (None, K))
    _require(w.tensor, "w", BF16, (N, K), x.device)
    M = x.shape[0]
    values = [_argument(a, kind, args[a], M, N, x.device) for a, kind in spec.args]
    out = torch.empty(M, N // 2 if spec.half_width else N, dtype=BF16, device=x.device)
    with torch.cuda.device(x.device):                     # launch on x's GPU, on its current stream
        stream = torch.cuda.current_stream(x.device).cuda_stream
        importlib.import_module(spec.module).dispatch(x, w.tensor, out, *values, stream)
    return out


def inv_rms(x, w):
    """r = 1 / rms(x @ W) per row, fp32 [M] (eps 1e-5), via partialrms + rms_reduce. Needs a plain weight
    without gamma. Does not return x @ W itself."""
    _check_weight(w, "plain", False, "inv_rms")
    N, K = w.tensor.shape
    _require(x, "x", BF16, (None, K))
    _require(w.tensor, "w", BF16, (N, K), x.device)
    M = x.shape[0]
    partials = torch.empty(N // 64, M, dtype=FP32, device=x.device)
    r = torch.empty(M, dtype=FP32, device=x.device)
    with torch.cuda.device(x.device):                     # launch on x's GPU, on its current stream
        stream = torch.cuda.current_stream(x.device).cuda_stream
        importlib.import_module("tk_partialrms").dispatch(x, w.tensor, partials, stream)
        importlib.import_module("tk_rms_reduce").dispatch(partials, r, stream)
    return r
