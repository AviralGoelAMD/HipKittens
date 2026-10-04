"""tk_rmsnorm_swiglu: c = silu(gate) * value with [gate | value] = r[:, None] * (a @ b.T), [M, N] -> [M, N/2]
(bf16, r = [M] fp32). The caller permutes b's rows as for SwiGLU (and folds the norm's gamma into b,
which this test does not need). Fast SiLU allows one bf16 step."""
import pytest
import torch

from conftest import BAD_SHAPES, DEV, SHAPES, assert_within_bf16_step, gemm_exact, inputs, module, silu_kernel
from test_swiglu import permute

SILU_HALF = 0.31122966560092725   # silu(0.5)


def r_like(M, seed=4):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.rand(M, generator=g, device=DEV) * 1.5 + 0.5


def run(a, b, r):
    c = torch.empty(a.shape[0], b.shape[0] // 2, dtype=torch.bfloat16, device=DEV)
    module("rmsnorm_swiglu").dispatch(a, permute(b), c, r)
    torch.cuda.synchronize()
    return c


def reference(a, b, r):
    h = gemm_exact(a, b) * r[:, None]                   # r first, as the kernel does
    d = h.shape[1] // 2
    return silu_kernel(h[:, :d]) * h[:, d:]


@pytest.mark.parametrize("kind", ["random", "large"])
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_shapes(M, N, K, kind):
    a, b = inputs(M, N, K, kind)
    r = r_like(M)
    assert_within_bf16_step(run(a, b, r), reference(a, b, r))


def test_zeros():
    a, b = inputs(256, 256, 128, "zeros")
    assert torch.count_nonzero(run(a, b, r_like(256))) == 0


def test_known_answer():
    # b = I and r = 0.5: gate = 0.5 * 1, value = 0.5 * 2, so c = silu(0.5) * 1.
    a = torch.cat([torch.ones(512, 128), 2 * torch.ones(512, 128)], dim=1).to(torch.bfloat16).to(DEV)
    eye = torch.eye(256, dtype=torch.bfloat16, device=DEV)
    r = torch.full((512,), 0.5, device=DEV)
    assert_within_bf16_step(run(a, eye, r), torch.full((512, 128), SILU_HALF, device=DEV))
    with pytest.raises(RuntimeError):
        module("rmsnorm_swiglu").dispatch(permute(eye), a, torch.empty(512, 128, dtype=torch.bfloat16, device=DEV), r)


@pytest.mark.parametrize("M,N,K", BAD_SHAPES)
def test_bad_shapes(M, N, K):
    a, b = inputs(M, N, K, "random")
    with pytest.raises(RuntimeError):
        run(a, b, r_like(M))


def test_rejects_wrong_dtype():
    a, b = inputs(256, 256, 128, "random")
    with pytest.raises(RuntimeError, match="must be torch"):
        run(a, b, r_like(256).bfloat16())
