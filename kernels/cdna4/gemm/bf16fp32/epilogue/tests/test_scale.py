"""tk_scale: c = alpha * (a @ b.T) (bf16), alpha a 1-element fp32 tensor. alpha * x is one IEEE fp32
multiply on both sides, so c must be bit-identical to the reference stored the way the kernel
stores bf16 (to_bf16_chopped)."""
import pytest
import torch

from conftest import BAD_SHAPES, DEV, SHAPES, assert_bitexact, gemm_exact, inputs, module


def alpha_tensor(value):
    return torch.full((1,), value, dtype=torch.float32, device=DEV)


def run(a, b, alpha):
    c = torch.empty(a.shape[0], b.shape[0], dtype=torch.bfloat16, device=DEV)
    module("scale").dispatch(a, b, c, alpha)
    torch.cuda.synchronize()
    return c


def reference(a, b, alpha):
    return gemm_exact(a, b) * alpha[0]          # fp32 * fp32


@pytest.mark.parametrize("kind", ["random", "large"])
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_shapes(M, N, K, kind):
    a, b = inputs(M, N, K, kind)
    alpha = alpha_tensor(0.37)
    assert_bitexact(run(a, b, alpha), reference(a, b, alpha))


def test_zeros():
    a, b = inputs(256, 256, 128, "zeros")
    assert torch.count_nonzero(run(a, b, alpha_tensor(0.37))) == 0


def test_known_answer():
    a = torch.ones(512, 256, dtype=torch.bfloat16, device=DEV)
    eye = torch.eye(256, dtype=torch.bfloat16, device=DEV)
    c = run(a, eye, alpha_tensor(0.5))
    assert torch.equal(c, torch.full_like(c, 0.5))            # 0.5, not 2.0: alpha multiplies
    with pytest.raises(RuntimeError):
        module("scale").dispatch(eye, a, torch.empty(512, 256, dtype=torch.bfloat16, device=DEV),
                                 alpha_tensor(0.5))


@pytest.mark.parametrize("M,N,K", BAD_SHAPES)
def test_bad_shapes(M, N, K):
    a, b = inputs(M, N, K, "random")
    with pytest.raises(RuntimeError):
        run(a, b, alpha_tensor(0.37))


@pytest.mark.xfail(strict=True, reason="bindings check dtypes from Phase 2 task 7")
def test_rejects_wrong_dtype():
    a, b = inputs(256, 256, 128, "random")
    with pytest.raises(RuntimeError, match="must be torch"):
        run(a, b, alpha_tensor(0.37).bfloat16())
