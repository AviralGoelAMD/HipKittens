"""tk_rmsnorm_scale: c = (a @ b.T) * r[:, None] * gamma[None, :] (bf16), r = [M] fp32, gamma = [N] bf16.
The kernel multiplies by r, then by gamma, each one IEEE fp32 multiply, so c must be bit-identical to the
reference stored the way the kernel stores bf16 (to_bf16_chopped)."""
import pytest
import torch

from conftest import BAD_SHAPES, DEV, SHAPES, assert_bitexact, gemm_exact, inputs, module


def vectors(M, N, seed=2):
    """fp32 r in [0.5, 2) (like 1/rms) and bf16 gamma in [0.25, 1.75)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = torch.rand(M, generator=g, device=DEV) * 1.5 + 0.5
    gamma = (torch.rand(N, generator=g, device=DEV) * 1.5 + 0.25).to(torch.bfloat16)
    return r, gamma


def run(a, b, r, gamma):
    c = torch.empty(a.shape[0], b.shape[0], dtype=torch.bfloat16, device=DEV)
    module("rmsnorm_scale").dispatch(a, b, c, r, gamma)
    torch.cuda.synchronize()
    return c


def reference(a, b, r, gamma):
    return gemm_exact(a, b) * r[:, None] * gamma.float()[None, :]   # r first, then gamma


@pytest.mark.parametrize("kind", ["random", "large"])
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_shapes(M, N, K, kind):
    a, b = inputs(M, N, K, kind)
    r, gamma = vectors(M, N)
    assert_bitexact(run(a, b, r, gamma), reference(a, b, r, gamma))


def test_zeros():
    a, b = inputs(256, 256, 128, "zeros")
    assert torch.count_nonzero(run(a, b, *vectors(256, 256))) == 0


def test_known_answer():
    # b = I, so the GEMM returns a = 1; then c[m, n] = r[m] * gamma[n].
    a = torch.ones(512, 256, dtype=torch.bfloat16, device=DEV)
    eye = torch.eye(256, dtype=torch.bfloat16, device=DEV)
    r = torch.tensor([1.0, 2.0], device=DEV).repeat(256)                           # alternates per row
    gamma = torch.tensor([4.0, 8.0], device=DEV).repeat(128).to(torch.bfloat16)    # alternates per column
    c = run(a, eye, r, gamma)
    assert torch.equal(c, (r[:, None] * gamma.float()[None, :]).to(torch.bfloat16))
    with pytest.raises(RuntimeError):                    # r and gamma swapped: wrong dtypes and shapes
        module("rmsnorm_scale").dispatch(a, eye, torch.empty_like(c), gamma, r)


@pytest.mark.parametrize("M,N,K", BAD_SHAPES)
def test_bad_shapes(M, N, K):
    a, b = inputs(M, N, K, "random")
    with pytest.raises(RuntimeError):
        run(a, b, *vectors(M, N))


def test_rejects_wrong_dtype():
    a, b = inputs(256, 256, 128, "random")
    r, gamma = vectors(256, 256)
    with pytest.raises(RuntimeError, match="must be torch"):
        run(a, b, r.bfloat16(), gamma)
