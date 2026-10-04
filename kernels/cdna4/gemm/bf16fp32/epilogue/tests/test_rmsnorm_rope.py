"""tk_rmsnorm_rope: c = RoPE(r[:, None] * (a @ b.T)), interleaved pairs, natural column order (bf16),
r = [M] fp32. Same store and rule as tk_rope (assert_rope_close). Here the rule's fp32 term matters: x and y
are r-scaled, so x*cos and y*sin carry fp32 rounding that is all that is left when they nearly cancel."""
import pytest
import torch

from conftest import BAD_SHAPES, DEV, SHAPES, gemm_exact, inputs, module
from test_rope import assert_rope_close, cos_sin_table, rope_perm


def r_like(M, seed=6):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.rand(M, generator=g, device=DEV) * 1.5 + 0.5


def run(a, b, r, cos_sin, c=None):
    perm = rope_perm(b.shape[0])
    if c is None:
        c = torch.empty(a.shape[0], b.shape[0], dtype=torch.bfloat16, device=DEV)
    module("rmsnorm_rope").dispatch(a, b[perm].contiguous(), c, r, cos_sin[:, perm].contiguous())
    torch.cuda.synchronize()
    return c


@pytest.mark.parametrize("kind", ["random", "large"])
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_shapes(M, N, K, kind):
    a, b = inputs(M, N, K, kind)
    r, cs = r_like(M), cos_sin_table(M, N)
    assert_rope_close(run(a, b, r, cs), gemm_exact(a, b) * r[:, None], cs)   # r first, as the kernel does


def test_zeros():
    a, b = inputs(256, 256, 128, "zeros")
    assert torch.count_nonzero(run(a, b, r_like(256), cos_sin_table(256, 256))) == 0


def test_known_answer():
    # b = I, r = 0.5, 90-degree rotation: out[2k] = 0.5 * a[2k+1], out[2k+1] = -0.5 * a[2k]. Exact.
    a = ((torch.arange(256, device=DEV) % 7) - 3).float().repeat(512, 1).to(torch.bfloat16)
    eye = torch.eye(256, dtype=torch.bfloat16, device=DEV)
    cs = torch.zeros(512, 256, device=DEV)
    cs[:, 1::2] = 1.0
    out = run(a, eye, torch.full((512,), 0.5, device=DEV), cs.to(torch.bfloat16))
    want = torch.empty_like(a)
    want[:, 0::2], want[:, 1::2] = 0.5 * a[:, 1::2], -0.5 * a[:, 0::2]
    assert torch.equal(out, want)
    with pytest.raises(RuntimeError):
        module("rmsnorm_rope").dispatch(eye, a, torch.empty(512, 256, dtype=torch.bfloat16, device=DEV),
                                        torch.full((512,), 0.5, device=DEV), cs.to(torch.bfloat16))


@pytest.mark.parametrize("M,N,K", BAD_SHAPES)
def test_bad_shapes(M, N, K):
    a, b = inputs(M, N, K, "random")
    with pytest.raises(RuntimeError):
        run(a, b, r_like(M), cos_sin_table(M, N))


def test_rejects_wrong_dtype():
    a, b = inputs(256, 256, 128, "random")
    with pytest.raises(RuntimeError, match="must be torch"):
        run(a, b, r_like(256).bfloat16(), cos_sin_table(256, 256))


def test_rejects_misaligned_output():
    a, b = inputs(256, 256, 128, "random")
    misaligned = torch.empty(256 * 256 + 1, dtype=torch.bfloat16, device=DEV)[1:].view(256, 256)
    with pytest.raises(RuntimeError, match="4-byte alignment"):
        run(a, b, r_like(256), cos_sin_table(256, 256), c=misaligned)
