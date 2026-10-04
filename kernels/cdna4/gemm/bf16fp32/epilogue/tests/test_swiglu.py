"""tk_swiglu: c = silu(gate) * value, [M, N] -> [M, N/2] (bf16). The caller permutes b's rows
(epilogues/swiglu.cuh); the reference uses the natural layout. Fast SiLU allows one bf16 step."""
import pytest
import torch

from conftest import BAD_SHAPES, DEV, SHAPES, assert_within_bf16_step, gemm_exact, inputs, module, silu_kernel

SILU_1 = 0.7310585786300049   # silu(1)
SILU_2 = 1.7615941559557649   # silu(2)


def permute(b):
    """Rows of b are W_gate_up's columns in natural order ([0, d) gate, [d, 2d) value). Gate j moves to
    row (j/128)*256 + j%128 and value j to 128 rows later, as epilogues/swiglu.cuh requires."""
    d = b.shape[0] // 2
    j = torch.arange(d, device=b.device)
    slot = (j // 128) * 256 + j % 128
    perm = torch.empty(2 * d, dtype=torch.long, device=b.device)
    perm[slot] = j
    perm[slot + 128] = d + j
    return b[perm].contiguous()


def run(a, b):
    c = torch.empty(a.shape[0], b.shape[0] // 2, dtype=torch.bfloat16, device=DEV)
    module("swiglu").dispatch(a, permute(b), c)
    torch.cuda.synchronize()
    return c


def reference(a, b):
    h = gemm_exact(a, b)
    d = h.shape[1] // 2
    return silu_kernel(h[:, :d]) * h[:, d:]


@pytest.mark.parametrize("kind", ["random", "large"])
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_shapes(M, N, K, kind):
    a, b = inputs(M, N, K, kind)
    assert_within_bf16_step(run(a, b), reference(a, b))


def test_zeros():
    a, b = inputs(256, 256, 128, "zeros")
    assert torch.count_nonzero(run(a, b)) == 0


def test_known_answer():
    # b = I, so the GEMM returns a: gate columns hold 1, value columns hold 2.
    a = torch.cat([torch.ones(512, 128), 2 * torch.ones(512, 128)], dim=1).to(torch.bfloat16).to(DEV)
    eye = torch.eye(256, dtype=torch.bfloat16, device=DEV)
    out = run(a, eye)
    assert_within_bf16_step(out, torch.full((512, 128), SILU_1 * 2, device=DEV))   # not silu(2) * 1
    assert not torch.allclose(out.float(), torch.full_like(out.float(), SILU_2))
    with pytest.raises(RuntimeError):
        module("swiglu").dispatch(permute(eye), a, torch.empty(512, 128, dtype=torch.bfloat16, device=DEV))


@pytest.mark.parametrize("M,N,K", BAD_SHAPES)
def test_bad_shapes(M, N, K):
    a, b = inputs(M, N, K, "random")
    with pytest.raises(RuntimeError):
        run(a, b)


@pytest.mark.xfail(strict=True, reason="bindings check dtypes from Phase 2 task 7")
def test_rejects_wrong_dtype():
    a, b = inputs(256, 256, 128, "random")
    c = torch.empty(256, 128, dtype=torch.float32, device=DEV)
    with pytest.raises(RuntimeError, match="must be torch"):
        module("swiglu").dispatch(a, permute(b), c)
