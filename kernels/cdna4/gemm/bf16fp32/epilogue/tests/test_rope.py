"""tk_rope: c = RoPE(a @ b.T) with interleaved pairs (2k, 2k+1), natural column order (bf16). The caller
permutes b's rows and cos_sin with rope_perm. The output store rounds to nearest (packed
v_cvt_pk_bf16_f32). Allowed difference (assert_rope_close): one bf16 step of the result plus the fp32
rounding of the rotation's two products and sum, 4 * 2^-24 * (|x||cos| + |y||sin|), which only matters
when the two products nearly cancel."""
import math

import pytest
import torch

from conftest import BAD_SHAPES, DEV, SHAPES, bf16_step, gemm_exact, inputs, module


def rope_perm(N):
    """perm[new_col] = old_col: pair k's even column 2k -> (k/128)*256 + k%128, odd column 2k+1 -> +128."""
    k = torch.arange(N // 2, device=DEV)
    slot = (k // 128) * 256 + k % 128
    perm = torch.empty(N, dtype=torch.long, device=DEV)
    perm[slot] = 2 * k
    perm[slot + 128] = 2 * k + 1
    return perm


def cos_sin_table(M, N, base=10000.0):
    """bf16 [M, N] natural interleaved table: [m, 2k] = cos(m * theta_k), [m, 2k+1] = sin(m * theta_k)."""
    k = torch.arange(N // 2, device=DEV)
    angle = torch.arange(M, device=DEV).float()[:, None] * base ** (-2.0 * k / N)[None, :]
    table = torch.empty(M, N, device=DEV)
    table[:, 0::2], table[:, 1::2] = torch.cos(angle), torch.sin(angle)
    return table.to(torch.bfloat16)


def rotate(h, cos_sin):
    """Interleaved RoPE in fp32 on natural columns, in the kernel's operation order."""
    cs = cos_sin.float()
    x, y, cos, sin = h[:, 0::2], h[:, 1::2], cs[:, 0::2], cs[:, 1::2]
    out = torch.empty_like(h)
    out[:, 0::2] = x * cos + y * sin
    out[:, 1::2] = y * cos - x * sin
    return out


def assert_rope_close(got, h, cos_sin):
    """got (bf16) vs rotate(h, cos_sin) rounded to nearest: <= 1 bf16 step + 4 * 2^-24 * magnitude, where
    magnitude is |x||cos| + |y||sin| for even outputs and |y||cos| + |x||sin| for odd outputs."""
    cs = cos_sin.float()
    x, y, cos, sin = h[:, 0::2].abs(), h[:, 1::2].abs(), cs[:, 0::2].abs(), cs[:, 1::2].abs()
    magnitude = torch.empty_like(h)
    magnitude[:, 0::2], magnitude[:, 1::2] = x * cos + y * sin, y * cos + x * sin
    want = rotate(h, cos_sin).to(torch.bfloat16).float()
    err = (got.float() - want).abs()
    bad = err > bf16_step(want) + 4 * 2.0 ** -24 * magnitude
    if bad.any():
        i = tuple(bad.nonzero()[0].tolist())
        raise AssertionError(f"{int(bad.sum())} of {bad.numel()} elements out of bound; first at {i}: got "
                             f"{got[i].item()!r}, want {want[i].item()!r}, magnitude {magnitude[i].item()!r}")


def run(a, b, cos_sin, c=None):
    perm = rope_perm(b.shape[0])
    if c is None:
        c = torch.empty(a.shape[0], b.shape[0], dtype=torch.bfloat16, device=DEV)
    module("rope").dispatch(a, b[perm].contiguous(), c, cos_sin[:, perm].contiguous())
    torch.cuda.synchronize()
    return c


@pytest.mark.parametrize("kind", ["random", "large"])
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_shapes(M, N, K, kind):
    a, b = inputs(M, N, K, kind)
    cs = cos_sin_table(M, N)
    assert_rope_close(run(a, b, cs), gemm_exact(a, b), cs)


def test_zeros():
    a, b = inputs(256, 256, 128, "zeros")
    assert torch.count_nonzero(run(a, b, cos_sin_table(256, 256))) == 0


def test_known_answer():
    # b = I and a 90-degree rotation (cos 0, sin 1): out[2k] = a[2k+1], out[2k+1] = -a[2k]. Exact.
    a = ((torch.arange(256, device=DEV) % 7) - 3).float().repeat(512, 1).to(torch.bfloat16)
    eye = torch.eye(256, dtype=torch.bfloat16, device=DEV)
    cs = torch.zeros(512, 256, device=DEV)
    cs[:, 1::2] = 1.0
    out = run(a, eye, cs.to(torch.bfloat16))
    want = torch.empty_like(a)
    want[:, 0::2], want[:, 1::2] = a[:, 1::2], -a[:, 0::2]
    assert torch.equal(out, want)
    with pytest.raises(RuntimeError):
        module("rope").dispatch(eye, a, torch.empty(512, 256, dtype=torch.bfloat16, device=DEV), cs.to(torch.bfloat16))


@pytest.mark.parametrize("M,N,K", BAD_SHAPES)
def test_bad_shapes(M, N, K):
    a, b = inputs(M, N, K, "random")
    with pytest.raises(RuntimeError):
        run(a, b, cos_sin_table(M, N))


def test_rejects_wrong_dtype():
    a, b = inputs(256, 256, 128, "random")
    with pytest.raises(RuntimeError, match="must be torch"):
        run(a, b, cos_sin_table(256, 256).float())


def test_rejects_misaligned_output():
    a, b = inputs(256, 256, 128, "random")
    misaligned = torch.empty(256 * 256 + 1, dtype=torch.bfloat16, device=DEV)[1:].view(256, 256)
    with pytest.raises(RuntimeError, match="4-byte alignment"):
        run(a, b, cos_sin_table(256, 256), c=misaligned)
