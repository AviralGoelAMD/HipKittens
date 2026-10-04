"""hk: the API's own behavior: preparation, refusals, stream and graph safety, inv_rms."""
import pytest
import torch

import hk
from conftest import DEV, assert_bitexact, gemm_exact, inputs, module, pow2_gamma, weight
from test_rope import rope_perm
from test_swiglu import permute as swiglu_permute

MATMUL_EPILOGUES = {"noop", "silu", "scale", "residual_add", "rmsnorm_scale", "swiglu", "rmsnorm_swiglu",
                    "rope", "rmsnorm_rope"}


def test_available():
    av = hk.available()
    assert set(av) == MATMUL_EPILOGUES
    assert av["rmsnorm_scale"] == ["r", "gamma"] and av["rmsnorm_rope"] == ["r", "cos_sin"] and av["noop"] == []


def test_prepare_matches_hand_layout():
    _, b = inputs(256, 512, 256, "random")                       # test operand b [N, K]
    gamma = pow2_gamma(256)
    folded = (b.float() * gamma.float()[None, :]).to(torch.bfloat16)
    assert torch.equal(weight(b).tensor, b)
    assert torch.equal(weight(b, layout="swiglu", gamma=gamma).tensor, swiglu_permute(folded))
    assert torch.equal(weight(b, layout="rope").tensor, b[rope_perm(512)])
    w = weight(b, layout="rope", gamma=gamma)
    assert w.layout == "rope" and w.gamma_folded


def test_layout_mismatch():
    a, b = inputs(256, 256, 128, "random")
    with pytest.raises(ValueError, match="'swiglu' weight"):
        hk.matmul(a, weight(b, layout="rope"), epilogue="swiglu")


def test_gamma_flag_mismatch():
    a, b = inputs(256, 256, 128, "random")
    r = torch.ones(256, device=DEV)
    with pytest.raises(ValueError, match="no gamma folded"):
        hk.matmul(a, weight(b, gamma=pow2_gamma(128)), epilogue="rmsnorm_scale", r=r,
                  gamma=torch.ones(256, dtype=torch.bfloat16, device=DEV))
    with pytest.raises(ValueError, match="gamma folded in"):
        hk.matmul(a, weight(b, layout="swiglu"), epilogue="rmsnorm_swiglu", r=r)


def test_argument_names():
    a, b = inputs(256, 256, 128, "random")
    w = weight(b)
    with pytest.raises(ValueError, match="unknown epilogue"):
        hk.matmul(a, w, epilogue="gelu")
    with pytest.raises(TypeError, match="takes arguments"):          # missing
        hk.matmul(a, w, epilogue="scale")
    with pytest.raises(TypeError, match="takes arguments"):          # extra
        hk.matmul(a, w, epilogue="silu", alpha=1.0)
    with pytest.raises(TypeError, match="takes arguments"):          # misspelled
        hk.matmul(a, w, epilogue="scale", alhpa=1.0)


def test_strict_inputs():
    a, b = inputs(256, 256, 128, "random")
    w = weight(b)
    for bad in (a.float(), a.t().contiguous().t(), a.cpu()):          # fp32, non-contiguous, CPU
        with pytest.raises(ValueError, match="must be"):
            hk.matmul(bad, w)
    with pytest.raises(TypeError, match="hk.prepare"):
        hk.matmul(a, b)
    with pytest.raises(ValueError, match="must be"):
        hk.prepare(b.float())
    with pytest.raises(TypeError, match="alpha must be a torch.Tensor"):  # bool is not a number here
        hk.matmul(a, w, epilogue="scale", alpha=True)


def test_binding_still_checks_dtype():
    a, b = inputs(256, 256, 128, "random")
    c = torch.empty(256, 256, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(RuntimeError, match="must be torch"):
        module("noop").dispatch(a.float(), b, c, torch.cuda.current_stream().cuda_stream)


def test_scalar_alpha_number_or_tensor():
    a, b = inputs(256, 256, 128, "random")
    module("scale")
    w = weight(b)
    assert torch.equal(hk.matmul(a, w, epilogue="scale", alpha=0.5),
                       hk.matmul(a, w, epilogue="scale", alpha=torch.full((1,), 0.5, device=DEV)))


def test_output_shapes():
    a, b = inputs(256, 512, 128, "random")
    module("noop"), module("swiglu")
    assert hk.matmul(a, weight(b)).shape == (256, 512)
    assert hk.matmul(a, weight(b, layout="swiglu"), epilogue="swiglu").shape == (256, 256)


def test_side_stream():
    """Inputs produced on stream s, matmul on s with no sync in between: the kernel must run on s."""
    module("noop")
    a0, b = inputs(4096, 4096, 4096, "random")
    w = weight(b)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        a = a0 * 1                                               # written on s
        out = hk.matmul(a, w)
    s.synchronize()
    assert_bitexact(out, gemm_exact(a0, b))


def test_graph_capture():
    """hk.matmul is captured into a graph and replays correctly on new inputs."""
    module("noop")
    a1, b = inputs(256, 256, 128, "random", seed=1)
    a2, _ = inputs(256, 256, 128, "random", seed=2)
    w = weight(b)
    static_a = a1.clone()
    hk.matmul(static_a, w)                                       # warm-up (loads the module's code object)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = hk.matmul(static_a, w)
    static_a.copy_(a2)
    graph.replay()
    torch.cuda.synchronize()
    assert_bitexact(static_out, gemm_exact(a2, b))


def test_inv_rms():
    module("partialrms"), module("rms_reduce")
    a, b = inputs(512, 1024, 256, "random")
    h = a.double() @ b.double().T
    want = torch.rsqrt(h.square().mean(1) + 1e-5)
    got = hk.inv_rms(a, weight(b)).double()
    assert ((got - want).abs() <= (64 + 16 + 4) * 2.0 ** -24 * want).all()
    with pytest.raises(ValueError):
        hk.inv_rms(a, weight(b, layout="rope"))
