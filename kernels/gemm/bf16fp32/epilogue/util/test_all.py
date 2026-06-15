#!/usr/bin/env python3
"""test_all.py - the single correctness suite for every GEMM-epilogue kernel.

Covers (each case prints a PASS/FAIL line, so coverage is visible, not just a final tally):
  - base GEMM (tk_noop vs an fp32 reference),
  - every registry epilogue: identity (bit-exact vs noop) + parameter sweep (vs ref(noop_baseline)),
  - the multi-output kernels: partialrms, residual_rms, residual_rms -> aux,
  - the fused residual_rms -> aux -> rmsnorm_scale chain,
  - math invariants: scale linearity, residual additivity, SiLU identity, RMSNorm unit-RMS rows.

Deterministic (seeded). Run from the epilogue dir after building the kernels:
    python3 util/test_all.py
"""
import os, sys, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # tk_*.so live here
import torch
from epilogue_testlib import (EPILOGUES, make_inputs, gemm_base, gemm_reference,
                              init_empty, init_randn, assert_sane, _f32, RTOL, ATOL, DTYPE)
from block_chain import fused_rmsnorm_block, EPS, REG_BLOCK_N
from rope import rope_perm, make_cos_sin, rope_ref

torch.manual_seed(0)
SQ_RTOL, SQ_ATOL = 2e-2, 1.0                 # partials are an fp32 sum over K -> looser than bf16 out
# small / single-block (incl. the K%128 edge 256x512x128), non-square, and larger shapes
SHAPES      = [(256,256,256), (256,512,128), (512,256,256), (768,768,256), (2048,1024,512), (512,1024,1024)]
GEMM_SHAPES = SHAPES + [(2048,2048,2048), (8192,8192,8192)]
CHAIN_SHAPES = [(256,256,256,256), (512,512,512,512), (512,1024,512,768), (768,256,768,512)]  # (M,K0,N,P)
CHAIN_REL = 2e-2                             # normwise ||out-ref||/||ref|| (robust for the two-GEMM chain)
ROPE_SHAPES = [(256,256,256), (512,512,128), (768,256,256), (1024,512,384), (2048,1024,512), (4096,2048,512), (8192,4096,512)]  # (M,N,K); N%256, K%128


def _p(tag, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {tag:<40} {detail}")
    return bool(ok)


def _rms_io(m, n):
    """Output buffers shared by the residual-RMS kernels: c (gamma-scaled output), save (h1), and
    partials (per-(group, row) sum-of-squares, shape [N/REG_BLOCK_N, M])."""
    c = init_empty((m, n))
    save = init_empty((m, n))
    partials = torch.zeros((n // REG_BLOCK_N, m), dtype=torch.float32, device="cuda")
    return c, save, partials


def test_base_gemm(noop):
    ok = True
    for (m, n, k) in GEMM_SHAPES:
        A, Bt = make_inputs(m, n, k)
        C = init_empty((m, n))
        noop.dispatch(A, Bt, C)
        torch.cuda.synchronize()
        assert_sane("C", C)
        ref = gemm_reference(A, Bt)
        e = (C.float() - ref).abs().max().item()
        ok &= _p(f"gemm {(m,n,k)}", torch.allclose(C.float(), ref, rtol=RTOL, atol=ATOL), f"max_err={e:.3g}")
    return ok


def test_registry(noop):
    ok = True
    for name, spec in EPILOGUES.items():
        if "ref" not in spec:          # dim-changing / multi-output kernels have dedicated tests
            continue
        fk = importlib.import_module(spec["module"])
        for (m, n, k) in SHAPES:
            A, Bt = make_inputs(m, n, k)
            Cn = gemm_base(noop, A, Bt, m, n)                   # bf16 no-op baseline (D)
            idf = spec["identity"]
            if idf is not None:                                # identity args -> output must equal noop, bit for bit
                args = idf(m, n, k)
                O = init_empty((m, n))
                fk.dispatch(A, Bt, O, *args)
                torch.cuda.synchronize()
                ok &= _p(f"{name} identity {(m,n,k)}", torch.equal(O, Cn))
            for args in spec["sweep"](m, n, k):                # value sweep -> output must match ref(noop_baseline)
                O = init_empty((m, n))
                fk.dispatch(A, Bt, O, *args)
                torch.cuda.synchronize()
                ref = init_empty((m, n))
                spec["ref"](Cn, ref, *args)
                fin = bool(torch.isfinite(O).all())
                e = (O.float() - ref.float()).abs().max().item()
                ok &= _p(f"{name} {spec['label'](args)} {(m,n,k)}",
                         fin and torch.allclose(O.float(), ref.float(), rtol=RTOL, atol=ATOL),
                         f"max_err={e:.3g}" + ("" if fin else " NON-FINITE"))
    return ok

def test_swiglu():
    import swiglu as sg
    ok = True
    for (m, d_ff, k) in [(256,128,256),(512,256,256),(256,512,128),(768,512,256),(2048,1024,512)]:
        X = init_randn((m, k))
        W = init_randn((k, 2*d_ff))
        out = sg.make_swiglu(W)(X)
        fin = bool(torch.isfinite(out).all())
        ref = sg.swiglu_ref(X, W)   # fp32 oracle: kernel accumulates AND applies the epilogue in fp32
        e = (out.float() - ref).abs().max().item()
        ok &= _p(f"swiglu {(m,d_ff,k)}",
                 fin and torch.allclose(out.float(), ref, rtol=RTOL, atol=ATOL),
                 f"max_err={e:.3g}" + ("" if fin else " NON-FINITE"))
    X = init_randn((256, 256)); W = init_randn((256, 2*256))
    out = sg.make_swiglu(W)(X).float()
    Hh = X.float() @ W.float()
    inv = torch.nn.functional.silu(Hh[:, :256]) * Hh[:, 256:]
    ok &= _p("invariant swiglu==silu(gate)*value", torch.allclose(out, inv, rtol=2e-2, atol=1e-1))
    return ok


def test_partialrms(noop, prms):
    ok = True
    for (m, n, k) in SHAPES:
        A, Bt = make_inputs(m, n, k)
        D = init_empty((m, n))
        noop.dispatch(A, Bt, D)                                # the SAME D the kernel squares
        torch.cuda.synchronize()
        c = init_empty((m, n))
        partials = torch.zeros((n // REG_BLOCK_N, m), dtype=torch.float32, device="cuda")
        prms.dispatch(A, Bt, c, partials)
        torch.cuda.synchronize()
        got = partials.sum(0)
        ref = D.float().pow(2).sum(-1)
        okc = bool(torch.isfinite(got).all() and got.abs().max() > 0
                   and torch.allclose(got, ref, rtol=SQ_RTOL, atol=SQ_ATOL))
        ok &= _p(f"partialrms {(m,n,k)}", okc)
    return ok


def test_residual_rms(rr):
    ok = True
    for (m, n, k) in SHAPES:
        A, Bt = make_inputs(m, n, k)
        residual = init_randn((m, n))
        gamma = init_randn((n,))
        h1 = gemm_reference(A, Bt) + residual.float()
        c, save, partials = _rms_io(m, n)
        rr.dispatch(A, Bt, c, residual, gamma, partials, save)
        torch.cuda.synchronize()
        s = torch.allclose(save.float(), h1, rtol=RTOL, atol=ATOL)
        o = torch.allclose(c.float(), h1 * gamma.float(), rtol=RTOL, atol=ATOL)
        q = torch.allclose(partials.sum(0), h1.pow(2).sum(-1), rtol=SQ_RTOL, atol=SQ_ATOL)
        ok &= _p(f"residual_rms {(m,n,k)}", s and o and q, f"save={s} out={o} partials={q}")
    return ok


def test_residual_rms_aux(rr, aux):
    ok = True
    for (m, n, k) in SHAPES:
        A, Bt = make_inputs(m, n, k)
        residual = init_randn((m, n))
        gamma = init_randn((n,))
        h1 = gemm_reference(A, Bt) + residual.float()
        c, save, partials = _rms_io(m, n)
        rr.dispatch(A, Bt, c, residual, gamma, partials, save)
        r = torch.empty(m, dtype=DTYPE, device="cuda")
        aux.reduce(partials, r)
        torch.cuda.synchronize()
        ref = torch.rsqrt(h1.pow(2).mean(-1) + EPS)
        ok &= _p(f"residual_rms->aux {(m,n,k)}", torch.allclose(r.float(), ref, rtol=SQ_RTOL, atol=1e-3))
    return ok


def test_aux_reduce(aux):
    """rms_reduce in isolation: synthetic per-(group,row) partials -> r = rsqrt(sum_groups/N + eps).
    Independent of residual_rms (whose partials feed the composition test above), so a bug in the
    cross-group sum or the rsqrt can't be masked by a compensating error upstream."""
    ok = True
    for (M, groups) in [(256, 4), (512, 16), (768, 2), (1024, 8)]:
        N = groups * REG_BLOCK_N                               # kernel derives N = partials.rows() * REG_BLOCK_N
        partials = (torch.rand((groups, M), dtype=torch.float32, device="cuda") + 0.05) * 50.0   # >0, varied
        r = torch.empty(M, dtype=DTYPE, device="cuda")
        aux.reduce(partials, r)
        torch.cuda.synchronize()
        ref = torch.rsqrt(partials.sum(0) / N + EPS)
        e = (r.float() - ref).abs().max().item()
        ok &= _p(f"aux_reduce M={M} groups={groups}",
                 torch.allclose(r.float(), ref, rtol=SQ_RTOL, atol=1e-2), f"max_err={e:.3g}")
    return ok


def test_chain():
    ok = True
    for (M, K0, N, P) in CHAIN_SHAPES:
        X = init_randn((M, K0))
        W0 = init_randn((K0, N))
        residual = init_randn((M, N))
        gamma = init_randn((N,))
        W1 = init_randn((N, P))
        out = fused_rmsnorm_block(X, W0, residual, gamma, W1)
        h1 = X.float() @ W0.float() + residual.float()
        hn = (h1 * torch.rsqrt(h1.pow(2).mean(-1, keepdim=True) + EPS)) * gamma.float()
        ref = hn @ W1.float()
        rel = (out.float() - ref).norm().item() / ref.norm().item()
        ok &= _p(f"chain {(M,K0,N,P)}", rel < CHAIN_REL, f"rel={rel:.2e}")
    return ok


def test_invariants(noop, scale_m, rms_m, resadd_m, silu_m):
    """Properties that hold for ANY input -> catch bug classes fixed cases miss."""
    ok = True
    m, n, k = 512, 1024, 256
    A, Bt = make_inputs(m, n, k)
    D = gemm_base(noop, A, Bt, m, n).float()

    # scale is linear in alpha: f(2a) == 2 f(a)
    O1 = init_empty((m, n))
    scale_m.dispatch(A, Bt, O1, _f32(1.0))
    O2 = init_empty((m, n))
    scale_m.dispatch(A, Bt, O2, _f32(2.0))
    torch.cuda.synchronize()
    ok &= _p("invariant scale linearity", torch.allclose(O2.float(), 2 * O1.float(), rtol=2e-2, atol=1e-1))

    # residual add is additive: out == D + residual
    res = init_randn((m, n))
    Or = init_empty((m, n))
    resadd_m.dispatch(A, Bt, Or, res)
    torch.cuda.synchronize()
    ok &= _p("invariant residual additivity", torch.allclose(Or.float(), D + res.float(), rtol=2e-2, atol=2e-1))

    # silu(x) == x * sigmoid(x)
    Os = init_empty((m, n))
    silu_m.dispatch(A, Bt, Os)
    torch.cuda.synchronize()
    ok &= _p("invariant silu==x*sigmoid(x)", torch.allclose(Os.float(), D * torch.sigmoid(D), rtol=2e-2, atol=1e-1))

    # rmsnorm with r=1/rms(D), gamma=1 -> every output row has ~unit RMS
    r = torch.rsqrt(D.pow(2).mean(-1) + EPS).to(DTYPE)
    g1 = torch.ones(n, dtype=DTYPE, device="cuda")
    Orm = init_empty((m, n))
    rms_m.dispatch(A, Bt, Orm, r, g1)
    torch.cuda.synchronize()
    row_rms = Orm.float().pow(2).mean(-1).sqrt()
    ok &= _p("invariant rmsnorm unit-RMS rows", torch.allclose(row_rms, torch.ones(m, device="cuda"), rtol=5e-2, atol=5e-2))
    return ok


def test_rope(rope_m):
    """gemm_rope, interleaved: permute B + cos_sin with rope_perm, rotate register-local, store
    permuted. Validate against the permuted reference rope(A@B, cos_sin)[:, perm]."""
    ok = True
    for (m, n, k) in ROPE_SHAPES:
        A, Bt = make_inputs(m, n, k)
        perm = rope_perm(n).to(A.device)
        D = gemm_reference(A, Bt)                                   # fp32 A@B [m,n]
        cs = make_cos_sin(m, n)                                     # natural interleaved [cos,sin]
        Bp = Bt[perm].contiguous(); csp = cs[:, perm].to(DTYPE).contiguous()   # held past async launch
        O = init_empty((m, n))
        rope_m.dispatch(A, Bp, O, csp); torch.cuda.synchronize()
        fin = bool(torch.isfinite(O).all())
        ref = rope_ref(D, cs.to(DTYPE))[:, perm]   # bf16 cos_sin to match the kernel's bf16 cos_sin input
        e = (O.float() - ref).abs().max().item()
        ok &= _p(f"rope {(m,n,k)}", fin and torch.allclose(O.float(), ref, rtol=RTOL, atol=ATOL),
                 f"max_err={e:.3g}" + ("" if fin else " NON-FINITE"))
        # identity: cos=1, sin=0 -> rotation is a no-op, output == D permuted (validates the plumbing)
        cs1 = torch.zeros(m, n, device="cuda"); cs1[:, 0::2] = 1.0
        csp1 = cs1[:, perm].to(DTYPE).contiguous(); Oi = init_empty((m, n))
        rope_m.dispatch(A, Bp, Oi, csp1); torch.cuda.synchronize()
        ok &= _p(f"rope identity {(m,n,k)}", torch.allclose(Oi.float(), D[:, perm], rtol=RTOL, atol=ATOL))
    return ok


def main():
    noop     = importlib.import_module("tk_noop")
    scale_m  = importlib.import_module("tk_scale")
    rms_m    = importlib.import_module("tk_rmsnorm_scale")
    resadd_m = importlib.import_module("tk_residual_add")
    silu_m   = importlib.import_module("tk_silu")
    prms     = importlib.import_module("tk_partialrms")
    rr       = importlib.import_module("tk_residual_rms")
    aux      = importlib.import_module("tk_aux_rms")
    rope_m   = importlib.import_module("tk_rope")

    # (section label, test fn, args) -- run in order, AND the pass flags together
    suite = [
        ("[base GEMM]",           test_base_gemm,        (noop,)),
        ("[registry epilogues]",  test_registry,         (noop,)),
        ("[swiglu]",              test_swiglu,           ()),
        ("[partialrms]",          test_partialrms,       (noop, prms)),
        ("[residual_rms]",        test_residual_rms,     (rr,)),
        ("[residual_rms -> aux]", test_residual_rms_aux, (rr, aux)),
        ("[aux_reduce]",          test_aux_reduce,       (aux,)),
        ("[chain]",               test_chain,            ()),
        ("[rope]",                test_rope,             (rope_m,)),
        ("[invariants]",          test_invariants,       (noop, scale_m, rms_m, resadd_m, silu_m)),
    ]
    allpass = True
    for label, fn, fargs in suite:
        print(label)
        allpass &= fn(*fargs)
    print("ALL PASSED" if allpass else "SOME FAILED")
    return allpass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
