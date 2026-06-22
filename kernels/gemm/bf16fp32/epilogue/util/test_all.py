#!/usr/bin/env python3
"""test_all.py - the single correctness suite for every GEMM-epilogue kernel.

Covers (each case prints a PASS/FAIL line, so coverage is visible, not just a final tally):
  - base GEMM (tk_noop vs an fp32 reference),
  - every registry epilogue: identity (bit-exact vs noop) + parameter sweep (vs ref(noop_baseline)),
  - the multi-output kernels: partialrms, residual_rms, residual_rms -> aux,
  - the fused residual_rms -> aux -> rmsnorm_scale chain,
  - the dim-reducing SwiGLU kernel (silu(gate)*value) and the RoPE epilogue (interleaved rotation),
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
SHAPES      = [(256,256,256), (256,512,128), (512,256,256), (768,768,256), (2048,1024,512), (512,1024,1024),
               (4096,4096,4096), (4096,11008,4096), (4096,4096,11008)]  # + Llama-7B: attn proj, FFN gate/up, FFN down
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
    # large-shape coverage for residual_add (the deferred (8192,8192,4096) correctness concern)
    fk = importlib.import_module("tk_residual_add"); spec = EPILOGUES["residual_add"]
    for (m, n, k) in [(8192, 8192, 4096)]:
        A, Bt = make_inputs(m, n, k)
        Cn = gemm_base(noop, A, Bt, m, n)
        args = spec["sweep"](m, n, k)[0]
        O = init_empty((m, n)); fk.dispatch(A, Bt, O, *args); torch.cuda.synchronize()
        ref = init_empty((m, n)); spec["ref"](Cn, ref, *args)
        fin = bool(torch.isfinite(O).all())
        e = (O.float() - ref.float()).abs().max().item()
        ok &= _p(f"residual_add large {(m,n,k)}",
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
    ok &= _p("invariant swiglu==silu(gate)*value", torch.allclose(out, inv, rtol=2e-2, atol=1e-1))  # fp32-recomputed invariant -> looser than bf16 RTOL/ATOL
    return ok


def test_rmsnorm_swiglu():
    import swiglu as sg
    ok = True
    for (m, d_ff, k) in [(256,128,256),(512,256,256),(256,512,128),(768,512,256),(2048,1024,512)]:
        X = init_randn((m, k)); W = init_randn((k, 2*d_ff)); gamma = init_randn((k,))
        r = torch.rsqrt(X.float().pow(2).mean(-1) + EPS).to(DTYPE)   # precomputed per-row inv-rms (bf16)
        out = sg.make_rmsnorm_swiglu(W, gamma)(X, r)
        fin = bool(torch.isfinite(out).all())
        ref = sg.rmsnorm_swiglu_ref(X, gamma, W, r)
        e = (out.float() - ref).abs().max().item()
        ok &= _p(f"rmsnorm_swiglu {(m,d_ff,k)}",
                 fin and torch.allclose(out.float(), ref, rtol=RTOL, atol=ATOL),
                 f"max_err={e:.3g}" + ("" if fin else " NON-FINITE"))
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
    ok &= _p("invariant scale linearity", torch.allclose(O2.float(), 2 * O1.float(), rtol=2e-2, atol=1e-1))  # fp32-recomputed invariant -> looser than bf16 RTOL/ATOL

    # residual add is additive: out == D + residual
    res = init_randn((m, n))
    Or = init_empty((m, n))
    resadd_m.dispatch(A, Bt, Or, res)
    torch.cuda.synchronize()
    ok &= _p("invariant residual additivity", torch.allclose(Or.float(), D + res.float(), rtol=2e-2, atol=2e-1))  # additive at magnitude -> atol 2e-1 (fp32 recompute)

    # silu(x) == x * sigmoid(x)
    Os = init_empty((m, n))
    silu_m.dispatch(A, Bt, Os)
    torch.cuda.synchronize()
    ok &= _p("invariant silu==x*sigmoid(x)", torch.allclose(Os.float(), D * torch.sigmoid(D), rtol=2e-2, atol=1e-1))  # fp32-recomputed invariant -> looser than bf16 RTOL/ATOL

    # rmsnorm with r=1/rms(D), gamma=1 -> every output row has ~unit RMS
    r = torch.rsqrt(D.pow(2).mean(-1) + EPS).to(DTYPE)
    g1 = torch.ones(n, dtype=DTYPE, device="cuda")
    Orm = init_empty((m, n))
    rms_m.dispatch(A, Bt, Orm, r, g1)
    torch.cuda.synchronize()
    row_rms = Orm.float().pow(2).mean(-1).sqrt()
    ok &= _p("invariant rmsnorm unit-RMS rows", torch.allclose(row_rms, torch.ones(m, device="cuda"), rtol=5e-2, atol=5e-2))  # row RMS within 5% (fp32 recompute)
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


def test_rmsnorm_rope():
    import rope as rp
    ok = True
    for (m, n, k) in ROPE_SHAPES:
        X = init_randn((m, k)); W = init_randn((k, n)); gamma = init_randn((k,))
        r = torch.rsqrt(X.float().pow(2).mean(-1) + EPS).to(DTYPE)   # precomputed per-row inv-rms (bf16)
        cos_sin = rp.make_cos_sin(m, n)
        out = rp.make_rmsnorm_rope(W, gamma)(X, r, cos_sin)
        fin = bool(torch.isfinite(out).all())
        ref = rp.rmsnorm_rope_ref(X, gamma, W, r, cos_sin)
        e = (out.float() - ref).abs().max().item()
        ok &= _p(f"rmsnorm_rope {(m,n,k)}",
                 fin and torch.allclose(out.float(), ref, rtol=RTOL, atol=ATOL),
                 f"max_err={e:.3g}" + ("" if fin else " NON-FINITE"))
    return ok

def test_ce():
    """Fused forward cross-entropy: logits = h@W_vocab never materialized; the epilogue emits
    per-(row,group) softmax partials, the aux kernel combines them and adds the O(K) target dot ->
    per-row loss. Validated against F.cross_entropy(reduction='none'), plus two independent
    cross-checks (lse==logsumexp, target==Z[arange,labels]) to localize any failure."""
    import cross_entropy as ce
    ok = True
    for (M, N, K) in [(256,512,256),(512,1024,256),(2048,4096,512)]:
        h = init_randn((M, K))
        W = init_randn((K, N))                                  # [d_model, vocab]
        labels = torch.randint(0, N, (M,), device="cuda")
        loss = ce.make_ce(W)(h, labels)
        ref = ce.ce_ref(h, W, labels)
        fin = bool(torch.isfinite(loss).all())
        e = (loss.float() - ref).abs().max().item()
        ok &= _p(f"cross_entropy {(M,N,K)}",
                 fin and torch.allclose(loss.float(), ref, rtol=RTOL, atol=ATOL),
                 f"max_err={e:.3g}" + ("" if fin else " NON-FINITE"))
        # independent cross-checks: localize a failure to target-dot vs combine.
        Z = h.float() @ W.float()                               # [M,vocab] (host-only oracle)
        target = Z[torch.arange(M, device="cuda"), labels]
        lse = torch.logsumexp(Z, -1)
        ok &= _p(f"cross_entropy lse==logsumexp {(M,N,K)}",
                 torch.allclose((loss.float() + target), lse, rtol=2e-2, atol=1e-1))
    return ok


def test_ce_rms():
    """Fused RMS->cross-entropy: logits = rmsnorm(h,gamma)@W_lm == r*(h@gamma-folded W_lm), never
    materialized; epilogue scales by per-row r before the softmax partials. Validated against
    ce_rms_ref (F.cross_entropy on the fp32 rmsnorm logits)."""
    import cross_entropy as ce
    ok = True
    for (M, N, K) in [(256,512,256),(512,1024,256),(2048,4096,512)]:
        h = init_randn((M, K))
        W = init_randn((K, N))                                  # [d_model, vocab]
        gamma = init_randn((K,))
        labels = torch.randint(0, N, (M,), device="cuda")
        loss = ce.make_ce_rms(W, gamma)(h, labels)
        ref = ce.ce_rms_ref(h, gamma, W, labels)
        fin = bool(torch.isfinite(loss).all())
        e = (loss.float() - ref).abs().max().item()
        ok &= _p(f"cross_entropy_rms {(M,N,K)}",
                 fin and torch.allclose(loss.float(), ref, rtol=RTOL, atol=ATOL),
                 f"max_err={e:.3g}" + ("" if fin else " NON-FINITE"))
    return ok



def test_ce_padded_vocab():
    """Review #2 (fixed): a real vocab that isn't a multiple of 256 is supported by padding W to the
    next multiple of 256 and passing valid_n=real_vocab -- the epilogue masks columns >= valid_n out
    of the softmax, so logsumexp is over the real vocab only.
      masked   : make_ce(W_pad, valid_n=real) == ce_ref over the REAL [M, real_vocab] logits  (the fix)
      unmasked : make_ce(W_pad)               diverges from ref_REAL                          (the leak)
    Both forward CE and RMS->CE are checked, at realistic GPT-2 padding and a heavy 37.5% pad."""
    import cross_entropy as ce
    ok = True
    for (real_vocab, padded) in [(50257, 50432), (320, 512)]:        # realistic GPT-2, then 37.5% pad
        M, K = 256, 256
        h = init_randn((M, K)); gamma = init_randn((K,))
        W_real = init_randn((K, real_vocab))
        W_pad  = torch.cat([W_real, init_randn((K, padded - real_vocab))], dim=1)   # random pad columns
        labels = torch.randint(0, real_vocab, (M,), device="cuda")
        ref_real = ce.ce_ref(h, W_real, labels)
        masked   = ce.make_ce(W_pad, valid_n=real_vocab)(h, labels)  # cols >= real_vocab masked out
        unmasked = ce.make_ce(W_pad)(h, labels)                      # valid_n defaults to padded -> leaks
        e_mask = (masked.float()   - ref_real).abs().max().item()
        e_un   = (unmasked.float() - ref_real).abs().max().item()
        ok &= _p(f"ce padded masked==real_vocab (V={real_vocab}->{padded})",
                 torch.allclose(masked.float(), ref_real, rtol=RTOL, atol=ATOL), f"max_err={e_mask:.3g}")
        ok &= _p(f"ce padded mask removes the leak  (V={real_vocab}->{padded})", e_un > 10 * e_mask,
                 f"unmasked_err={e_un:.3g} >> masked_err={e_mask:.3g}")
        masked_r = ce.make_ce_rms(W_pad, gamma, valid_n=real_vocab)(h, labels)
        ref_r    = ce.ce_rms_ref(h, gamma, W_real, labels)
        e_mr = (masked_r.float() - ref_r).abs().max().item()
        ok &= _p(f"ce_rms padded masked==real_vocab (V={real_vocab}->{padded})",
                 torch.allclose(masked_r.float(), ref_r, rtol=RTOL, atol=ATOL), f"max_err={e_mr:.3g}")
        pad_labels = torch.full((M,), real_vocab, dtype=torch.long, device="cuda")   # labels pointing into the pad region
        loss_pad = ce.make_ce(W_pad, valid_n=real_vocab)(h, pad_labels)
        ok &= _p(f"ce padded: pad-range label -> loss 0 (V={real_vocab}->{padded})",
                 bool((loss_pad.float().abs() < 1e-6).all()), f"max|loss|={loss_pad.float().abs().max().item():.3g}")
    return ok


def test_binding_arity():
    """Review #4: each binding's dispatch arity must equal its registry args entry. The positional
    order must agree across three places (bind_function list, registry args lambda, hk.run *extra);
    an arity desync there compiles+runs and returns garbage. py::bind_function emits
    'dispatch(arg0: object, ...) -> None', so we parse the arity and compare to 3 (a,b,c) + len(extra)."""
    import re
    def arity(modname):
        doc = (importlib.import_module(modname).dispatch.__doc__ or "").replace("\n", " ")
        m = re.search(r"dispatch\s*\((.*?)\)\s*->", doc)
        if not m:
            return None, doc
        s = m.group(1).strip()
        return (0 if not s else len(s.split(","))), doc
    ok = True
    for name, spec in sorted(EPILOGUES.items()):             # store epilogues: a,b,c + own inputs
        got, doc = arity(spec["module"])
        expect = 3 + (len(spec["args"](256, 512, 256)) if "args" in spec else 0)
        ok &= _p(f"binding arity {name}", got == expect,
                 f"binding={got} registry={expect}" + ("" if got is not None else f"  unparsed:{doc[:48]!r}"))
    for modname, expect in [("tk_cross_entropy", 5), ("tk_ce_rms", 6)]:   # partials-only: a,b,max,sumexp,valid_n[,r]
        got, doc = arity(modname)
        ok &= _p(f"binding arity {modname}", got == expect, f"binding={got} expected={expect}")
    return ok


def test_known_answer():
    """Review #4: give the no-identity epilogues (silu, swiglu, ce) a hand-computed known answer, so an
    arg-order / gate<->value / max<->sumexp swap surfaces as wrong numbers rather than a same-order-ref
    tautology. Inputs are constructed so A@B is a known constant."""
    import math, hk, cross_entropy as ce
    M, K, N = 256, 256, 512
    sig1 = 1.0 / (1.0 + math.exp(-1.0))                      # sigmoid(1)
    A  = torch.ones(M, K, device="cuda", dtype=DTYPE)
    Bc = torch.full((K, N), 1.0 / K, device="cuda", dtype=DTYPE)   # A@Bc == 1 everywhere
    osi = hk.run("silu", A, Bc)
    ok = _p("known-answer silu(1)=sigmoid(1)", torch.allclose(osi.float(), torch.full_like(osi.float(), sig1), atol=2e-2),
            f"got={osi.float().mean().item():.4f} want={sig1:.4f}")
    d_ff = N // 2                                            # swiglu: gate=1, value=2 (distinct => a swap shows)
    Bsw = torch.empty(K, N, device="cuda", dtype=DTYPE)
    Bsw[:, :d_ff] = 1.0 / K; Bsw[:, d_ff:] = 2.0 / K        # natural [gate | value]; hk permutes internally
    osw = hk.run("swiglu", A, Bsw); want_sw = sig1 * 2.0     # silu(1)*2, NOT silu(2)*1
    ok &= _p("known-answer swiglu silu(1)*2", torch.allclose(osw.float(), torch.full_like(osw.float(), want_sw), atol=3e-2),
             f"got={osw.float().mean().item():.4f} want={want_sw:.4f}")
    h = torch.ones(M, K, device="cuda", dtype=DTYPE)         # ce: uniform logits => loss == log(vocab)
    W = torch.full((K, N), 1.0 / K, device="cuda", dtype=DTYPE)
    loss = ce.make_ce(W)(h, torch.randint(0, N, (M,), device="cuda")); want_ce = math.log(N)
    ok &= _p("known-answer ce uniform=log(vocab)", torch.allclose(loss.float(), torch.full_like(loss.float(), want_ce), atol=5e-2),
             f"got={loss.float().mean().item():.4f} want={want_ce:.4f}")
    return ok


def test_hk_run():
    """Review #5: drive every registered epilogue through hk.run (the user-facing entry) so its
    coercion / transpose / out_shape / weight_perm paths actually run (the rest of the suite calls
    mod.dispatch directly). Scalar fp32 args go in as plain floats (hk.run's float->fp32 path)."""
    import hk
    from swiglu import swiglu_ref
    M, N, K = 512, 512, 256
    ok = True
    for name, spec in sorted(EPILOGUES.items()):
        A = init_randn((M, K)); B = init_randn((K, N))
        reg_extra = spec["args"](M, N, K) if "args" in spec else ()
        hk_extra = tuple(float(x.reshape(-1)[0].item()) if (torch.is_tensor(x) and x.numel() == 1) else x
                         for x in reg_extra)                  # scale's alpha: fp32 tensor -> float for hk.run
        out = hk.run(name, A, B, *hk_extra)
        if "ref" in spec:
            D = hk.run("noop", A, B)                          # the bf16 kernel GEMM == the base the epilogue sees
            exp = torch.empty_like(out)
            spec["ref"](D, exp, *reg_extra)
            ref = exp.float()
        else:                                                # swiglu has no registry ref
            ref = swiglu_ref(A, B)
        e = (out.float() - ref).abs().max().item()
        ok &= _p(f"hk.run {name}", torch.allclose(out.float(), ref, rtol=RTOL, atol=ATOL), f"max_err={e:.3g}")
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
        ("[rmsnorm_swiglu]",      test_rmsnorm_swiglu,   ()),
        ("[partialrms]",          test_partialrms,       (noop, prms)),
        ("[residual_rms]",        test_residual_rms,     (rr,)),
        ("[residual_rms -> aux]", test_residual_rms_aux, (rr, aux)),
        ("[aux_reduce]",          test_aux_reduce,       (aux,)),
        ("[chain]",               test_chain,            ()),
        ("[rope]",                test_rope,             (rope_m,)),
        ("[rmsnorm_rope]",        test_rmsnorm_rope,     ()),
        ("[invariants]",          test_invariants,       (noop, scale_m, rms_m, resadd_m, silu_m)),
        ("[cross_entropy]",       test_ce,               ()),
        ("[cross_entropy_rms]",   test_ce_rms,           ()),
        ("[ce padded vocab]",     test_ce_padded_vocab,  ()),
        ("[binding arity]",       test_binding_arity,    ()),
        ("[known answer]",        test_known_answer,     ()),
        ("[hk.run coverage]",     test_hk_run,           ()),
    ]
    allpass = True
    for label, fn, fargs in suite:
        print(label)
        allpass &= fn(*fargs)
    print("ALL PASSED" if allpass else "SOME FAILED")
    return allpass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
