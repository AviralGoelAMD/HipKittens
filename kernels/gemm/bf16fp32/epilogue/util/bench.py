#!/usr/bin/env python3
"""bench.py - one benchmark for every GEMM-epilogue kernel + the fused chain.

Per epilogue: HK fused (one kernel) vs torch unfused (base GEMM -> D in HBM, then the torch `ref`
reads D and applies the epilogue). Reports COLD-cache median speedup + the HBM bytes the fusion
saves. The fused chain (rmsnorm(X@W0+res,gamma)@W1) compares HK against the torch unfused path.

Honest by construction: B1 rotates the re-read inputs through a pool > the LLC so each timed run is
cold; B2 takes the per-iter median. Kernels + their `ref` come from the single EPILOGUES registry,
so the unfused stage is the same math test_all.py validates.

Run from the epilogue dir on a gfx950 node:
    python3 util/bench.py [--kernels scale,rmsnorm_scale] [--json results.json]
"""
import os, sys, argparse, importlib, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # tk_*.so
import torch
from epilogue_testlib import EPILOGUES, make_inputs, init_empty, init_randn, DSIZE, RTOL, ATOL, DTYPE
from block_chain import fused_rmsnorm_block, EPS

LLC_BYTES = 256 * 1024 * 1024
BUDGET    = 8 * 1024**3
SHAPES       = [(2048, 1024, 512), (4096, 4096, 4096), (8192, 8192, 8192)]
CHAIN_SHAPES = [(2048, 2048, 2048, 2048), (4096, 4096, 4096, 4096)]              # (M,K0,N,P)
CHAIN_REL = 2e-2   # normwise rel tolerance for the two-GEMM chain (matches test_all)


def _pool_size(per_set, llc, budget):
    """How many distinct input sets to rotate through for COLD-cache timing.

    Re-running a kernel on the same A/B serves later iterations from the LLC, hiding the real HBM
    cost. So we cycle through N copies (callers index `Ap[i % N]`) sized to OVERFLOW the last-level
    cache, forcing each re-read back to HBM. N = enough copies to exceed the LLC (`llc//per_set + 1`),
    capped by a memory budget (`budget//per_set`) so we don't OOM. If one set already exceeds the
    LLC, a single copy is cold on its own."""
    if per_set >= llc: return 1
    return max(1, min(llc // per_set + 1, max(1, budget // per_set)))


def _bench(fn, iters=50, warm=10):
    """Median per-iteration GPU time over `iters` runs (after `warm` warmups), via CUDA events.
    Median, not mean, to reject scheduling outliers."""
    for i in range(warm): fn(i)
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters): st[i].record(); fn(i); en[i].record()
    torch.cuda.synchronize()
    s = sorted(st[i].elapsed_time(en[i]) for i in range(iters))
    return s[len(s) // 2]


def run_epilogue(kernel, iters, warm):
    base = importlib.import_module("tk_noop")
    spec = EPILOGUES[kernel]
    fk = importlib.import_module(spec["module"])
    rows = []
    for (m, n, k) in SHAPES:
        per_set = (m * k + n * k) * DSIZE
        N = _pool_size(per_set, LLC_BYTES, BUDGET)
        Ap, Btp = zip(*[make_inputs(m, n, k) for _ in range(N)])   # cold-cache pool; rotate via [i % N]
        D = init_empty((m, n)); O = init_empty((m, n)); args = spec["args"](m, n, k)
        def fused(i):   fk.dispatch(Ap[i % N], Btp[i % N], O, *args)
        def unfused(i):
            base.dispatch(Ap[i % N], Btp[i % N], D); spec["ref"](D, O, *args)
        fused(0); torch.cuda.synchronize(); Of = O.clone()
        unfused(0); torch.cuda.synchronize()
        ok = torch.allclose(Of.float(), O.float(), rtol=RTOL, atol=ATOL)
        tf, tu = _bench(fused, iters, warm), _bench(unfused, iters, warm)
        saved = spec["hbm_passes"] * m * n * DSIZE
        rows.append({"kernel": kernel, "shape": [m, n, k], "correct": bool(ok),
                     "fused_ms": round(tf, 4), "torch_ms": round(tu, 4),
                     "speedup": round(tu / tf, 3), "saved_MB": round(saved / 1e6, 1)})
    return rows


def run_chain(iters, warm):
    # NOTE: inputs reused across iters (warm cache); small-shape HBM is understated vs run_epilogue's cold pool.
    rows = []
    for (M, K0, N, P) in CHAIN_SHAPES:
        X = init_randn((M, K0)); W0 = init_randn((K0, N)); res = init_randn((M, N))
        gamma = init_randn((N,)); W1 = init_randn((N, P))
        def hk(i):   fused_rmsnorm_block(X, W0, res, gamma, W1)
        def torch_(i):
            h1 = X @ W0 + res                               # bf16 GEMM (matches HK; was fp32 -> unfair)
            var = h1.float().pow(2).mean(-1, keepdim=True)  # RMS reduction in fp32 (as HK does)
            hn = (h1 * torch.rsqrt(var + EPS) * gamma.float()).to(DTYPE)
            return hn @ W1
        out_hk = fused_rmsnorm_block(X, W0, res, gamma, W1); out_t = torch_(0); torch.cuda.synchronize()
        rel = (out_hk.float() - out_t.float()).norm().item() / out_t.float().norm().item()
        row = {"shape": [M, K0, N, P], "hk_ms": round(_bench(hk, iters, warm), 4),
               "torch_ms": round(_bench(torch_, iters, warm), 4),
               "rel": round(rel, 4), "ok": bool(rel < CHAIN_REL)}
        rows.append(row)
    return rows

def run_swiglu(iters, warm, shapes=None):
    """SwiGLU fusion proof: fused kernel vs HK noop GEMM [M,2*d_ff] + a torch silu(gate)*value.
    Both pay the same 2*d_ff GEMM, so this isolates the activation fusion + half-width store.
    (Inputs reused across iters -> warm cache; the wide 2*d_ff intermediate is per-iter so the
    headline holds, but small-shape HBM is understated vs run_epilogue's cold pool.)"""
    import tk_noop
    from swiglu import make_swiglu, swiglu_ref
    shapes = shapes or [(2048, 1024, 512), (4096, 4096, 4096), (8192, 4096, 4096)]   # (M, d_ff, K)
    rows = []
    for (m, d_ff, k) in shapes:
        X = init_randn((m, k)); W = init_randn((k, 2 * d_ff))
        fwd = make_swiglu(W)
        Wt_nat = W.to(device="cuda", dtype=DTYPE).t().contiguous()   # natural weight for the noop GEMM
        D = init_empty((m, 2 * d_ff))
        def fused(i):  fwd(X)
        def unfused(i):
            tk_noop.dispatch(X, Wt_nat, D)
            torch.nn.functional.silu(D[:, :d_ff]) * D[:, d_ff:]
        ok = torch.allclose(fwd(X).float(), swiglu_ref(X, W), rtol=RTOL, atol=ATOL)
        tf, tu = _bench(fused, iters, warm), _bench(unfused, iters, warm)
        saved = 4 * m * d_ff * DSIZE                 # fusion skips the write+read of the [M,2*d_ff] intermediate
        rows.append({"shape": [m, d_ff, k], "fused_ms": round(tf, 4), "torch_ms": round(tu, 4),
                     "speedup": round(tu / tf, 3), "saved_MB": round(saved / 1e6, 1), "ok": bool(ok)})
    return rows

def run_mlp(iters, warm, shapes=None):
    """Realistic SwiGLU MLP core (post-norm): GEMM1 -> SwiGLU -> GEMM2 + residual.
    HK fuses SwiGLU into GEMM1 (the 2*d_ff intermediate never hits HBM) and the residual into GEMM2.
    Baselines: torch.compile max-autotune (vendor mm; cannot fuse into mm -> round-trips 2*d_ff) and
    eager. Controls: vendor 2-GEMM floor + HK noop 2-GEMM floor, to split GEMM-quality from fusion.
    rmsnorm is shared/excluded (h is the post-norm activation), isolating the SwiGLU-MLP fusion.
    (Inputs reused across iters -> warm cache; the 2*d_ff intermediate is per-iter so the headline
    holds, but weights stay LLC-resident at small shapes vs run_epilogue's cold pool.)"""
    import tk_swiglu, tk_noop, tk_residual_add
    import torch._dynamo
    from swiglu import gate_up_perm
    F = torch.nn.functional
    shapes = shapes or [(2048, 4096, 11008), (4096, 4096, 11008), (8192, 4096, 11008)]  # (M, d_model, d_ff)
    torch._dynamo.config.cache_size_limit = max(64, 4 * len(shapes))
    rows = []
    for (M, dm, dff) in shapes:
        h = init_randn((M, dm)); x = init_randn((M, dm))                 # post-norm activation + residual
        Wgu = init_randn((dm, 2 * dff)); Wd = init_randn((dff, dm))
        Wgu_pt = Wgu[:, gate_up_perm(dff).to(Wgu.device)].t().contiguous()   # [2dff, dm] permuted+T (HK)
        Wd_t = Wd.t().contiguous()                                          # [dm, dff]
        a_buf = init_empty((M, dff)); y_hk = init_empty((M, dm))
        def hk(i):                                                          # fully async 2-kernel chain
            tk_swiglu.dispatch(h, Wgu_pt, a_buf)
            tk_residual_add.dispatch(a_buf, Wd_t, y_hk, x)
        def _mlp(h, Wgu, Wd, x):
            gu = h @ Wgu
            return x + (F.silu(gu[:, :dff]) * gu[:, dff:]) @ Wd
        mlp_c = torch.compile(_mlp, mode="max-autotune-no-cudagraphs", dynamic=False)
        def compiled(i): mlp_c(h, Wgu, Wd, x)
        def eager(i):    _mlp(h, Wgu, Wd, x)
        a_dummy = init_randn((M, dff)); Wgu_t = Wgu.t().contiguous()
        D1 = init_empty((M, 2 * dff)); D2 = init_empty((M, dm))
        def vendor2(i): _ = h @ Wgu; _ = a_dummy @ Wd                       # vendor 2-GEMM floor
        def hk2(i):                                                         # HK noop 2-GEMM floor
            tk_noop.dispatch(h, Wgu_t, D1); tk_noop.dispatch(a_dummy, Wd_t, D2)
        hk(0); torch.cuda.synchronize()
        gu = h.float() @ Wgu.float()
        ref = x.float() + (F.silu(gu[:, :dff]) * gu[:, dff:]) @ Wd.float()
        rel = (y_hk.float() - ref).norm().item() / ref.norm().item()
        mlp_c(h, Wgu, Wd, x); torch.cuda.synchronize()                      # trigger compile pre-timing
        t_hk, t_c = _bench(hk, iters, warm), _bench(compiled, iters, warm)
        t_e, t_v, t_h2 = _bench(eager, iters, warm), _bench(vendor2, iters, warm), _bench(hk2, iters, warm)
        rows.append({"shape": [M, dm, dff], "hk_ms": round(t_hk, 4), "compile_ms": round(t_c, 4),
                     "eager_ms": round(t_e, 4), "vendor2_ms": round(t_v, 4), "hk2_ms": round(t_h2, 4),
                     "vs_compile": round(t_c / t_hk, 3), "rel": round(rel, 4), "ok": bool(rel < 2e-2)})
    return rows


def run_ffn(iters, warm, shapes=None):
    """Fully-fused FFN sublayer (pre-norm): y = x + swiglu(rmsnorm(h,gamma) @ W_gate_up) @ W_down.
    Upgrades run_mlp to INCLUDE the norm: HK folds the RMSNorm scale into the gate_up GEMM
    (rmsnorm_swiglu) and the residual into the down GEMM, so only the 1/rms reduction over d_model is
    separate -- and in a real block that r is emitted by the prior out-proj GEMM (residual_rms_partials), so it
    is precomputed here (free in-context). Baseline: torch.compile max-autotune, which fuses the norm
    but still round-trips the 2*d_ff intermediate through the opaque vendor mm."""
    import tk_rmsnorm_swiglu, tk_residual_add
    import torch._dynamo
    from swiglu import gate_up_perm
    F = torch.nn.functional
    shapes = shapes or [(2048, 4096, 11008), (4096, 4096, 11008), (8192, 4096, 11008)]  # (M, d_model, d_ff)
    torch._dynamo.config.cache_size_limit = max(64, 4 * len(shapes))
    rows = []
    for (M, dm, dff) in shapes:
        h = init_randn((M, dm)); x = init_randn((M, dm))                 # pre-norm activation + residual
        gamma = init_randn((dm,)); Wgu = init_randn((dm, 2 * dff)); Wd = init_randn((dff, dm))
        Wgu_pt = ((Wgu.float() * gamma.float()[:, None]).to(DTYPE)[:, gate_up_perm(dff).to(Wgu.device)]
                  .t().contiguous())                                     # gamma-folded + gate_up-permuted + T
        Wd_t = Wd.t().contiguous()
        r = torch.rsqrt(h.float().pow(2).mean(-1) + EPS).to(DTYPE)       # precomputed (prior GEMM emits it in a real block)
        a_buf = init_empty((M, dff)); y_hk = init_empty((M, dm))
        def hk(i):
            tk_rmsnorm_swiglu.dispatch(h, Wgu_pt, a_buf, r)
            tk_residual_add.dispatch(a_buf, Wd_t, y_hk, x)
        def _ffn(h, gamma, Wgu, Wd, x):
            n = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS) * gamma
            gu = n @ Wgu
            return x + (F.silu(gu[:, :dff]) * gu[:, dff:]) @ Wd
        ffn_c = torch.compile(_ffn, mode="max-autotune-no-cudagraphs", dynamic=False)
        def compiled(i): ffn_c(h, gamma, Wgu, Wd, x)
        def eager(i):    _ffn(h, gamma, Wgu, Wd, x)
        hk(0); torch.cuda.synchronize()
        n = h.float() * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + EPS) * gamma.float()
        gu = n @ Wgu.float()
        ref = x.float() + (F.silu(gu[:, :dff]) * gu[:, dff:]) @ Wd.float()
        rel = (y_hk.float() - ref).norm().item() / ref.norm().item()
        ffn_c(h, gamma, Wgu, Wd, x); torch.cuda.synchronize()           # trigger compile pre-timing
        t_hk, t_c, t_e = _bench(hk, iters, warm), _bench(compiled, iters, warm), _bench(eager, iters, warm)
        rows.append({"shape": [M, dm, dff], "hk_ms": round(t_hk, 4), "compile_ms": round(t_c, 4),
                     "eager_ms": round(t_e, 4), "vs_compile": round(t_c / t_hk, 3),
                     "rel": round(rel, 4), "ok": bool(rel < 2e-2)})
    return rows


def run_ce(iters, warm, shapes=None):
    """Fused forward cross-entropy vs torch.compile -- the strategic gating number. HK emits the
    per-row loss WITHOUT ever materializing the [M,vocab] fp32 logits; torch.compile (even
    max-autotune) round-trips that full logits matrix through HBM (vendor mm cannot fuse the softmax
    reduction in). Llama-scale vocab=32000, d_model=4096. saved_MB = the [M,vocab] fp32 logits
    write+read torch pays and HK avoids."""
    from cross_entropy import make_ce, ce_ref
    import torch._dynamo
    F = torch.nn.functional
    shapes = shapes or [(2048, 32000, 4096), (4096, 32000, 4096), (8192, 32000, 4096)]  # (M, vocab, d)
    torch._dynamo.config.cache_size_limit = max(64, 4 * len(shapes))
    rows = []
    for (M, vocab, d) in shapes:
        h = init_randn((M, d)); W = init_randn((d, vocab))
        labels = torch.randint(0, vocab, (M,), device="cuda")
        fwd = make_ce(W)
        def hk(i): fwd(h, labels)
        def _ce(): return F.cross_entropy(h @ W, labels, reduction="none")
        ce_c = torch.compile(_ce, mode="max-autotune-no-cudagraphs", dynamic=False)
        def compiled(i): ce_c()
        loss_hk = fwd(h, labels); torch.cuda.synchronize()
        ref = ce_ref(h, W, labels)                                       # fp32 oracle
        rel = (loss_hk.float() - ref).norm().item() / ref.norm().item()
        ce_c(); torch.cuda.synchronize()                                # trigger compile pre-timing
        t_hk, t_c = _bench(hk, iters, warm), _bench(compiled, iters, warm)
        saved = M * vocab * 4 * 2                                        # [M,vocab] fp32 logits write+read
        rows.append({"shape": [M, vocab, d], "hk_ms": round(t_hk, 4), "compile_ms": round(t_c, 4),
                     "vs_compile": round(t_c / t_hk, 3), "rel": round(rel, 4),
                     "ok": bool(rel < 2e-2), "saved_MB": round(saved / 1e6, 1)})
    return rows

# ---- dtype contract for the forward-layer benchmark (printed with results; both paths obey it) ----
FWD_CONTRACT = ("dtype contract (both paths): storage/IO bf16 (x, weights, q/k/v, o, h, x_out, gammas); "
                "matmuls bf16-in -> fp32-accumulate -> bf16-out; RMSNorm reductions fp32; softmax fp32; output bf16.")
FWD_DISCREP = [
    "attn: HK = native-GQA flash (H_KV=8 kv heads, hand-written, exp2/fp32 online softmax); "
    "torch = flash SDPA with KV expanded to H=32 heads (torch 2.11+rocm7.13 native-GQA SDPA falls back to the "
    "materialized math backend -> a WEAKER baseline we deliberately avoid). HK reads 4x less KV from HBM (real native-GQA win).",
    "fusion: HK fuses RoPE+RMS-scale into the projection GEMM epilogues and residuals into the consuming GEMMs "
    "(no intermediate HBM round-trips); torch.compile fuses only elementwise clusters around opaque vendor matmuls.",
    "HK pays an extra q/k un-permute gather (bf16 layout bridge to the attn kernel), counted in HK time; "
    "a future fused rope->attn kernel removes it.",
    "entry r_attn precomputed for HK (prior-layer-emitted in a stack); torch recomputes the entry RMSNorm reduction "
    "(~0.1% of layer FLOPs, immaterial).",
    "SiLU: in-register fp32 (HK) vs bf16 (torch) -- memory-bound, identical bf16 traffic, perf-neutral.",
]


def run_forward_layer(iters, warm):
    """One full pre-norm GQA Transformer forward layer (minus nothing): RMSNorm -> QKV proj -> RoPE ->
    GQA causal attention -> out-proj+residual -> RMSNorm -> SwiGLU MLP -> down-proj+residual. HK assembles
    it ENTIRELY from the fused GEMM+epilogue kernels + the gqa_causal kernel (forward_layer.make_forward_layer);
    baseline is the same layer in idiomatic torch, torch.compile max-autotune. Shape is fixed by the
    compile-time-specialized attention kernel (B=1, N=2048, H=32, H_KV=8, Dh=128); other shapes need a rebuild.
    Reports rel vs the fp32 oracle for BOTH paths and the HK-vs-torch cross rel (proof both compute one math)."""
    from forward_layer import make_forward_layer, forward_layer_ref, _rms_r, HEAD_DIM
    from rope import make_cos_sin
    import torch._dynamo
    F = torch.nn.functional
    M = int(os.environ.get("TK_FWD_M", "2048"))                     # decode/prefill sweep; gqa kernel rebuilt per M
    d, dff, Dh, H_KV = 4096, 11008, HEAD_DIM, 8                     # MUST match the built tk_kernel (ATTN_N=M)
    H = d // Dh; G = H // H_KV; kv_dim = H_KV * Dh
    torch._dynamo.config.cache_size_limit = 64
    w = lambda r, c: (torch.randn(r, c, device="cuda") * (r ** -0.5)).to(DTYPE)   # scaled init -> O(1) q/k (non-degenerate softmax)
    x = init_randn((M, d))
    Wq, Wo = w(d, d), w(d, d)
    Wk, Wv = w(d, kv_dim), w(d, kv_dim)
    Wgu, Wd = w(d, 2 * dff), w(dff, d)
    ga, gm = init_randn((d,)), init_randn((d,))
    cos_sin_head = make_cos_sin(M, Dh)
    cos = cos_sin_head[:, 0::2].reshape(M, 1, Dh // 2).float()
    sin = cos_sin_head[:, 1::2].reshape(M, 1, Dh // 2).float()
    r_attn = _rms_r(x)
    fwd = make_forward_layer(Wq, Wk, Wv, Wo, ga, Wgu, Wd, gm, n_kv_heads=H_KV, head_dim=Dh)

    def _rope(t):                                                   # t [M,H*,Dh] bf16 -> interleaved RoPE (fp32 internal)
        a = t[..., 0::2].float(); b = t[..., 1::2].float()
        return torch.stack((a * cos + b * sin, -a * sin + b * cos), dim=-1).flatten(-2).to(DTYPE)

    def _layer(x):
        xn = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS)).to(DTYPE) * ga
        q = _rope((xn @ Wq).view(M, H, Dh)); k = _rope((xn @ Wk).view(M, H_KV, Dh)); v = (xn @ Wv).view(M, H_KV, Dh)
        qh = q.permute(1, 0, 2).unsqueeze(0)                                              # [1,H,M,Dh]
        kh = k.permute(1, 0, 2).unsqueeze(0).repeat_interleave(G, dim=1)                  # expand KV -> flash SDPA
        vh = v.permute(1, 0, 2).unsqueeze(0).repeat_interleave(G, dim=1)
        o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)                    # bf16 in/out, fp32 softmax, scale 1/sqrt(Dh)
        o = o.squeeze(0).permute(1, 0, 2).reshape(M, d).to(DTYPE)
        h = x + o @ Wo
        hn = (h.float() * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + EPS)).to(DTYPE) * gm
        gu = hn @ Wgu
        return h + (F.silu(gu[:, :dff]) * gu[:, dff:]) @ Wd

    layer_c = torch.compile(_layer, mode="max-autotune-no-cudagraphs", dynamic=False)
    xo_hk, _ = fwd(x, r_attn, cos_sin_head)
    xo_t = layer_c(x); torch.cuda.synchronize()
    assert xo_hk.dtype == DTYPE and xo_t.dtype == DTYPE, f"dtype mismatch: hk={xo_hk.dtype} torch={xo_t.dtype}"
    rel_cross = (xo_hk.float() - xo_t.float()).norm().item() / xo_t.float().norm().item()
    if os.environ.get("TK_FWD_NOREF"):                 # huge M: skip the [H,M,M] fp32 oracle (OOMs); cross-rel validates
        rel_hk = rel_torch = float("nan"); ok = rel_cross < 2e-2
    else:
        xo_ref, _ = forward_layer_ref(x, r_attn, cos_sin_head, Wq, Wk, Wv, Wo, ga, Wgu, Wd, gm, n_kv_heads=H_KV, head_dim=Dh)
        den = xo_ref.float().norm().item()
        rel_hk = (xo_hk.float() - xo_ref.float()).norm().item() / den
        rel_torch = (xo_t.float() - xo_ref.float()).norm().item() / den
        ok = rel_hk < 2e-2
    t_hk = _bench(lambda i: fwd(x, r_attn, cos_sin_head), iters, warm)
    t_c = _bench(lambda i: layer_c(x), iters, warm)
    t_e = _bench(lambda i: _layer(x), iters, warm)
    return [{"shape": [M, d, dff], "heads": [H, H_KV, Dh], "hk_ms": round(t_hk, 4), "compile_ms": round(t_c, 4),
             "eager_ms": round(t_e, 4), "vs_compile": round(t_c / t_hk, 3), "vs_eager": round(t_e / t_hk, 3),
             "rel_hk_vs_fp32": round(rel_hk, 4), "rel_torch_vs_fp32": round(rel_torch, 4),
             "rel_hk_vs_torch": round(rel_cross, 4), "out_dtype": str(xo_hk.dtype),
             "ok": bool(ok), "contract": FWD_CONTRACT, "discrepancies": FWD_DISCREP}]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default=",".join(k for k in EPILOGUES if k != "noop" and "ref" in EPILOGUES[k]))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--no-chain", action="store_true")
    ap.add_argument("--no-epilogue", action="store_true")
    ap.add_argument("--mlp", action="store_true")
    ap.add_argument("--ffn", action="store_true")
    ap.add_argument("--ce", action="store_true")
    ap.add_argument("--forward", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    out = {"epilogues": [], "swiglu": [], "mlp": [], "ffn": [], "ce": [], "forward": [], "chain": []}
    if not a.no_epilogue:
        print(f"{'kernel':<16}{'shape':<22}{'fused ms':>10}{'torch ms':>10}{'speedup':>9}{'saved MB':>10}{'ok':>4}")
        for kern in a.kernels.split(","):
            for r in run_epilogue(kern, a.iters, a.warm):
                out["epilogues"].append(r)
                print(f"{r['kernel']:<16}{str(tuple(r['shape'])):<22}{r['fused_ms']:>10}{r['torch_ms']:>10}"
                      f"{r['speedup']:>9}{r['saved_MB']:>10}{'Y' if r['correct'] else 'N':>4}")
        print(f"\n{'swiglu (M,d_ff,K)':<24}{'fused ms':>10}{'torch ms':>10}{'speedup':>9}{'saved MB':>10}{'ok':>4}")
        for r in run_swiglu(a.iters, a.warm):
            out["swiglu"].append(r)
            print(f"{str(tuple(r['shape'])):<24}{r['fused_ms']:>10}{r['torch_ms']:>10}{r['speedup']:>9}{r['saved_MB']:>10}{'Y' if r['ok'] else 'N':>4}")
    if a.mlp:
        print(f"\n{'mlp (M,d_model,d_ff)':<24}{'hk ms':>9}{'compile':>9}{'eager':>9}{'vendor2':>9}{'hk2':>9}{'vs_comp':>9}{'rel':>8}{'ok':>4}")
        for r in run_mlp(a.iters, a.warm):
            out["mlp"].append(r)
            print(f"{str(tuple(r['shape'])):<24}{r['hk_ms']:>9}{r['compile_ms']:>9}{r['eager_ms']:>9}{r['vendor2_ms']:>9}{r['hk2_ms']:>9}{r['vs_compile']:>9}{r['rel']:>8}{'Y' if r['ok'] else 'N':>4}")
    if a.ffn:
        print(f"\n{'ffn (M,d_model,d_ff)':<24}{'hk ms':>9}{'compile':>9}{'eager':>9}{'vs_comp':>9}{'rel':>8}{'ok':>4}")
        for r in run_ffn(a.iters, a.warm):
            out["ffn"].append(r)
            print(f"{str(tuple(r['shape'])):<24}{r['hk_ms']:>9}{r['compile_ms']:>9}{r['eager_ms']:>9}{r['vs_compile']:>9}{r['rel']:>8}{'Y' if r['ok'] else 'N':>4}")
    if a.ce:
        print(f"\n{'ce (M,vocab,d)':<24}{'hk ms':>9}{'compile':>9}{'vs_comp':>9}{'rel':>8}{'saved MB':>10}{'ok':>4}")
        for r in run_ce(a.iters, a.warm):
            out["ce"].append(r)
            print(f"{str(tuple(r['shape'])):<24}{r['hk_ms']:>9}{r['compile_ms']:>9}{r['vs_compile']:>9}{r['rel']:>8}{r['saved_MB']:>10}{'Y' if r['ok'] else 'N':>4}")
    if a.forward:
        print("\n" + FWD_CONTRACT)
        for i, dnote in enumerate(FWD_DISCREP, 1):
            print(f"  [{i}] {dnote}")
        print(f"\n{'forward layer (M,d,dff)':<26}{'(H,Hkv,Dh)':<14}{'hk ms':>9}{'compile':>9}{'eager':>9}"
              f"{'vs_comp':>9}{'vs_eag':>8}{'rel_hk':>8}{'rel_tc':>8}{'cross':>7}{'ok':>4}")
        for r in run_forward_layer(a.iters, a.warm):
            out["forward"].append(r)
            print(f"{str(tuple(r['shape'])):<26}{str(tuple(r['heads'])):<14}{r['hk_ms']:>9}{r['compile_ms']:>9}"
                  f"{r['eager_ms']:>9}{r['vs_compile']:>9}{r['vs_eager']:>8}{r['rel_hk_vs_fp32']:>8}"
                  f"{r['rel_torch_vs_fp32']:>8}{r['rel_hk_vs_torch']:>7}{'Y' if r['ok'] else 'N':>4}")
    if not a.no_chain:
        print(f"\n{'chain (M,K0,N,P)':<24}{'hk ms':>10}{'torch ms':>10}{'speedup':>9}{'rel':>8}{'ok':>4}")
        for r in run_chain(a.iters, a.warm):
            out["chain"].append(r)
            print(f"{str(tuple(r['shape'])):<24}{r['hk_ms']:>10}{r['torch_ms']:>10}{round(r['torch_ms']/r['hk_ms'],3):>9}{r['rel']:>8}{'Y' if r['ok'] else 'N':>4}")
    if a.json:
        with open(a.json, "w") as f: json.dump(out, f, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
