"""forward_layer.py - a full pre-norm Transformer forward layer assembled ENTIRELY from the fused
GEMM+epilogue kernels plus the HipKittens GQA causal-attention kernel (no new kernels -- pure
composition).

Milestone 1 used a torch attention stub; Milestone 2 swaps in the real multi-head GQA kernel
(kernels/attn/gqa_causal, module `tk_kernel`).

Reparameterization (CODA): RMSNorm(y,gamma) = r * y * gamma, r = 1/rms(y) per row. Because r is
per-row it commutes through the next GEMM, and gamma scales the contracted axis so it folds into the
next weight's ROWS. So each sublayer boundary is:

    producing GEMM emits (residual stream h, partial Sum h^2)   [tk_residual_rms_partials, gamma=ones]
    aux combines the partials -> r = 1/rms(h)                   [tk_rms_reduce]
    consuming GEMM applies r in its epilogue (+ activation)     [rope q/k | scale v | swiglu gate-up]

Multi-head attention wiring. q is [M, H*Dh], k/v are [M, H_KV*Dh] (grouped-query, GROUP=H/H_KV).
RoPE is HEAD-LOCAL: each head's Dh dims rotate with the same [M,Dh] table, so the full cos_sin is
that table tiled H (resp. H_KV) times. The fused rope kernel stores its output in rope_perm column
order (a 256-tile co-residency artifact that cancels in single-head Q*K^T but mixes adjacent
Dh=128 heads across 256-blocks); for per-head GQA we therefore UN-PERMUTE q/k back to natural
column order before reshaping into [.,H,Dh] for the attention kernel. The attention scale is
1/sqrt(Dh) (head dim), matching the kernel's TEMPERATURE_SCALE.

One layer (input residual stream x, precomputed r_attn = 1/rms(x)):
    q = unperm(RoPE_hl(r_attn * (x @ Wq_gfold)))   # tk_rmsnorm_rope  -> [M,H,Dh]
    k = unperm(RoPE_hl(r_attn * (x @ Wk_gfold)))   # tk_rmsnorm_rope  -> [M,H_KV,Dh]
    v =              r_attn * (x @ Wv_gfold)        # tk_rmsnorm_scale -> [M,H_KV,Dh]
    o = gqa_causal(q, k, v)                         # tk_kernel (GQA)  -> [M,H*Dh]=[M,d]
    h = x + o @ Wo ; partials = Sum h^2             # tk_residual_rms_partials (gamma=ones)
    r_mlp = 1/rms(h)                                # tk_rms_reduce
    g = SwiGLU(r_mlp * (h @ Wgu_gfold))            # tk_rmsnorm_swiglu (gamma_mlp folded into Wgu rows)
    x_out = h + g @ Wd ; partials = Sum x_out^2     # tk_residual_rms_partials (gamma=ones) -> next layer's r_attn
    r_attn_next = 1/rms(x_out)                      # tk_rms_reduce

Shapes (batch=1, causal): x[M,d], Wq/Wo[d,d], Wk/Wv[d,H_KV*Dh], Wgu[d,2*dff], Wd[dff,d], gammas[d].
Constraints inherited from the GEMM/attn: M,d,2*dff,dff,H_KV*Dh % 256 ; d % 128 ; Dh=128 ; H=d/Dh ;
H % H_KV == 0. The attention module (tk_kernel) is compile-time specialized for (B,N,H,H_KV,Dh).
"""
import math, os, sys, torch
import tk_residual_rms_partials, tk_rms_reduce, tk_rmsnorm_scale
from swiglu import make_rmsnorm_swiglu, rmsnorm_swiglu_ref
from rope import make_rmsnorm_rope, rmsnorm_rope_ref, make_cos_sin, rope_perm

# the GQA causal attention kernel (PYBIND11_MODULE(tk_kernel)) lives in kernels/attn/gqa_causal
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "..", "attn", "gqa_causal"))
import tk_kernel as tk_gqa

DTYPE = torch.bfloat16
EPS = 1e-5
REG_BLOCK_N = 64
HEAD_DIM = 128       # the gqa_causal kernel's ATTN_D (kernel.cpp / kernel_d64.cpp)


def _rms_r(y):  # per-row inv-rms (bf16), the value the prior GEMM's aux would emit
    return torch.rsqrt(y.float().pow(2).mean(-1) + EPS).to(DTYPE)


def _k4(prev, Wt, residual):
    """tk_residual_rms_partials with gamma=ones: c = prev@W + residual (the residual stream), partials = Sum c^2;
    returns (c, r=1/rms(c)). gamma=ones because the next norm's gamma folds into the next weight."""
    M, N = residual.shape
    c = torch.empty((M, N), dtype=DTYPE, device="cuda")
    save = torch.empty((M, N), dtype=DTYPE, device="cuda")
    partials = torch.empty((N // REG_BLOCK_N, M), dtype=torch.float32, device="cuda")
    g_ones = torch.ones(N, dtype=DTYPE, device="cuda")
    tk_residual_rms_partials.dispatch(prev, Wt, c, residual, g_ones, partials, save)
    r = torch.empty(M, dtype=DTYPE, device="cuda")
    tk_rms_reduce.reduce(partials, r)
    return c, r


def _gqa(q, k, v, n_heads):
    """Run the gqa_causal kernel. q[M,d] / k,v[M,kv_dim] in NATURAL per-head column order (un-perm'd).
    Returns o[M,d] (heads concatenated). The kernel applies scale 1/sqrt(Dh) and causal masking."""
    M, d = q.shape
    kv_dim = k.shape[1]
    Dh = HEAD_DIM
    H, H_KV = n_heads, kv_dim // Dh
    qg = q.contiguous().view(1, M, H, Dh)
    kg = k.contiguous().view(1, M, H_KV, Dh)
    vg = v.contiguous().view(1, M, H_KV, Dh)
    out = torch.empty(1, M, H, Dh, dtype=DTYPE, device="cuda")
    lse = torch.empty(1, H, 1, M, dtype=torch.float32, device="cuda")
    tk_gqa.dispatch_micro(qg, kg, vg, out, lse)
    return out.view(M, d)


def make_forward_layer(Wq, Wk, Wv, Wo, gamma_attn, Wgu, Wd, gamma_mlp, n_kv_heads, head_dim=HEAD_DIM):
    """Build a forward layer once (fold gammas, transpose weights, precompute un-perm indices).
    Returns forward(x, r_attn, cos_sin_head) -> (x_out, r_attn_next). cos_sin_head is the per-head
    [M, head_dim] interleaved RoPE table (tiled across heads internally). batch=1, causal, GQA."""
    d = Wq.shape[0]
    kv_dim = Wk.shape[1]
    Dh = head_dim
    H, H_KV = d // Dh, n_kv_heads
    assert kv_dim == H_KV * Dh and H % H_KV == 0, "GQA: kv_dim must be H_KV*Dh and H divisible by H_KV"
    qk = (make_rmsnorm_rope(Wq, gamma_attn), make_rmsnorm_rope(Wk, gamma_attn))   # RoPE proj for q,k
    inv_q = torch.argsort(rope_perm(d)).to("cuda")                                # un-perm to natural cols
    inv_k = torch.argsort(rope_perm(kv_dim)).to("cuda")
    Wv_t = (Wv.to("cuda", DTYPE).float() * gamma_attn.to("cuda", DTYPE).float()[:, None]).to(DTYPE).t().contiguous()
    Wo_t = Wo.to("cuda", DTYPE).t().contiguous()
    mlp = make_rmsnorm_swiglu(Wgu, gamma_mlp)                                     # gate/up + SwiGLU
    Wd_t = Wd.to("cuda", DTYPE).t().contiguous()
    g_ones_kv = torch.ones(kv_dim, dtype=DTYPE, device="cuda")

    def forward(x, r_attn, cos_sin_head):
        M = x.shape[0]
        cs_q = cos_sin_head.repeat(1, H)                                          # [M,d]      head-local, tiled
        cs_kv = cos_sin_head.repeat(1, H_KV)                                      # [M,kv_dim]
        q = qk[0](x, r_attn, cs_q)[:, inv_q]                                      # [M,d]      rope -> natural
        k = qk[1](x, r_attn, cs_kv)[:, inv_k]                                     # [M,kv_dim]
        v = torch.empty((M, kv_dim), dtype=DTYPE, device="cuda")
        tk_rmsnorm_scale.dispatch(x.to(DTYPE).contiguous(), Wv_t, v, r_attn, g_ones_kv)
        o = _gqa(q, k, v, H)                                                      # [M,d]      GQA causal
        h, r_mlp = _k4(o, Wo_t, x.to(DTYPE).contiguous())                        # out-proj + residual
        g = mlp(h, r_mlp)                                                         # [M,dff]    gate/up+SwiGLU
        x_out, r_attn_next = _k4(g, Wd_t, h)                                      # down-proj + residual
        return x_out, r_attn_next                                                 # caller syncs (test via .item, bench via events)
    return forward


def forward_layer_ref(x, r_attn, cos_sin_head, Wq, Wk, Wv, Wo, gamma_attn, Wgu, Wd, gamma_mlp,
                      n_kv_heads, head_dim=HEAD_DIM):
    """fp32 torch reference mirroring the kernels' order/dtypes: q/k via the rope ref (un-perm'd to
    natural per-head columns), GQA causal attention (scale 1/sqrt(Dh)), then out-proj/MLP/down-proj."""
    d = Wq.shape[0]; kv_dim = Wk.shape[1]; M = x.shape[0]
    Dh = head_dim; H, H_KV = d // Dh, n_kv_heads; G = H // H_KV
    cs_q = cos_sin_head.repeat(1, H); cs_kv = cos_sin_head.repeat(1, H_KV)
    inv_q = torch.argsort(rope_perm(d)).to("cuda"); inv_k = torch.argsort(rope_perm(kv_dim)).to("cuda")
    q = rmsnorm_rope_ref(x, gamma_attn, Wq, r_attn, cs_q)[:, inv_q].reshape(M, H, Dh)
    k = rmsnorm_rope_ref(x, gamma_attn, Wk, r_attn, cs_kv)[:, inv_k].reshape(M, H_KV, Dh)
    Wv_g = (Wv.float() * gamma_attn.float()[:, None]).to(DTYPE).float()
    v = (((x.float() @ Wv_g) * r_attn.float()[:, None]).to(DTYPE)).reshape(M, H_KV, Dh)
    # GQA causal attention in fp32 on the bf16-rounded q,k,v
    ke = k.repeat_interleave(G, dim=1); ve = v.repeat_interleave(G, dim=1)        # [M,H,Dh]
    qh = q.permute(1, 0, 2).float(); kh = ke.permute(1, 0, 2).float(); vh = ve.permute(1, 0, 2).float()
    scale = 1.0 / math.sqrt(Dh)
    scores = torch.bmm(qh, kh.transpose(1, 2)) * scale                           # [H,M,M]
    scores = scores + torch.triu(torch.full((M, M), float("-inf"), device="cuda"), 1)
    o = torch.bmm(torch.softmax(scores, dim=-1), vh).permute(1, 0, 2).reshape(M, d).to(DTYPE)
    h = (x.float() + (o.float() @ Wo.float())).to(DTYPE)                          # out-proj + residual
    r_mlp = _rms_r(h)
    g = rmsnorm_swiglu_ref(h, gamma_mlp, Wgu, r_mlp)                             # [M,dff]
    x_out = (h.float() + (g.float() @ Wd.float())).to(DTYPE)                     # down-proj + residual
    return x_out, _rms_r(x_out)


def test_forward_layer():
    torch.manual_seed(0)
    M, d, dff, Dh = 2048, 4096, 11008, HEAD_DIM
    H_KV = 8
    kv_dim = H_KV * Dh
    mk = lambda *s: torch.randn(*s, device="cuda", dtype=DTYPE)
    w  = lambda r, c: (torch.randn(r, c, device="cuda") * (r ** -0.5)).to(DTYPE)   # scaled init: O(1) q/k -> non-degenerate softmax
    x = mk(M, d); r_attn = _rms_r(x)
    Wq, Wo = w(d, d), w(d, d)
    Wk, Wv = w(d, kv_dim), w(d, kv_dim)
    Wgu, Wd = w(d, 2 * dff), w(dff, d)
    ga, gm = mk(d), mk(d)
    cos_sin_head = make_cos_sin(M, Dh)
    fwd = make_forward_layer(Wq, Wk, Wv, Wo, ga, Wgu, Wd, gm, n_kv_heads=H_KV, head_dim=Dh)
    xo, rn = fwd(x, r_attn, cos_sin_head)
    xo_ref, rn_ref = forward_layer_ref(x, r_attn, cos_sin_head, Wq, Wk, Wv, Wo, ga, Wgu, Wd, gm,
                                       n_kv_heads=H_KV, head_dim=Dh)
    rel = (xo.float() - xo_ref.float()).norm().item() / xo_ref.float().norm().item()
    rel_r = (rn.float() - rn_ref.float()).norm().item() / rn_ref.float().norm().item()
    ok = rel < 2e-2 and rel_r < 2e-2
    print(f"forward_layer GQA (M={M}, d={d}, dff={dff}, H={d//Dh}, H_KV={H_KV}, Dh={Dh})  "
          f"x_out rel={rel:.4f}  r_next rel={rel_r:.4f}  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    test_forward_layer()
