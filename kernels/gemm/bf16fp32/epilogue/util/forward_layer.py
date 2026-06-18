"""forward_layer.py - Milestone 1: a full pre-norm Transformer forward layer assembled ENTIRELY from
the fused GEMM+epilogue kernels (no new kernels -- pure composition), with attention as a torch stub
("minus attn"; Milestone 2 swaps in the HK GQA kernel).

Reparameterization (CODA): RMSNorm(y,gamma) = r * y * gamma, r = 1/rms(y) per row. Because r is
per-row it commutes through the next GEMM, and gamma scales the contracted axis so it folds into the
next weight's ROWS. So each sublayer boundary is:

    producing GEMM emits (residual stream h, partial Sum h^2)   [tk_residual_rms, gamma=ones]
    aux combines the partials -> r = 1/rms(h)                   [tk_aux_rms]
    consuming GEMM applies r in its epilogue (+ activation)     [K7 q/k | K5 v | K6 gate-up]

One layer (input residual stream x, precomputed r_attn = 1/rms(x)):
    q = RoPE(r_attn * (x @ Wq_gfold))          # tk_rmsnorm_rope   (gamma_attn folded into Wq rows)
    k = RoPE(r_attn * (x @ Wk_gfold))          # tk_rmsnorm_rope
    v =       r_attn * (x @ Wv_gfold)          # tk_rmsnorm_scale  (no RoPE)
    o = attention(q, k, v)                     # TORCH stub (M1)   -> [M, n*hd]
    h = x + o @ Wo ; partials = Sum h^2        # tk_residual_rms   (gamma=ones; gamma_mlp folds into Wgu)
    r_mlp = 1/rms(h)                           # tk_aux_rms
    g = SwiGLU(r_mlp * (h @ Wgu_gfold))        # tk_rmsnorm_swiglu (gamma_mlp folded into Wgu rows)
    x_out = h + g @ Wd ; partials = Sum x_out^2  # tk_residual_rms (gamma=ones) -> next layer's r_attn
    r_attn_next = 1/rms(x_out)                 # tk_aux_rms

Shapes (single-head, batch=1): x[M,d], Wq/Wk/Wv[d,d], Wo[d,d], Wgu[d,2*dff], Wd[dff,d], gammas[d].
Constraints inherited from the GEMM: M,d,2*dff,dff % 256 ; d % 128.
"""
import math, torch
import tk_residual_rms, tk_aux_rms, tk_rmsnorm_scale
from swiglu import make_rmsnorm_swiglu, rmsnorm_swiglu_ref
from rope import make_rmsnorm_rope, rmsnorm_rope_ref, make_cos_sin, rope_perm

DTYPE = torch.bfloat16
EPS = 1e-5
REG_BLOCK_N = 64


def _rms_r(y):  # per-row inv-rms (bf16), the value the prior GEMM's aux would emit
    return torch.rsqrt(y.float().pow(2).mean(-1) + EPS).to(DTYPE)


def _k4(prev, Wt, residual):
    """tk_residual_rms with gamma=ones: c = prev@W + residual (the residual stream), partials = Sum c^2;
    returns (c, r=1/rms(c)). gamma=ones because the next norm's gamma folds into the next weight."""
    M, N = residual.shape
    c = torch.empty((M, N), dtype=DTYPE, device="cuda")
    save = torch.empty((M, N), dtype=DTYPE, device="cuda")
    partials = torch.empty((N // REG_BLOCK_N, M), dtype=torch.float32, device="cuda")
    g_ones = torch.ones(N, dtype=DTYPE, device="cuda")
    tk_residual_rms.dispatch(prev, Wt, c, residual, g_ones, partials, save)
    r = torch.empty(M, dtype=DTYPE, device="cuda")
    tk_aux_rms.reduce(partials, r)
    return c, r


def make_forward_layer(Wq, Wk, Wv, Wo, gamma_attn, Wgu, Wd, gamma_mlp):
    """Build a forward layer once (fold gammas, transpose weights). Returns
    forward(x, r_attn, cos_sin) -> (x_out, r_attn_next).  Single-head, batch=1, causal."""
    d = Wq.shape[0]
    qk = (make_rmsnorm_rope(Wq, gamma_attn), make_rmsnorm_rope(Wk, gamma_attn))   # K7 (RoPE) for q,k
    Wv_t = (Wv.to("cuda", DTYPE).float() * gamma_attn.to("cuda", DTYPE).float()[:, None]).to(DTYPE).t().contiguous()
    Wo_t = Wo.to("cuda", DTYPE).t().contiguous()
    mlp = make_rmsnorm_swiglu(Wgu, gamma_mlp)                                     # K6 (gate/up + SwiGLU)
    Wd_t = Wd.to("cuda", DTYPE).t().contiguous()
    g_ones_d = torch.ones(d, dtype=DTYPE, device="cuda")
    scale = 1.0 / math.sqrt(d)

    def forward(x, r_attn, cos_sin):
        M = x.shape[0]
        q = qk[0](x, r_attn, cos_sin)                                            # [M,d]  K7
        k = qk[1](x, r_attn, cos_sin)                                            # [M,d]  K7
        v = torch.empty((M, d), dtype=DTYPE, device="cuda")
        tk_rmsnorm_scale.dispatch(x.to(DTYPE).contiguous(), Wv_t, v, r_attn, g_ones_d)   # K5: r*(x@Wv_gfold)
        # --- attention: torch stub (M1 "minus attn"); q,k are in rope_perm column order, consistent ---
        att = torch.softmax((q.float() @ k.float().t()) * scale
                            + torch.triu(torch.full((M, M), float("-inf"), device="cuda"), 1), dim=-1)
        o = (att @ v.float()).to(DTYPE)                                          # [M,d]
        h, r_mlp = _k4(o, Wo_t, x.to(DTYPE).contiguous())                        # K4 out-proj + residual
        g = mlp(h, r_mlp)                                                        # [M,dff]  K6
        x_out, r_attn_next = _k4(g, Wd_t, h)                                     # K4 down-proj + residual
        torch.cuda.synchronize()
        return x_out, r_attn_next
    return forward


def forward_layer_ref(x, r_attn, cos_sin, Wq, Wk, Wv, Wo, gamma_attn, Wgu, Wd, gamma_mlp):
    """fp32 torch reference for the whole layer, mirroring the kernels' order/dtypes and the SAME torch
    attention. q,k are produced in rope_perm order (as the kernel stores them) so Q*K^T matches."""
    d = Wq.shape[0]
    perm = rope_perm(d)
    q = rmsnorm_rope_ref(x, gamma_attn, Wq, r_attn, cos_sin)                     # already rope_perm'd
    k = rmsnorm_rope_ref(x, gamma_attn, Wk, r_attn, cos_sin)
    Wv_g = (Wv.float() * gamma_attn.float()[:, None]).to(DTYPE).float()
    v = ((x.float() @ Wv_g) * r_attn.float()[:, None]).to(DTYPE)
    M = x.shape[0]; scale = 1.0 / math.sqrt(d)
    att = torch.softmax((q.float() @ k.float().t()) * scale
                        + torch.triu(torch.full((M, M), float("-inf"), device="cuda"), 1), dim=-1)
    o = (att @ v.float()).to(DTYPE)
    h = (x.float() + (o.float() @ Wo.float())).to(DTYPE)                         # out-proj + residual
    r_mlp = _rms_r(h)
    g = rmsnorm_swiglu_ref(h, gamma_mlp, Wgu, r_mlp)                             # [M,dff]
    x_out = (h.float() + (g.float() @ Wd.float())).to(DTYPE)                     # down-proj + residual
    return x_out, _rms_r(x_out)


def test_forward_layer():
    torch.manual_seed(0)
    M, d, dff = 2048, 4096, 11008
    mk = lambda *s: torch.randn(*s, device="cuda", dtype=DTYPE)
    w  = lambda r, c: (torch.randn(r, c, device="cuda") * (r ** -0.5)).to(DTYPE)   # scaled init: O(1) q/k -> non-degenerate softmax (unscaled randn -> ~O(d) logits -> argmax flips on bf16)
    x = mk(M, d); r_attn = _rms_r(x)
    Wq, Wk, Wv, Wo = w(d, d), w(d, d), w(d, d), w(d, d)
    Wgu, Wd = w(d, 2 * dff), w(dff, d)
    ga, gm = mk(d), mk(d)
    cos_sin = make_cos_sin(M, d)
    fwd = make_forward_layer(Wq, Wk, Wv, Wo, ga, Wgu, Wd, gm)
    xo, rn = fwd(x, r_attn, cos_sin)
    xo_ref, rn_ref = forward_layer_ref(x, r_attn, cos_sin, Wq, Wk, Wv, Wo, ga, Wgu, Wd, gm)
    rel = (xo.float() - xo_ref.float()).norm().item() / xo_ref.float().norm().item()
    rel_r = (rn.float() - rn_ref.float()).norm().item() / rn_ref.float().norm().item()
    ok = rel < 2e-2 and rel_r < 2e-2
    print(f"forward_layer (M={M}, d={d}, dff={dff})  x_out rel={rel:.4f}  r_next rel={rel_r:.4f}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    test_forward_layer()
