"""Goals 2 and 3: one pre-norm GQA Transformer forward layer, HK vs PyTorch eager vs torch.compile.

--phases  each of the five phases timed on its own, on intermediates captured from a full run of the same
          implementation. Each torch phase is compiled separately.

Phases (HK launches): qkv (3), attention (1), out_proj (2), gate_up (1), down (2).
HK splits RMSNorm across phases: the producer emits r = 1/rms, the consumer applies it. Torch normalizes in
the consumer and computes r_next at the end of the layer, so each side does each RMS reduction once. The
HK layer's input r_attn = 1/rms(x) is computed once, outside timing (a previous layer would emit it).
Per-phase times therefore do not line up op-for-op; the phase sums do.

Build first: bench/build.sh (the epilogue modules in MODULES and the attention kernel for each M).
"""
import dataclasses

import torch
import torch._dynamo
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import common
import hk

MODULES = ["rmsnorm_rope", "rmsnorm_scale", "residual_rms", "rms_reduce", "rmsnorm_swiglu"]
HK_LAUNCHES = dict(qkv=3, attention=1, out_proj=2, gate_up=1, down=2)
COMPILE = dict(mode="max-autotune-no-cudagraphs", dynamic=False)


def choose_sdpa(H, H_KV, Dh):
    """Pick the strongest fused SDPA path: flash before memory-efficient, native GQA before expanded KV.
    Disables the math backend globally so torch never falls back to it. Returns (backend name, native GQA)."""
    q = torch.randn(1, H, 256, Dh, device="cuda", dtype=common.BF16)
    kv = torch.randn(1, H_KV, 256, Dh, device="cuda", dtype=common.BF16)
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
        for native in (True, False):
            k = kv if native else kv.repeat_interleave(H // H_KV, dim=1)
            try:
                with sdpa_kernel(backend):
                    F.scaled_dot_product_attention(q, k, k, is_causal=True, enable_gqa=native)
                torch.cuda.synchronize()
            except RuntimeError:
                continue
            torch.backends.cuda.enable_flash_sdp(backend == SDPBackend.FLASH_ATTENTION)
            torch.backends.cuda.enable_mem_efficient_sdp(backend == SDPBackend.EFFICIENT_ATTENTION)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)
            return backend.name, native
    raise RuntimeError("no fused SDPA backend is available; refusing the math fallback")


def load_gqa(cfg, M):
    """The gqa_causal build for this shape. build.sh gives every build its own module name; Python returns the
    first-loaded module for any later load of the same name, so a shared name would run the wrong shape."""
    name = f"tk_attn_H{cfg['H']}_KV{cfg['H_KV']}_D{cfg['Dh']}_N{M}"
    hint = f"{common.BENCH / 'build.sh'} --config <config> --M {M}"
    return common.load_extension(common.find_extension(common.BENCH / "attn_build", name, hint), name=name)


class Layer:
    """Weights, prepared HK operands, and the five phases of both implementations at one M."""

    def __init__(self, cfg, M, gqa, native_gqa):
        self.M, self.d, self.d_ff = M, cfg["d"], cfg["d_ff"]
        self.H, self.H_KV, self.Dh = cfg["H"], cfg["H_KV"], cfg["Dh"]
        d, kv = self.d, self.H_KV * self.Dh
        self.W = dict(q=common.weight(d, d), k=common.weight(d, kv), v=common.weight(d, kv), o=common.weight(d, d),
                      gu=common.weight(d, 2 * self.d_ff), dn=common.weight(self.d_ff, d))
        self.g_attn, self.g_mlp = common.gamma(d), common.gamma(d)
        self.x = common.activation(M, d)
        self.cs = common.rope_table(M, self.Dh)                     # one head's table [M, Dh]
        W, ga, gm = self.W, self.g_attn, self.g_mlp
        self.P = dict(q=hk.prepare(W["q"], layout="rope", gamma=ga), k=hk.prepare(W["k"], layout="rope", gamma=ga),
                      v=hk.prepare(W["v"], gamma=ga), o=hk.prepare(W["o"]),
                      gu=hk.prepare(W["gu"], layout="swiglu", gamma=gm), dn=hk.prepare(W["dn"]))
        self.cs_q = hk.prepare_rope_table(self.cs.repeat(1, self.H))
        self.cs_k = hk.prepare_rope_table(self.cs.repeat(1, self.H_KV))
        self.r_attn = common.inv_rms(self.x)
        self.gqa, self.native_gqa = gqa, native_gqa

    # ---- HK phases ----
    def hk_qkv(self, x, r):
        q = hk.matmul(x, self.P["q"], "rmsnorm_rope", r=r, cos_sin=self.cs_q)
        k = hk.matmul(x, self.P["k"], "rmsnorm_rope", r=r, cos_sin=self.cs_k)
        v = hk.matmul(x, self.P["v"], "rmsnorm", r=r)
        return q, k, v

    def hk_attention(self, q, k, v):
        M, H, H_KV, Dh = self.M, self.H, self.H_KV, self.Dh
        o = torch.empty(1, M, H, Dh, dtype=common.BF16, device="cuda")
        lse = torch.empty(1, H, 1, M, dtype=common.FP32, device="cuda")
        self.gqa.dispatch_micro(q.view(1, M, H, Dh), k.view(1, M, H_KV, Dh), v.view(1, M, H_KV, Dh), o, lse)
        return o.view(M, H * Dh)

    def hk_out_proj(self, o, x):
        return hk.residual_rms(o, self.P["o"], x)                   # h, r_mlp

    def hk_gate_up(self, h, r):
        return hk.matmul(h, self.P["gu"], "rmsnorm_swiglu", r=r)

    def hk_down(self, g, h):
        return hk.residual_rms(g, self.P["dn"], h)                  # x_out, r_next

    # ---- torch phases (eager and torch.compile run this same code) ----
    def t_qkv(self, x):
        M, Dh, W = self.M, self.Dh, self.W
        xn = F.rms_norm(x, (self.d,), self.g_attn, common.EPS)
        cs = self.cs.view(M, 1, Dh)
        q = common.rope((xn @ W["q"]).view(M, self.H, Dh), cs)
        k = common.rope((xn @ W["k"]).view(M, self.H_KV, Dh), cs)
        v = (xn @ W["v"]).view(M, self.H_KV, Dh)
        return q, k, v

    def t_attention(self, q, k, v):
        qh, kh, vh = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))        # [1, heads, M, Dh]
        if not self.native_gqa:
            G = self.H // self.H_KV
            kh, vh = kh.repeat_interleave(G, dim=1), vh.repeat_interleave(G, dim=1)
        o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=self.native_gqa)
        return o.squeeze(0).transpose(0, 1).reshape(self.M, self.d)

    def t_out_proj(self, o, x):
        return x + o @ self.W["o"]

    def t_gate_up(self, h):
        gu = F.rms_norm(h, (self.d,), self.g_mlp, common.EPS) @ self.W["gu"]
        return F.silu(gu[:, :self.d_ff]) * gu[:, self.d_ff:]

    def t_down(self, g, h):
        x_out = h + g @ self.W["dn"]
        r_next = torch.rsqrt(torch.linalg.vector_norm(x_out, dim=-1, dtype=torch.float32).square() / self.d + common.EPS)
        return x_out, r_next

    # ---- fp32 references, applied to either implementation's captured inputs ----
    def ref_qkv(self, x):
        M, Dh, W = self.M, self.Dh, self.W
        xn = x.float() * common.inv_rms(x)[:, None] * self.g_attn.float()
        cs = self.cs.float().view(M, 1, Dh)
        q = common.rope((xn @ W["q"].float()).view(M, self.H, Dh), cs).reshape(M, -1)
        k = common.rope((xn @ W["k"].float()).view(M, self.H_KV, Dh), cs).reshape(M, -1)
        return q, k, xn @ W["v"].float()

    def ref_attention(self, q, k, v):
        M, H, H_KV, Dh = self.M, self.H, self.H_KV, self.Dh
        G = H // H_KV
        qh = q.float().reshape(M, H, Dh).transpose(0, 1)                                   # [H, M, Dh]
        kh = k.float().reshape(M, H_KV, Dh).transpose(0, 1).repeat_interleave(G, dim=0)
        vh = v.float().reshape(M, H_KV, Dh).transpose(0, 1).repeat_interleave(G, dim=0)
        s = (qh @ kh.transpose(1, 2)) * Dh ** -0.5
        s.masked_fill_(torch.ones(M, M, dtype=torch.bool, device="cuda").triu(1), float("-inf"))
        return (torch.softmax(s, dim=-1) @ vh).transpose(0, 1).reshape(M, H * Dh)

    def ref_out_proj(self, o, x):
        h = x.float() + o.float() @ self.W["o"].float()
        return h, common.inv_rms(h)

    def ref_gate_up(self, h):
        hn = h.float() * common.inv_rms(h)[:, None] * self.g_mlp.float()
        gu = hn @ self.W["gu"].float()
        return F.silu(gu[:, :self.d_ff]) * gu[:, self.d_ff:]

    def ref_down(self, g, h):
        x_out = h.float() + g.float() @ self.W["dn"].float()
        return x_out, common.inv_rms(x_out)

    # ---- captured intermediates from one full run of each implementation ----
    def capture_hk(self):
        x, r = self.x, self.r_attn
        q, k, v = self.hk_qkv(x, r)
        o = self.hk_attention(q, k, v)
        h, r_mlp = self.hk_out_proj(o, x)
        g = self.hk_gate_up(h, r_mlp)
        return dict(x=x, r=r, q=q, k=k, v=v, o=o, h=h, r_mlp=r_mlp, g=g)

    def capture_torch(self):
        x = self.x
        q, k, v = self.t_qkv(x)
        o = self.t_attention(q, k, v)
        h = self.t_out_proj(o, x)
        g = self.t_gate_up(h)
        return dict(x=x, q=q, k=k, v=v, o=o, h=h, g=g)


@dataclasses.dataclass
class Phase:
    hk: object            # () -> HK outputs on HK-captured inputs
    torch_fn: object      # torch phase function
    torch_args: tuple     # its torch-captured inputs
    ref_hk: object        # () -> fp32 reference for the HK inputs
    ref_torch: object     # () -> fp32 reference for the torch inputs
    tol_hk: tuple
    tol_torch: tuple


def phases(L, a, b):
    S, V = common.STAGE_TOL, common.VECTOR_TOL
    return {
        "qkv": Phase(lambda: L.hk_qkv(a["x"], a["r"]), L.t_qkv, (b["x"],),
                     lambda: L.ref_qkv(a["x"]), lambda: L.ref_qkv(b["x"]), (S, S, S), (S, S, S)),
        "attention": Phase(lambda: L.hk_attention(a["q"], a["k"], a["v"]), L.t_attention, (b["q"], b["k"], b["v"]),
                           lambda: L.ref_attention(a["q"], a["k"], a["v"]),
                           lambda: L.ref_attention(b["q"], b["k"], b["v"]), (S,), (S,)),
        "out_proj": Phase(lambda: L.hk_out_proj(a["o"], a["x"]), L.t_out_proj, (b["o"], b["x"]),
                          lambda: L.ref_out_proj(a["o"], a["x"]), lambda: L.ref_out_proj(b["o"], b["x"])[:1],
                          (S, V), (S,)),
        "gate_up": Phase(lambda: L.hk_gate_up(a["h"], a["r_mlp"]), L.t_gate_up, (b["h"],),
                         lambda: L.ref_gate_up(a["h"]), lambda: L.ref_gate_up(b["h"]), (S,), (S,)),
        "down": Phase(lambda: L.hk_down(a["g"], a["h"]), L.t_down, (b["g"], b["h"]),
                      lambda: L.ref_down(a["g"], a["h"]), lambda: L.ref_down(b["g"], b["h"]), (S, V), (S, V)),
    }


def run_phases(L, timer):
    a, b = L.capture_hk(), L.capture_torch()
    rows = []
    for name, p in phases(L, a, b).items():
        compiled = torch.compile(p.torch_fn, **COMPILE)
        eager = lambda: p.torch_fn(*p.torch_args)
        comp = lambda: compiled(*p.torch_args)
        ok_hk, rel = common.check(p.hk(), p.ref_hk(), p.tol_hk)
        ref_t = p.ref_torch()
        ok_eager, _ = common.check(eager(), ref_t, p.tol_torch)
        ok_comp, _ = common.check(comp(), ref_t, p.tol_torch)     # first call compiles + autotunes, untimed
        ok = ok_hk and ok_eager and ok_comp
        row = dict(M=L.M, phase=name, hk_launches=HK_LAUNCHES[name], rel=rel, status="PASS" if ok else "FAIL")
        if ok:
            t_hk, t_eager, t_comp = timer(p.hk), timer(eager), timer(comp)
            row.update(**common.ms_fields("hk", t_hk), **common.ms_fields("eager", t_eager),
                       **common.ms_fields("compile", t_comp),
                       eager_x=t_eager.median / t_hk.median, compile_x=t_comp.median / t_hk.median)
        rows.append(row)
        print(f"  M={L.M} {name}: {row['status']}", flush=True)
    rows.append(sum_row(L.M, rows))
    return rows


def sum_row(M, rows):
    ok = all(r["status"] == "PASS" for r in rows)
    row = dict(M=M, phase="sum", hk_launches=sum(r["hk_launches"] for r in rows), status="PASS" if ok else "FAIL")
    if ok:
        for arm in ("hk", "eager", "compile"):
            row[f"{arm}_ms"] = sum(r[f"{arm}_ms"] for r in rows)
        row["eager_x"] = row["eager_ms"] / row["hk_ms"]
        row["compile_x"] = row["compile_ms"] / row["hk_ms"]
    return row


PHASE_COLUMNS = [("M", "M", "d"), ("phase", "phase", ""), ("HK launches", "hk_launches", "d"),
                 ("HK ms", "hk_ms", ".3f"), ("eager ms", "eager_ms", ".3f"), ("compile ms", "compile_ms", ".3f"),
                 ("eager/HK", "eager_x", ".2f"), ("compile/HK", "compile_x", ".2f"), ("rel err", "rel", ".1e"),
                 ("status", "status", "")]


def main():
    args = common.parse_args("Goals 2-3: forward layer, HK vs eager vs torch.compile",
                             add=lambda ap: ap.add_argument("--phases", action="store_true"))
    if not args.phases:
        raise SystemExit("choose --phases")
    common.require_epilogue_modules(MODULES)
    torch.manual_seed(0)
    torch._dynamo.config.cache_size_limit = 64
    cfg = args.cfg
    backend, native = choose_sdpa(cfg["H"], cfg["H_KV"], cfg["Dh"])
    print(f"SDPA baseline: {backend}, native GQA: {native}")
    timer = common.Timer(args.warmup, args.iters)
    rows = []
    for M in args.M:
        L = Layer(cfg, M, load_gqa(cfg, M), native)
        rows += run_phases(L, timer)
        del L
        torch.cuda.empty_cache()
    common.print_table(f"Layer phases ({args.config}, cold cache; SDPA {backend}, native GQA {native})",
                       PHASE_COLUMNS, rows, args.markdown)
    if args.json:
        common.write_json(args.json, "bench_layer --phases", args.config, rows,
                          extra=dict(sdpa_backend=backend, native_gqa=native))
    raise SystemExit(common.exit_code(rows))


if __name__ == "__main__":
    main()
