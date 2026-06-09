"""epilogue_testlib.py - single source of truth for each epilogue kernel.

Both the correctness harness (test_epilogue.py) and the fusion bench (fusion_win.py)
import EPILOGUES from here, so a kernel is defined ONCE. Adding a fused kernel = one entry.

Self-contained (no external deps beyond torch) so it can ship in the repo.

Each entry:
  module      : the built pybind module name, "tk_<k>"
  args(m,n,k) : tuple of dispatch extra-args (also passed to ref); the default config
  ref(D,out,*args) : the torch reference epilogue -> writes epilogue(D) into `out` (bf16).
                     This ONE function is BOTH the test's correctness oracle AND the bench's
                     unfused second stage, so they can never measure against different math.
  identity(m,n,k) : args that make the epilogue an identity (== noop), for the bit-exact
                    cross-check; None if the epilogue has no identity case.
  sweep(m,n,k) : list of arg-tuples to sweep in the correctness test (default [args]).
  label(args)  : short string for test output.
  hbm_passes   : # of M*N D-transfers the unfused path pays and fusion skips (bench only;
                 2 = D write + D read for a single round-trip; more if the epilogue re-reads D).
"""
import torch

DTYPE = torch.bfloat16
DSIZE = 2          # bytes per bf16 element of the intermediate D
RTOL, ATOL = 1e-2, 1e-1   # project bf16 tolerance vs an fp32-precision reference


# --- tiny tensor helpers (inlined so this module is self-contained / committable) ---
def init_randn(shape, dtype=DTYPE, device="cuda", scale=1):
    return scale * torch.randn(shape, dtype=dtype, device=device)


def init_empty(shape, dtype=DTYPE, device="cuda"):
    return torch.empty(shape, dtype=dtype, device=device)


def _f32(x):
    return torch.full((1,), x, dtype=torch.float32, device="cuda")


EPILOGUES = {
    "noop": {
        "module": "tk_noop",
        "args":     lambda m, n, k: (),
        "ref":      lambda D, out: out.copy_(D),                 # identity (control)
        "identity": None,
        "sweep":    lambda m, n, k: [()],
        "label":    lambda args: "noop",
        "hbm_passes": 2,
    },
    "scale": {
        "module": "tk_scale",
        "args":     lambda m, n, k: (_f32(0.5),),
        "ref":      lambda D, out, alpha: torch.mul(D, alpha.item(), out=out),
        "identity": lambda m, n, k: (_f32(1.0),),                # alpha=1 -> == noop
        "sweep":    lambda m, n, k: [(_f32(a),) for a in (0.0, 1.0, -1.0, 0.5, 1e-3, 1e3)],
        "label":    lambda args: f"a={args[0].item():g}",
        "hbm_passes": 2,
    },
    "k5": {  # Stage-1 RMSNorm scale: precomputed per-row r + per-feature gamma (dummy alpha fills the binding slot)
        "module": "tk_k5",
        "args":     lambda m, n, k: (_f32(1.0), init_randn((m,)), init_randn((n,))),  # r is [1,1,1,M] (1-D, last axis)
        "ref":      lambda D, out, alpha, r, gamma: out.copy_((D.float() * r.float().view(-1, 1) * gamma.float().view(1, -1)).to(DTYPE)),
        "identity": lambda m, n, k: (_f32(1.0), torch.ones((m,), dtype=DTYPE, device="cuda"), torch.ones((n,), dtype=DTYPE, device="cuda")),
        "sweep":    lambda m, n, k: [(_f32(1.0), init_randn((m,)), init_randn((n,)))],  # random r,gamma = [C2] direction guard
        "label":    lambda args: "rms+gamma",
        "hbm_passes": 2,
    },
    "resadd": {  # Stage 2 Task 2.1: residual add  out = (A@B) + residual  ([M,N] skip connection)
        "module": "tk_resadd",
        # binding is positional ([C12b]); residual sits after alpha/r/gamma, so pass throwaway
        # alpha/r/gamma (ignored by ResAddEpilogue) to reach the residual slot.
        "args":     lambda m, n, k: (_f32(1.0), torch.ones((m,), dtype=DTYPE, device="cuda"),
                                     torch.ones((n,), dtype=DTYPE, device="cuda"), init_randn((m, n))),
        "ref":      lambda D, out, alpha, r, gamma, residual: out.copy_((D.float() + residual.float()).to(DTYPE)),
        "identity": lambda m, n, k: (_f32(1.0), torch.ones((m,), dtype=DTYPE, device="cuda"),
                                     torch.ones((n,), dtype=DTYPE, device="cuda"),
                                     torch.zeros((m, n), dtype=DTYPE, device="cuda")),  # residual=0 -> == noop
        "sweep":    lambda m, n, k: [(_f32(1.0), torch.ones((m,), dtype=DTYPE, device="cuda"),
                                      torch.ones((n,), dtype=DTYPE, device="cuda"), init_randn((m, n)))],
        "label":    lambda args: "resadd",
        "hbm_passes": 2,
    },
    "silu": {  # Stage 3.1: SiLU activation  out = silu(A@B) = x * sigmoid(x)
        "module": "tk_silu",
        "args":     lambda m, n, k: (),
        "ref":      lambda D, out: out.copy_((D.float() * torch.sigmoid(D.float())).to(DTYPE)),
        "identity": None,                              # silu has no identity param
        "sweep":    lambda m, n, k: [()],
        "label":    lambda args: "silu",
        "hbm_passes": 2,
    },
    # K5 example (when it lands):
    # "rmsnorm": {
    #     "module": "tk_rmsnorm",
    #     "args":  lambda m,n,k: (init_randn((n,)),),
    #     "ref":   lambda D, out, g: out.copy_((D.float()*torch.rsqrt(D.float().pow(2).mean(-1,keepdim=True)+1e-6)*g).to(DTYPE)),
    #     "identity": None,                 # rmsnorm has no identity param
    #     "sweep": lambda m,n,k: [(init_randn((n,)),)],
    #     "label": lambda args: "rms",
    #     "hbm_passes": 3,                  # unfused re-reads D for the reduction
    # },
}


def make_inputs(m, n, k):
    """A and the pre-transposed Bt (contiguous, kept alive past the async launch)."""
    A = init_randn((m, k))
    B = init_randn((k, n))
    Bt = B.t().contiguous()
    return A, Bt


def gemm_base(noop_mod, A, Bt, m, n):
    """Run the no-op (pure GEMM) kernel -> the bf16 baseline each epilogue is isolated against."""
    C = init_empty((m, n))
    noop_mod.dispatch_micro(A, Bt, C)
    torch.cuda.synchronize()
    return C


def gemm_reference(A, Bt):
    """fp32 ground-truth C = A @ B  (Bt is B transposed, shape [N,K])."""
    return A.float() @ Bt.t().float()


def assert_sane(name, t):
    """Guard against degenerate tensors: finite, non-zero, and not ~constant. Catches dead
    kernels (all-zero C) and degenerate inits that would let isolate-vs-noop pass vacuously."""
    tf = t.float()
    assert torch.isfinite(tf).all(), f"{name}: non-finite values (NaN/Inf)"
    assert tf.abs().max().item() > 0, f"{name}: all zeros"
    assert tf.std().item() > 1e-3, f"{name}: ~constant (std={tf.std().item():.2e}) - degenerate"
