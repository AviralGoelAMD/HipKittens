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
