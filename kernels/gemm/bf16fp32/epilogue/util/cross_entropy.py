"""cross_entropy.py - forward fused cross-entropy helpers (make_ce and make_ce_rms).

The kernel computes logits = h @ W_vocab (a GEMM) and, in the epilogue, emits only per-(row,
64-col-group) softmax partials (per-group max + sum-of-exp) -- the full [M, vocab] logits are NEVER
written to HBM. A tiny aux kernel (cross_entropy_reduce) combines the N/REG_BLOCK_N groups per row
(max-correction) and adds the target logit as the O(K) dot <h[row], Wt[label[row]]>, yielding the
per-row loss = logsumexp(logits) - logits[row, label[row]].

W_vocab is the natural [d_model, vocab] projection; like make_swiglu it is transposed to [vocab,
d_model] (the kernel's b operand [N, K]) once at setup."""
import torch

REG_BLOCK_N = 64   # BLOCK_SIZE / WARPS_N = 256/4; single source: epilogue_args.cuh
DTYPE = torch.bfloat16


def ce_ref(h, W_vocab, labels):
    """fp32 oracle: per-row cross-entropy loss (reduction='none') of (h @ W_vocab) vs labels."""
    logits = h.float() @ W_vocab.float()                       # [M, vocab]
    return torch.nn.functional.cross_entropy(logits, labels, reduction="none")


def make_ce(W_vocab):
    """Prepare the vocab projection once: transpose W_vocab [d_model, vocab] -> [vocab, d_model] (the
    kernel b operand). Returns forward(h, labels) -> per-row loss [M]. Rebuild if W_vocab changes."""
    import tk_cross_entropy, tk_ce_reduce
    N = W_vocab.shape[1]                                        # vocab
    assert N % 256 == 0, f"vocab N={N} must be a multiple of 256"
    Wt = W_vocab.to(device="cuda", dtype=DTYPE).t().contiguous()   # [vocab, d_model]
    groups = N // REG_BLOCK_N

    def forward(h, labels):
        h = h.to(device="cuda", dtype=DTYPE).contiguous()
        M = h.shape[0]
        labels = labels.to(device="cuda", dtype=torch.float32).contiguous()       # fp32 (exact: vocab << 2^24)
        max_buf    = torch.empty((groups, M), dtype=torch.float32, device="cuda")
        sumexp_buf = torch.empty((groups, M), dtype=torch.float32, device="cuda")
        loss       = torch.empty((M,),        dtype=torch.float32, device="cuda")
        tk_cross_entropy.dispatch(h, Wt, max_buf, sumexp_buf)                      # softmax partials only
        tk_ce_reduce.reduce(max_buf, sumexp_buf, h, Wt, labels, loss)             # +O(K) target dot
        torch.cuda.synchronize()
        return loss
    return forward


RMS_EPS = 1e-5


def ce_rms_ref(h, gamma, W_lm, labels):
    """fp32 oracle for the RMS path: per-row cross-entropy of rmsnorm(h,gamma) @ W_lm vs labels, where
    rmsnorm(h,gamma) = h * rsqrt(mean(h^2,-1,keepdim)+1e-5) * gamma."""
    n = h.float() * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + RMS_EPS) * gamma.float()
    logits = n @ W_lm.float()                                   # [M, vocab]
    return torch.nn.functional.cross_entropy(logits, labels, reduction="none")


def make_ce_rms(W_lm, gamma):
    """Prepare the RMS->cross-entropy vocab projection once: fold the norm's per-d_model `gamma` into W_lm's ROWS (the
    contraction axis), then transpose [d_model, vocab] -> [vocab, d_model] (the kernel b operand).
    Returns forward(h, labels) -> per-row loss [M]; r (per-row inv-rms) is computed from h each call.
    Rebuild if W_lm/gamma change. (Mirrors make_rmsnorm_swiglu's gamma-fold + r.)"""
    import tk_ce_rms, tk_ce_reduce
    N = W_lm.shape[1]                                           # vocab
    assert N % 256 == 0, f"vocab N={N} must be a multiple of 256"
    Wg = (W_lm.to(device="cuda", dtype=DTYPE).float()
          * gamma.to(device="cuda", dtype=DTYPE).float()[:, None]).to(DTYPE)   # bf16-rounded gamma fold
    Wt = Wg.t().contiguous()                                    # [vocab, d_model]
    groups = N // REG_BLOCK_N

    def forward(h, labels):
        h = h.to(device="cuda", dtype=DTYPE).contiguous()
        M = h.shape[0]
        r = torch.rsqrt(h.float().pow(2).mean(-1) + RMS_EPS).to(DTYPE).contiguous()  # per-row inv-rms (bf16)
        labels = labels.to(device="cuda", dtype=torch.float32).contiguous()          # fp32 (exact: vocab << 2^24)
        max_buf    = torch.empty((groups, M), dtype=torch.float32, device="cuda")
        sumexp_buf = torch.empty((groups, M), dtype=torch.float32, device="cuda")
        loss       = torch.empty((M,),        dtype=torch.float32, device="cuda")
        tk_ce_rms.dispatch(h, Wt, max_buf, sumexp_buf, r)                            # r-scaled softmax partials
        tk_ce_reduce.reduce_rms(max_buf, sumexp_buf, h, Wt, labels, r, loss)         # +O(K) r-scaled target dot
        torch.cuda.synchronize()
        return loss
    return forward
