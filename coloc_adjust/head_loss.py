"""The tenant's output layer and loss, written out instead of left to the model's own forward.

Measured on Nov 21 2026 (segment_bytes.py): of the 57 GB the usual tenant moves to and from GPU memory per
step, 56 % belongs to the output layer and the loss, in four stretches of 8 GB that cannot be paused. The
logits of one micro-batch are 1,024 positions x 151,936 vocabulary entries (311 MB in bf16, 622 MB in
float32) and are read and written about a dozen times.

  split_lm_loss   the same operators as the model's own loss, with a checkpoint between them (forward: a
                  direct call; backward: a tensor hook). Same kernels, same loss, same gradients; the 8 GB
                  stretch becomes eight pieces.
  vocab_lm_loss   the vocabulary in blocks of `vb` rows, two passes over the output layer's weight. Pass 1
                  sums exp(logit) per position and picks the target's logit; pass 2 recomputes the block's
                  logits, turns them into probabilities and adds their product with the weight block to the
                  gradient w.r.t. the hidden states. The logits of a block are dropped after use, so the
                  full logits never exist; every block is a place to pause. Costs one more matrix product
                  than the usual way. Same loss and gradients up to rounding (the sum over the vocabulary is
                  kept in float32 per block instead of in one bf16 product).

Both return what ft_worker's train_step needs; `ckpt(label)` is ft_worker's checkpoint (pacing, gate, mark).
The output layer's weight is frozen (LoRA tenant), which vocab_lm_loss relies on."""
import torch, torch.nn.functional as F

def _hidden(peft_model, ids, body):
    lm = peft_model.base_model.model
    hs = (body if body is not None else lm.model)(input_ids=ids).last_hidden_state   # body: a compiled lm.model, if any
    y = F.pad(ids, (0, 1), value=-100)[..., 1:].reshape(-1)   # as transformers' ForCausalLMLoss: position t predicts token t+1, the last one is ignored
    return lm, hs.reshape(-1, hs.shape[-1]), y

def split_lm_loss(peft_model, ids, ckpt, body=None):
    """Returns the loss (mean over positions); loss.backward() as usual."""
    lm, h, y = _hidden(peft_model, ids, body)
    ckpt("head:linear")
    logits = lm.lm_head(h); logits.register_hook(lambda g: ckpt("head:linear_bwd"))        # the hook runs when the gradient w.r.t. this tensor is ready,
    ckpt("head:cast")                                                                      # i.e. just before the backward of the operator that made it
    lf = logits.float(); lf.register_hook(lambda g: ckpt("head:cast_bwd"))
    ckpt("head:logsoftmax")
    lp = F.log_softmax(lf, dim=-1); lp.register_hook(lambda g: ckpt("head:logsoftmax_bwd"))
    ckpt("head:nll")
    return F.nll_loss(lp, y, ignore_index=-100)

_FUSED = {}
def _fused_block_fns():
    """R1 of the plan of Nov 21: the operators of one vocabulary block, compiled into few kernels. Everything between the
    two matrix products (float32 cast, exp, the sums over the block, the one-hot of the target) is element-wise or a
    reduction over the block, so torch.compile can emit it as one generated kernel that keeps its intermediates in
    registers; the matrix products stay library calls. The same two functions serve every block (the last, shorter block
    is one more compilation)."""
    if not _FUSED:
        def p1(hd, Wb, lse, idx, inbf):
            lb = F.linear(hd, Wb).float()
            hot = (torch.arange(lb.shape[1], device=lb.device)[None, :] == idx[:, None])
            return torch.logaddexp(lse, torch.logsumexp(lb, dim=1)), (lb * hot).sum(dim=1) * inbf
        def p2(hd, Wb, lse, idx, inbf, coef):
            lb = F.linear(hd, Wb).float()
            hot = (torch.arange(lb.shape[1], device=lb.device)[None, :] == idx[:, None])
            p = (torch.exp(lb - lse[:, None]) - hot * inbf[:, None]) * coef
            return (p.to(hd.dtype) @ Wb).float()
        _FUSED["p1"], _FUSED["p2"] = torch.compile(p1), torch.compile(p2)
    return _FUSED["p1"], _FUSED["p2"]

def vocab_lm_loss(peft_model, ids, vb, ckpt, body=None, fused=False):
    """Returns (surrogate, loss): surrogate.backward() leaves on the trainable parameters the gradients that
    loss.backward() of the usual computation would; loss is a detached float32 scalar.
    fused: the operators of a block run as compiled kernels (_fused_block_fns)."""
    lm, h, y = _hidden(peft_model, ids, body)
    if fused:
        p1, p2 = _fused_block_fns(); W = lm.lm_head.weight; V = W.shape[0]; hd = h.detach(); N = hd.shape[0]
        valid = y >= 0; n = valid.sum().clamp(min=1).float(); yc = y.clamp(min=0); vf = valid.float(); coef = (vf / n)[:, None]
        lse = torch.full((N,), float("-inf"), device=hd.device, dtype=torch.float32); tgt = torch.zeros(N, device=hd.device, dtype=torch.float32)
        with torch.no_grad():
            for b in range(0, V, vb):
                ckpt("head:pass1")
                idx = yc - b; inbf = ((idx >= 0) & (idx < min(vb, V - b))).float() * vf
                lse, t = p1(hd, W[b:b + vb], lse, idx, inbf); tgt = tgt + t
            loss = ((lse - tgt) * vf).sum() / n
            g = torch.zeros(hd.shape, device=hd.device, dtype=torch.float32)
            for b in range(0, V, vb):
                ckpt("head:pass2")
                idx = yc - b; inbf = ((idx >= 0) & (idx < min(vb, V - b))).float() * vf
                g = g + p2(hd, W[b:b + vb], lse, idx, inbf, coef)
        ckpt("head:surrogate")
        return (h * g.to(h.dtype)).sum(), loss
    W = lm.lm_head.weight; V = W.shape[0]; hd = h.detach(); N = hd.shape[0]
    valid = y >= 0; n = valid.sum().clamp(min=1).float(); yc = y.clamp(min=0)
    lse = torch.full((N,), float("-inf"), device=hd.device, dtype=torch.float32); tgt = torch.zeros(N, device=hd.device, dtype=torch.float32)
    for b in range(0, V, vb):                                   # pass 1: log of the sum of exp(logit) per position, and the target's logit
        ckpt("head:pass1")
        lb = F.linear(hd, W[b:b + vb]).float(); w = lb.shape[1]
        lse = torch.logaddexp(lse, torch.logsumexp(lb, dim=1))
        idx = yc - b; inb = valid & (idx >= 0) & (idx < w)
        tgt += lb.gather(1, idx.clamp(0, w - 1)[:, None])[:, 0] * inb
    loss = ((lse - tgt) * valid).sum() / n
    g = torch.zeros(hd.shape, device=hd.device, dtype=torch.float32); coef = (valid / n)[:, None]
    for b in range(0, V, vb):                                   # pass 2: (softmax - one_hot) / n, times the weight block
        ckpt("head:pass2")
        Wb = W[b:b + vb]; p = F.linear(hd, Wb).float(); w = p.shape[1]
        p.sub_(lse[:, None]).exp_()
        idx = yc - b; inb = valid & (idx >= 0) & (idx < w)
        p.scatter_add_(1, idx.clamp(0, w - 1)[:, None], -inb.float()[:, None])
        p.mul_(coef)
        g += (p.to(hd.dtype) @ Wb).float()
    ckpt("head:surrogate")
    return (h * g.to(h.dtype)).sum(), loss
