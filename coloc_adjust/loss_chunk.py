"""C2 means 3 (PLAN_overnight3.md): the output layer and the loss over a few tokens at a time, so the
vocabulary-sized logits of a whole micro-batch are never held in GPU memory at once. Same loss, same gradients."""
import torch, torch.nn.functional as F

def chunked_lm_loss(peft_model, ids, chunk, body=None):
    """Next-token cross-entropy (mean over positions) of a causal LM wrapped by peft, computed in
    chunks of `chunk` positions. Returns (surrogate, loss): surrogate.backward() leaves the same
    gradients on the trainable parameters as loss.backward() of the usual computation would; the
    output layer itself is frozen."""
    lm = peft_model.base_model.model
    hs = (body if body is not None else lm.model)(input_ids=ids).last_hidden_state   # body: a compiled lm.model, if any
    h = hs[:, :-1, :].reshape(-1, hs.shape[-1]); y = ids[:, 1:].reshape(-1); n = y.numel()
    hd = h.detach(); g = torch.empty_like(hd); tot = torch.zeros((), device=h.device, dtype=torch.float32)
    for s in range(0, n, chunk):
        x = hd[s:s + chunk].requires_grad_(True)
        l = F.cross_entropy(lm.lm_head(x).float(), y[s:s + chunk], reduction="sum") / n
        g[s:s + chunk] = torch.autograd.grad(l, x)[0]
        tot += l.detach()
    return (h * g).sum(), tot
