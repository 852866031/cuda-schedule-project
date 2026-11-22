"""CPU check of head_loss.py against the model's own loss (transformers' ForCausalLMLoss): loss and the
gradients on the body's parameters, for a small random model in float32.   python test_head_loss.py"""
import types, torch, torch.nn as nn, torch.nn.functional as F
from head_loss import split_lm_loss, vocab_lm_loss

class Body(nn.Module):
    def __init__(s, V, H):
        super().__init__(); s.emb = nn.Embedding(V, H); s.l1 = nn.Linear(H, H); s.l2 = nn.Linear(H, H)
    def forward(s, input_ids): return types.SimpleNamespace(last_hidden_state=s.l2(torch.tanh(s.l1(s.emb(input_ids)))))
def make(V=1003, H=24, seed=0):
    torch.manual_seed(seed); lm = types.SimpleNamespace(model=Body(V, H), lm_head=nn.Linear(H, V, bias=False))
    lm.lm_head.weight.requires_grad_(False); lm.lm_head.weight.mul_(3.0)       # sharper distribution than the default init
    return types.SimpleNamespace(base_model=types.SimpleNamespace(model=lm)), lm
def grads(lm): return torch.cat([p.grad.flatten() for p in lm.model.parameters()])
def reference(lm, ids):
    logits = lm.lm_head(lm.model(input_ids=ids).last_hidden_state).float()
    y = F.pad(ids, (0, 1), value=-100)[..., 1:]
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)

calls = []
def ckpt(label): calls.append(label)
pm, lm = make(); ids = torch.randint(0, 1003, (3, 17))
ref = reference(lm, ids); ref.backward(); g0 = grads(lm).clone(); lm.model.zero_grad()
ok = True
l = split_lm_loss(pm, ids, ckpt); l.backward(); g = grads(lm).clone(); lm.model.zero_grad()
d = (g - g0).abs().max().item(); print(f"split : loss {l.item():.6f} (ref {ref.item():.6f}), max |grad diff| {d:.2e}, checkpoints {calls}"); ok &= l.item() == ref.item() and d == 0.0
for vb in (1, 7, 100, 1003, 5000):
    calls.clear(); sur, l = vocab_lm_loss(pm, ids, vb, ckpt); sur.backward(); g = grads(lm).clone(); lm.model.zero_grad()
    rel = ((g - g0).norm() / g0.norm()).item()
    print(f"vocab block {vb:5d}: loss {l.item():.6f} (diff {l.item() - ref.item():+.1e}), relative grad diff {rel:.2e}, checkpoints {len(calls)}"); ok &= abs(l.item() - ref.item()) < 1e-4 and rel < 1e-4
print("PASS" if ok else "FAIL"); raise SystemExit(0 if ok else 1)
