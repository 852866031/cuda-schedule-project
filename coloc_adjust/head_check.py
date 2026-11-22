#!/usr/bin/env python
"""GPU check of head_loss.py on the real tenant model (Qwen2.5-0.5B, bf16, LoRA on q/k/v/o): loss and the
gradients on the LoRA parameters of one micro-batch (2 x 512 tokens), against the model's own loss.
A second run of the model's own loss shows how much two identical runs differ.
  SCRIPT=head_check.py GPU=1 ./run_knee_corun.sh [--vocab-blocks 4096,8192,16384]"""
import argparse, time, torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from head_loss import split_lm_loss, vocab_lm_loss
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B"); ap.add_argument("--vocab-blocks", default="4096,8192,16384,32768")
args = ap.parse_args(); dev = "cuda:0"
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
model.enable_input_require_grads(); model.to(dev).train()
torch.manual_seed(1)
for n, p in model.named_parameters():                       # lora_B starts at zero, which would leave lora_A without gradient
    if "lora_B" in n: torch.nn.init.normal_(p, std=0.02)
ps = [p for p in model.parameters() if p.requires_grad]
ids = torch.randint(0, model.config.vocab_size, (2, 512), generator=torch.Generator().manual_seed(0)).to(dev)
def run(fn):
    model.zero_grad(set_to_none=True); torch.cuda.synchronize(); t0 = time.time(); l = fn(); torch.cuda.synchronize()
    return float(l), torch.cat([p.grad.float().flatten() for p in ps]), (time.time() - t0) * 1e3
def own():
    l = model(input_ids=ids, labels=ids).loss; l.backward(); return l
def split():
    l = split_lm_loss(model, ids, lambda *_: None); l.backward(); return l
def vocab(vb, fused=False):
    def f():
        sur, l = vocab_lm_loss(model, ids, vb, lambda *_: None, None, fused); sur.backward(); return l
    return f
for _ in range(2): run(own)
l0, g0, t0 = run(own); print(f"model's own loss   : loss {l0:.5f}, |grad| {g0.norm():.4e}, {t0:.1f} ms")
def show(name, r):
    l, g, t = r; cos = torch.nn.functional.cosine_similarity(g, g0, dim=0).item()
    print(f"{name:19s}: loss {l:.5f} (diff {l - l0:+.1e}), grad relative diff {((g - g0).norm() / g0.norm()).item():.2e}, cosine {cos:.6f}, {t:.1f} ms", flush=True)
show("own, second run", run(own)); run(split); show("split", run(split))
for vb in [int(x) for x in args.vocab_blocks.split(",")]:
    run(vocab(vb)); show(f"vocab block {vb}", run(vocab(vb)))
    for _ in range(3): run(vocab(vb, True))
    show(f"  fused kernels {vb}", run(vocab(vb, True)))
