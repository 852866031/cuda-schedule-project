#!/usr/bin/env python3
"""Inventory of one tenant micro-batch, on the CPU (no GPU used): every primitive operator of the
forward and backward pass, how many bytes it reads and writes, and whether a read is a weight or an
intermediate result. Gives an analytic bytes-per-token figure to compare with the measured one.

  CUDA_VISIBLE_DEVICES= python tenant_inventory.py [--seqs 2 --seq 512] [--top-layers 0] [--forward-only]

Method: a TorchDispatchMode sees each primitive call with its inputs and outputs. Views (no data
moved) are skipped. Bytes are counted at 2 bytes per floating-point element, as on the GPU (bf16),
whatever dtype the CPU run uses. Assumption behind the total: in eager execution every operator is
its own kernel, reads its inputs from GPU memory and writes its output there. Fused GPU kernels
(for example a fused attention) move less than this count; that difference is what D1 measures."""
import argparse, collections, os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("HF_HOME", "/mnt/storage/huggingface")
import torch
from torch.utils._python_dispatch import TorchDispatchMode
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B"); ap.add_argument("--seqs", type=int, default=2, help="sequences in one micro-batch (batch 8, k=4 -> 2)")
ap.add_argument("--seq", type=int, default=512); ap.add_argument("--lora-r", type=int, default=16)
ap.add_argument("--top-layers", type=int, default=0, help="LoRA only on the top N layers (0 = all, as the tenant does)")
ap.add_argument("--forward-only", action="store_true"); ap.add_argument("--attn", default=None, help="attn_implementation passed to from_pretrained (eager / sdpa)")
ap.add_argument("--threads", type=int, default=8); ap.add_argument("--out", default=None)
args = ap.parse_args(); torch.set_num_threads(args.threads); assert not torch.cuda.is_available()
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
kw = dict(attn_implementation=args.attn) if args.attn else {}
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, **kw)
n_layers = model.config.num_hidden_layers
cfg = dict(r=args.lora_r, lora_alpha=2 * args.lora_r, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
if args.top_layers: cfg["layers_to_transform"] = list(range(n_layers - args.top_layers, n_layers))
model = get_peft_model(model, LoraConfig(**cfg))
if not args.top_layers: model.enable_input_require_grads()      # as ft_worker.py does
model.train()
PARAM = {p.data_ptr(): n for n, p in model.named_parameters()}
weight_bytes = sum(p.numel() for p in model.parameters()) * 2
FB = lambda t: t.numel() * (2 if t.is_floating_point() else t.element_size())
rows = collections.defaultdict(lambda: dict(n=0, rw=0, ra=0, wr=0)); phase = {"p": "forward"}
def tensors(x):
    if isinstance(x, torch.Tensor): yield x
    elif isinstance(x, (list, tuple)):
        for y in x: yield from tensors(y)
class Tally(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, a=(), k=None):
        out = func(*a, **(k or {}))
        ins = list(tensors(a)) + list(tensors(list((k or {}).values()))); outs = list(tensors(out))
        in_ptr = {t.data_ptr() for t in ins if t.numel()}
        fresh = [t for t in outs if t.numel() and t.data_ptr() not in in_ptr]         # not a view of an input
        inplace = [t for t in outs if t.numel() and any(t is i for i in ins)]         # x.add_(...) style
        if not fresh and not inplace: return out                                      # a pure view: no bytes move
        r = rows[(phase["p"], func.__name__)]; r["n"] += 1; seen = set()
        for t in ins:
            if not t.numel() or t.data_ptr() in seen: continue
            seen.add(t.data_ptr()); r["rw" if t.data_ptr() in PARAM else "ra"] += FB(t)
        r["wr"] += sum(FB(t) for t in fresh) + sum(FB(t) for t in inplace)
        return out
ids = torch.randint(0, model.config.vocab_size, (args.seqs, args.seq)); tokens = ids.numel()
with Tally():
    out = model(input_ids=ids, labels=ids); loss = out.loss
    if not args.forward_only:
        phase["p"] = "backward"; loss.backward()
MB = 2 ** 20; tot = collections.defaultdict(lambda: dict(n=0, rw=0, ra=0, wr=0))
for (ph, _), r in rows.items():
    for k_ in r: tot[ph][k_] += r[k_]; tot["all"][k_] += r[k_]
L = [f"# One tenant micro-batch, operator by operator ({args.seqs} x {args.seq} tokens, LoRA r={args.lora_r}"
     f"{', top ' + str(args.top_layers) + ' layers' if args.top_layers else ', all layers'}{', forward only' if args.forward_only else ''})", "",
     f"Model {args.model}: {n_layers} layers, hidden {model.config.hidden_size}, attention implementation `{getattr(model.config, '_attn_implementation', '?')}`; "
     f"all parameters = {weight_bytes / MB:.0f} MB at 2 bytes per element. Counted on the CPU; no GPU was used.", "",
     "| pass | operator calls that move data | weights read | intermediates read | written | total | per token |", "|---|---|---|---|---|---|---|"]
for ph in ("forward", "backward", "all"):
    t = tot[ph]; s = t["rw"] + t["ra"] + t["wr"]
    L.append(f"| {ph} | {t['n']:,} | {t['rw'] / MB:,.0f} MB | {t['ra'] / MB:,.0f} MB | {t['wr'] / MB:,.0f} MB | {s / MB:,.0f} MB | {s / tokens / MB:.2f} MB |")
a_ = tot["all"]; s_all = a_["rw"] + a_["ra"] + a_["wr"]
L += ["", f"Weights are {a_['rw'] / s_all * 100:.0f} % of the bytes; intermediate results (read and written) are {(a_['ra'] + a_['wr']) / s_all * 100:.0f} %.", "",
      "## The operators that move the most", "", "| pass | operator | calls | weights read | intermediates read | written | share of all bytes |", "|---|---|---|---|---|---|---|"]
for (ph, name), r in sorted(rows.items(), key=lambda kv: -(kv[1]["rw"] + kv[1]["ra"] + kv[1]["wr"]))[:18]:
    s = r["rw"] + r["ra"] + r["wr"]
    L.append(f"| {ph} | `{name}` | {r['n']} | {r['rw'] / MB:,.0f} MB | {r['ra'] / MB:,.0f} MB | {r['wr'] / MB:,.0f} MB | {s / s_all * 100:.1f} % |")
txt = "\n".join(L) + "\n"; print(txt)
if args.out: open(args.out, "w").write(txt)
