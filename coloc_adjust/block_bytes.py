#!/usr/bin/env python
"""Where the bytes of the 24 blocks go: one micro-batch of the usual tenant (not compiled), with a mark before every part of
every block, measured like segment_bytes.py (Nsight Systems, memory read + write throughput at 10 kHz, tenant alone on a
green context of 32 SMs).

Parts of a block, forward: norm1 | q, k, v projections (each with its LoRA branch) | attention itself (rotary positions,
scaled dot product) | output projection (with LoRA) and the residual addition | norm2 | MLP and the residual addition.
Backward: the same parts in reverse order. Everything outside the blocks is "embedding" or "output layer and loss".
  SCRIPT=block_bytes.py GPU=1 ./run_knee_corun.sh            (driver; it starts itself again as the worker under nsys)
Writes results_prof/block_parts.csv and .md."""
import argparse, csv, os, re, statistics as st, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--worker", action="store_true"); ap.add_argument("--sms", type=int, default=32)
ap.add_argument("--batches", type=int, default=16); ap.add_argument("--skip", type=int, default=4); ap.add_argument("--out", default=os.path.join(HERE, "results_prof", "block_parts"))
ap.add_argument("--keep", action="store_true")
args = ap.parse_args()

if args.worker:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    from green_ctx import GreenCtx
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    model.enable_input_require_grads(); model.to("cuda:0").train()
    g = GreenCtx(args.sms, device=0) if args.sms else None; S = g.torch_stream() if g else torch.cuda.default_stream()
    if g: torch.cuda.set_stream(S)
    nv = torch.cuda.nvtx; n = {"i": 0}
    def mark(label): S.synchronize(); nv.mark(f"b{n['i']}:{label}")          # the mark sits where the work issued so far has finished
    pre = lambda label: (lambda *a: mark(label))
    lm = model.base_model.model; core = lm.model
    for L in core.layers:
        at = L.self_attn
        L.register_forward_pre_hook(pre("fwd:norm1")); at.q_proj.register_forward_pre_hook(pre("fwd:qkv_proj")); at.v_proj.register_forward_hook(pre("fwd:attention"))
        at.o_proj.register_forward_pre_hook(pre("fwd:o_proj+residual")); L.post_attention_layernorm.register_forward_pre_hook(pre("fwd:norm2")); L.mlp.register_forward_pre_hook(pre("fwd:mlp+residual"))
        L.mlp.register_full_backward_pre_hook(pre("bwd:mlp+residual")); L.post_attention_layernorm.register_full_backward_pre_hook(pre("bwd:norm2"))
        at.o_proj.register_full_backward_pre_hook(pre("bwd:o_proj+residual")); at.o_proj.register_full_backward_hook(pre("bwd:attention"))
        for m in (at.q_proj, at.k_proj, at.v_proj): m.register_full_backward_pre_hook(pre("bwd:qkv_proj"))
        L.input_layernorm.register_full_backward_pre_hook(pre("bwd:norm1"))
    core.norm.register_forward_pre_hook(pre("head")); core.layers[0].input_layernorm.register_full_backward_hook(pre("embedding_bwd"))
    gen = torch.Generator().manual_seed(0)
    for i in range(args.batches):
        ids = torch.randint(0, model.config.vocab_size, (2, 512), generator=gen).to("cuda:0")
        n["i"] = i; model.zero_grad(set_to_none=True); mark("begin")
        model(input_ids=ids, labels=ids).loss.backward(); mark("end")
    sys.exit(0)

import nsys_tools
from nsys_tools import Capture, drop
K = nsys_tools.calibrate(os.path.join(HERE, "results_prof", "seg"))
db = nsys_tools.capture(args.out, "run", [sys.executable, os.path.abspath(__file__), "--worker", "--sms", str(args.sms), "--batches", str(args.batches)])
cap = Capture(db); per = {}
for s, t in cap.marks:
    m = re.fullmatch(r"b(\d+):(.*)", t)
    if m: per.setdefault(int(m.group(1)), []).append((s, m.group(2)))
tot = {}; tims = {}; cnt = {}; whole = []
for i, ms in sorted(per.items()):
    if i < args.skip or ms[0][1] != "begin" or ms[-1][1] != "end": continue
    acc = {}; tacc = {}; c = {}
    for (a, la), (b, _) in zip(ms, ms[1:]):
        lab = "embedding" if la == "begin" else "output layer and loss" if la == "head" else "embedding" if la == "embedding_bwd" else la
        gb = cap.pct_s("read", a, b) * K["read"] + cap.pct_s("write", a, b) * K["write"]
        acc[lab] = acc.get(lab, 0) + gb; tacc[lab] = tacc.get(lab, 0) + (b - a) / 1e6; c[lab] = c.get(lab, 0) + 1
    for k in acc: tot.setdefault(k, []).append(acc[k]); tims.setdefault(k, []).append(tacc[k]); cnt[k] = c[k]
    whole.append(sum(acc.values()))
order = ["fwd:norm1", "fwd:qkv_proj", "fwd:attention", "fwd:o_proj+residual", "fwd:norm2", "fwd:mlp+residual", "bwd:mlp+residual", "bwd:norm2", "bwd:o_proj+residual", "bwd:attention", "bwd:qkv_proj", "bwd:norm1", "embedding", "output layer and loss"]
rows = [dict(part=k, gb_per_microbatch=st.mean(tot[k]), ms=st.mean(tims[k]), segments=cnt[k]) for k in order if k in tot] + [dict(part=k, gb_per_microbatch=st.mean(tot[k]), ms=st.mean(tims[k]), segments=cnt[k]) for k in tot if k not in order]
blocks = sum(r["gb_per_microbatch"] for r in rows if r["part"].startswith(("fwd:", "bwd:"))); W = st.mean(whole)
with open(args.out + ".csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["part", "gb_per_microbatch", "share_of_blocks_pct", "ms", "segments"])
    for r in rows: w.writerow([r["part"], f"{r['gb_per_microbatch']:.3f}", f"{r['gb_per_microbatch'] / blocks * 100:.1f}" if r["part"].startswith(("fwd:", "bwd:")) else "", f"{r['ms']:.2f}", r["segments"]])
md = [f"# Parts of the 24 blocks: bytes per micro-batch of the usual tenant ({time.strftime('%Y-%m-%d')})\n",
      f"Tenant alone, green context of {args.sms} SMs, {len(whole)} micro-batches of 2 x 512 tokens; a mark (with a wait for the GPU) before every part, so the micro-batch runs slower than in the tenant. "
      f"One micro-batch: {W:.2f} GB, of it the 24 blocks {blocks:.2f} GB.\n", "| part (summed over the 24 blocks) | GB per micro-batch | share of the blocks | ms |", "|---|---|---|---|"]
for r in rows: md.append(f"| {r['part']} | {r['gb_per_microbatch']:.3f} | " + (f"{r['gb_per_microbatch'] / blocks * 100:.0f} %" if r["part"].startswith(("fwd:", "bwd:")) else "–") + f" | {r['ms']:.1f} |")
open(args.out + ".md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
if not args.keep: drop(db)
