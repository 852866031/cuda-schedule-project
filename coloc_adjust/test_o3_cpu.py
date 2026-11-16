#!/usr/bin/env python
"""CPU checks of the overnight-3 preparation code (no GPU):  CUDA_VISIBLE_DEVICES= python test_o3_cpu.py"""
import json, os, sys, tempfile
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from pacing import Bpt, BlockPacer, cap_tok_s

# ---- 1. GB/s -> tokens/s, and the bytes-per-token table
assert abs(cap_tok_s(50, 13.8) - 3623.2) < 0.1
f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); json.dump({"base": {"2": 14.0, "3": 16.0, "5": 60.0}, "top4": {"2": 7.0}}, f); f.close()
b = Bpt(f.name, "base"); v = Bpt(f.name, "top4")
assert b.mb(2) == 14.0 and b.mb(5) == 60.0 and b.mb(0) == 14.0, "a shape that was not measured uses the first measured one"
assert v.mb(2) == 7.0 and v.mb(3) == 8.0, "a variant measured for one shape is scaled with the base ratios"
assert Bpt("/nonexistent").mb(2) == 13.8
print("1 ok: 50 GB/s ->", [f"cfg {c}: {cap_tok_s(50, b.mb(c)):.0f} tok/s" for c in (2, 3, 5)])

# ---- 2. block pacing with a simulated clock: 192 ticks per step of 4096 tokens, each tick is 2.4 ms of work
class Sim:
    def __init__(self): self.t = 0.0; self.slept = []
    def clock(self): return self.t
    def sleep(self, d): self.t += d
def run(slice_blocks, cap, steps=6, work=0.0024, stall_at=None):
    sim = Sim(); syncs = [0]
    p = BlockPacer(lambda: syncs.__setitem__(0, syncs[0] + 1), sim.clock, sim.sleep)
    rates = []; runs = []
    for s in range(steps):
        t0 = sim.t; p.begin_step("cfg2", 4096, cap, slice_blocks); run_len = 0.0
        for u in range(192):
            sim.t += work; run_len += work
            if stall_at == (s, u): sim.t += 1.0                       # the tenant is held up for 1 s
            before = sim.t; p.tick()
            if sim.t > before: runs.append(run_len); run_len = 0.0
        st = p.end_step(); rates.append(4096 / (sim.t - t0))
    return rates, runs, syncs[0], st
for sl in (1, 8, 48):
    rates, runs, syncs, st = run(sl, 2400.0)
    assert rates[0] > 8000, "the first step of a shape is not paced inside"
    assert all(abs(r - 2400) / 2400 < 0.02 for r in rates[1:]), rates
    assert abs(max(runs) - sl * 0.0024) < 1e-9, (sl, max(runs))
    print(f"2 ok: slice {sl:2d} blocks: paced steps at {rates[1]:.0f} tok/s (cap 2400), longest run {max(runs)*1e3:.1f} ms, {st['slices']} sleeps and {syncs // 6} syncs per step")
rates, runs, _, _ = run(8, 2400.0, stall_at=(2, 40))
assert max(runs) <= 2 * 8 * 0.0024 + 1e-9, max(runs)
print(f"2 ok: after a 1-s stall the longest uninterrupted run is {max(runs)*1e3:.1f} ms (no catch-up burst)")
rates, _, _, _ = run(8, 1e9)
assert all(r > 8000 for r in rates)
print("2 ok: a cap above the tenant's speed never sleeps")

# ---- 2b. the slow loop on the TPOT log of the Nov 9 tpot_slo15 run (open loop: what it would have published)
import csv
from pacing import slow_update
rows = list(csv.DictReader(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "tpot_slo15_controller.csv"))))
t0 = float(rows[0]["t"]); g = 150.0; last = -1e9; trace = []; fast = []
for r in rows:
    t = float(r["t"]) - t0
    if r["tok_s_cap"]: fast.append((t, float(r["tok_s_cap"])))
    if t - last >= 2.0 and r["tpot_ms"] and r["tpot_base_ms"]:
        last = t; g = slow_update(g, float(r["tpot_ms"]) / float(r["tpot_base_ms"]) - 1.0, 0.15, 4.0, 0.10, 10.0, 150.0); trace.append((t, g))
steps = [abs(b[1] / a[1] - 1) for a, b in zip(trace, trace[1:])]
assert max(steps) <= 0.1 + 1e-9 and min(x for _, x in trace) >= 10.0 and max(x for _, x in trace) <= 150.0
for lo, hi, rps in ((30, 90, 1), (100, 180, 2), (430, 480, 0.5)):
    seg = [x for t, x in trace if lo <= t < hi]; fs = [x for t, x in fast if lo <= t < hi]
    print(f"2b ok: {rps} req/s phase, {lo}-{hi} s: slow loop {min(seg):.0f}-{max(seg):.0f} GB/s (ratio {max(seg)/min(seg):.1f}); the fast loop published {min(fs):.0f}-{max(fs):.0f} tok/s (ratio {max(fs)/min(fs):.0f})")

# ---- 3. chunked output layer + loss: same loss and same LoRA gradients as the usual computation
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from loss_chunk import chunked_lm_loss
torch.manual_seed(0)
for top in (0, 4):
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32)
    cfg = dict(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    if top: cfg["layers_to_transform"] = list(range(m.config.num_hidden_layers - top, m.config.num_hidden_layers))
    m = get_peft_model(m, LoraConfig(**cfg)); m.train()
    if not top: m.enable_input_require_grads()
    for n_, p_ in m.named_parameters():                               # lora_B starts at zero: give it values so lora_A gets a gradient too
        if "lora_B" in n_: torch.nn.init.normal_(p_, std=0.02)
    ids = torch.randint(0, m.config.vocab_size, (2, 48))
    out = m(input_ids=ids, labels=ids); out.loss.backward()
    ref = {n: p.grad.clone() for n, p in m.named_parameters() if p.requires_grad}; m.zero_grad()
    assert min(g_.abs().max().item() for g_ in ref.values()) > 0, "every trainable tensor has a gradient"
    sur, loss = chunked_lm_loss(m, ids, 16); sur.backward()
    worst = max(((p.grad - ref[n]).abs().max() / (ref[n].abs().max() + 1e-12)).item() for n, p in m.named_parameters() if p.requires_grad)
    assert all(p.grad is not ref[n] for n, p in m.named_parameters() if p.requires_grad)
    assert abs(loss.item() - out.loss.item()) < 1e-4 * abs(out.loss.item()), (loss.item(), out.loss.item())
    assert worst < 1e-3, worst
    print(f"3 ok: LoRA on {'top ' + str(top) if top else 'all'} layers, {len(ref)} trainable tensors: loss {out.loss.item():.5f} vs {loss.item():.5f}, largest relative gradient difference {worst:.1e}")
print("all CPU checks passed")
