#!/usr/bin/env python3
"""§6.3 figure: the Orion-lite gate against the static MPS caps.

Left: decode's TTFT p50 and TPOT, normalized to the decode-alone baseline — the
gate's headline is the TTFT bar returning to 1.0. Right: what each arm buys the
trainer. Data: output/raw/coloc_lmcache_zipf_b26_*.json + the split study's b26
row. Run from the repo root.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GRY = "#57606a"
BLU = "#1f6feb"
ORG = "#c1440e"
AMB = "#9a6700"
GRN = "#2e7d4f"

TOK_PER_STEP = 2 * 512


def load(name):
    return json.load(open(f"output/raw/{name}.json"))


def tpot(rec):
    s = rec["summary"]
    v = s.get("tpot_ms", {}).get("mean")
    return v if v is not None else s["itl_ms"]["mean"]  # equal for fixed 128-tok outputs


def ft_rate(rec):
    if "ft" not in rec:
        return 0.0
    return (rec["ft"]["measure_window"]["steps"] * TOK_PER_STEP
            / (rec["t_measure1"] - rec["t_measure0"]))


ref = load("split_lmcache_zipf_b26_fwd_ng")
arm_files = [("MPS cap 50%", "coloc_lmcache_zipf_b26_g2mps50"),
             ("MPS no cap", "coloc_lmcache_zipf_b26_g2mps100"),
             ("idle-window\n(unfenced)", "coloc_lmcache_zipf_b26_g2gate"),
             ("idle-window\n+ 1-block fence", "coloc_lmcache_zipf_b26_g2gate2")]
arms = [("decode alone\n(baseline)", ref)] + [(lbl, load(f)) for lbl, f in arm_files]

ttft0 = ref["summary"]["ttft_ms"]["p50"]
tpot0 = tpot(ref)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                              gridspec_kw={"width_ratios": [1.2, 1]})

xpos = np.arange(len(arms))
ttft_n = [r["summary"]["ttft_ms"]["p50"] / ttft0 for _, r in arms]
tpot_n = [tpot(r) / tpot0 for _, r in arms]
b1 = ax.bar(xpos - 0.19, ttft_n, 0.38, color="#f0d9cd", edgecolor=ORG, lw=1.2,
            label="TTFT p50 / baseline")
b2 = ax.bar(xpos + 0.19, tpot_n, 0.38, color=BLU, label="TPOT / baseline")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
                f"{b.get_height():.2f}×", ha="center", fontsize=8.5, color=GRY)
ax.axhline(1.0, color=GRY, lw=1.0, ls=":")
ax.set_xticks(xpos, [l for l, _ in arms], fontsize=9)
ax.set_ylim(0, 2.05)
ax.set_ylabel("× the decode-alone baseline", fontsize=10)
ax.set_title("Idle-window scheduling returns TTFT to baseline; caps never do",
             fontsize=11, color=GRY)
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(alpha=0.25, axis="y")

ft = [ft_rate(r) / 1e3 for _, r in arms]
bars = ax2.bar(xpos, ft, 0.5, color=AMB)
for b, v in zip(bars, ft):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.8,
             "—" if v == 0 else f"{v:.1f}k", ha="center", fontsize=9, color=AMB)
ax2.set_xticks(xpos, [l for l, _ in arms], fontsize=9)
ax2.set_ylabel("fine-tune tok/s (thousands)", fontsize=10)
ax2.set_ylim(0, 50)
ax2.set_title("What each arm buys the trainer", fontsize=11, color=GRY)
ax2.grid(alpha=0.25, axis="y")

fig.suptitle(
    "Idle-window scheduling: the decode engine publishes each step's busy window to "
    "shared memory, and the trainer\nissues each transformer block's kernels only "
    "while that window is idle — compared with static MPS SM caps\n"
    "(GPT-2 LoRA fine-tune beside the split decode node, 26 GiB budget, reference "
    "workload at 2 QPS, MPS on in all arms)", fontsize=10.5, y=1.10)
fig.tight_layout()
fig.savefig("figures/coloc_gate_arms.png", dpi=140, bbox_inches="tight",
            facecolor="white")
print("wrote figures/coloc_gate_arms.png")

# ============ companion: how the mechanism works ============
from matplotlib.patches import FancyArrowPatch, Rectangle

fig2, (axA, axB) = plt.subplots(2, 1, figsize=(13, 7.6),
                                gridspec_kw={"height_ratios": [1, 1.15]})
for a in (axA, axB):
    a.set_xlim(0, 130); a.axis("off")
axA.set_ylim(0, 30); axB.set_ylim(0, 32)
fig2.suptitle("Idle-window scheduling, mechanically", fontsize=12.5, color=GRY,
              fontweight="bold", y=0.98)


def box(ax_, x, y, w, h, fc, ec, lw=1.4):
    ax_.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw,
                            zorder=2))


def arrow(ax_, x0, y0, x1, y1, color, ls="-"):
    ax_.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                  mutation_scale=13, color=color, lw=1.7,
                                  linestyle=ls, zorder=5))


# ---- panel A: the three pieces
box(axA, 2, 6, 40, 18, "#ddeee4", GRN)
axA.text(22, 21.2, "vLLM decode engine (GPU1)", fontsize=10.5, ha="center",
         fontweight="bold", color=GRN)
axA.text(22, 13.5, "sitecustomize patch wraps execute_model():\n"
         "busy ← 1\nrun the step (CUDA-graph replay, 15–25 ms)\nbusy ← 0",
         fontsize=8.8, ha="center", va="center", color=GRY)
axA.text(22, 4.2, "engine code untouched — the wrapper is loaded via PYTHONPATH",
         fontsize=7.8, ha="center", color=GRY)

box(axA, 50, 10, 30, 10, "#f7e3d8", ORG)
axA.text(65, 21.2, "shared-memory page (/dev/shm)", fontsize=10.5, ha="center",
         fontweight="bold", color=ORG)
axA.text(65, 15, "busy flag\nheartbeat (refreshed every 100 ms)", fontsize=8.8,
         ha="center", va="center", color=GRY)
axA.text(65, 7.8, "heartbeat stale > 1.5 s ⇒ gate opens\n(a dead engine can never hang the trainer)",
         fontsize=7.8, ha="center", color=ORG)

box(axA, 88, 6, 40, 18, "#fbf0d9", AMB)
axA.text(108, 21.2, "LoRA trainer (same GPU)", fontsize=10.5, ha="center",
         fontweight="bold", color=AMB)
axA.text(108, 12.8, "hook before EVERY transformer block (fwd + bwd):\n"
         "1. fence: wait until the previous block's\n    kernels finished (1-block lookahead)\n"
         "2. wait while busy = 1   (poll 0.2 ms)\n"
         "3. issue this block's kernels (~0.5–1 ms)",
         fontsize=8.8, ha="center", va="center", color=GRY)

arrow(axA, 42, 16.5, 50, 16.5, GRN)
axA.text(46, 18, "write", fontsize=8.2, ha="center", color=GRN)
arrow(axA, 80, 16.5, 88, 16.5, ORG)
axA.text(84, 18, "read", fontsize=8.2, ha="center", color=ORG)

# ---- panel B: one stretch of time
axB.text(2, 30.5, "the same ~55 ms of wall-clock, three views:", fontsize=10,
         color=GRY)
STEPS = [(18, 38), (58, 82), (104, 128)]   # decode busy intervals
GAPS = [(38, 58), (82, 104)]

axB.text(1, 24.5, "decode\nGPU work", fontsize=9, ha="left", va="center", color=GRN)
for x0, x1 in STEPS:
    box(axB, x0, 22, x1 - x0, 5, "#ddeee4", GRN)
    axB.text((x0 + x1) / 2, 24.5, "decode step", fontsize=8.4, ha="center",
             va="center", color=GRN)
for x0, x1 in GAPS:
    axB.text((x0 + x1) / 2, 24.5, "CPU gap", fontsize=8.0, ha="center",
             va="center", color=GRY)

axB.text(1, 17, "busy flag", fontsize=9, ha="left", va="center", color=ORG)
for x0, x1 in STEPS:
    box(axB, x0, 15.5, x1 - x0, 3, "#f7e3d8", ORG)
    axB.text((x0 + x1) / 2, 17, "1", fontsize=8.4, ha="center", va="center",
             color=ORG)
for x0, x1 in GAPS:
    axB.plot([x0, x1], [15.5, 15.5], color=ORG, lw=1.4)
    axB.text((x0 + x1) / 2, 16.8, "0", fontsize=8.4, ha="center", color=ORG)

axB.text(1, 9.5, "trainer\nkernel issue", fontsize=9, ha="left", va="center",
         color=AMB)
for x0, x1 in GAPS:
    x = x0 + 1.2
    while x + 3.2 < x1:
        box(axB, x, 7.5, 3.2, 4, "#9a6700", AMB)
        x += 4.4
    # the bounded overshoot: the last issued block's kernels drain into the step
    box(axB, x, 7.5, 3.2, 4, "#fbf0d9", AMB)
for x0, x1 in STEPS[:2]:
    axB.text((x0 + x1) / 2, 9.5, "gated (waiting)", fontsize=8.0, ha="center",
             va="center", color=AMB)
axB.text(116, 9.5, "gated (waiting)", fontsize=8.0, ha="center", va="center",
         color=AMB)
axB.annotate("≤ 1 block drains into the step\n(the fence's bound — v1 leaked a whole queue here)",
             xy=(57.5, 8.5), xytext=(34, 2.2), fontsize=8.4, color=GRY,
             arrowprops=dict(arrowstyle="->", color=GRY, lw=1.2))
axB.text(112, 3.2, "filled = issued & runs in the gap\nhollow = issued in the gap,\nfinishes just past its edge",
         fontsize=7.8, ha="center", color=GRY)
axB.annotate("", xy=(128, 29.2), xytext=(18, 29.2),
             arrowprops=dict(arrowstyle="->", color=GRY, lw=1.0))
axB.text(66, 29.9, "time", fontsize=8.5, ha="center", color=GRY)

fig2.tight_layout(rect=(0, 0, 1, 0.96))
fig2.savefig("figures/coloc_gate_mechanism.png", dpi=140, bbox_inches="tight",
             facecolor="white")
print("wrote figures/coloc_gate_mechanism.png")

# ============ §6.4: kernel-level (CUPTI) gate vs block-level gate ============
karm_files = [("decode alone\n(baseline)", None),
              ("MPS no cap\n(no gating)", "coloc_lmcache_zipf_b26_g2mps100"),
              ("block-level gate\n(python hooks)", "coloc_lmcache_zipf_b26_g2gate2"),
              ("kernel-level gate\n(CUPTI, K=8)", "coloc_lmcache_zipf_b26_g2kgate"),
              ("kernel-level gate\n(CUPTI, tight K=2)", "coloc_lmcache_zipf_b26_g2kgatet")]
karms = [(lbl, ref if f is None else load(f)) for lbl, f in karm_files]

fig3, (axk, axk2) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                 gridspec_kw={"width_ratios": [1.2, 1]})
xk = np.arange(len(karms))
kttft = [r["summary"]["ttft_ms"]["p50"] / ttft0 for _, r in karms]
ktpot = [tpot(r) / tpot0 for _, r in karms]
kb1 = axk.bar(xk - 0.19, kttft, 0.38, color="#f0d9cd", edgecolor=ORG, lw=1.2,
              label="TTFT p50 / baseline")
kb2 = axk.bar(xk + 0.19, ktpot, 0.38, color=BLU, label="TPOT / baseline")
for bars in (kb1, kb2):
    for b in bars:
        axk.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
                 f"{b.get_height():.2f}×", ha="center", fontsize=8.3, color=GRY)
axk.axhline(1.0, color=GRY, lw=1.0, ls=":")
axk.set_xticks(xk, [l for l, _ in karms], fontsize=8.3)
axk.set_ylim(0, 2.05)
axk.set_ylabel("× the decode-alone baseline", fontsize=10)
axk.set_title("Kernel-level interception matches the block gate's protection",
              fontsize=11, color=GRY)
axk.legend(fontsize=8.5, loc="upper right")
axk.grid(alpha=0.25, axis="y")

kft = [ft_rate(r) / 1e3 for _, r in karms]
kbars = axk2.bar(xk, kft, 0.5, color=AMB)
for b, v in zip(kbars, kft):
    axk2.text(b.get_x() + b.get_width() / 2, v + 0.8,
              "—" if v == 0 else f"{v:.1f}k", ha="center", fontsize=9, color=AMB)
axk2.set_xticks(xk, [l for l, _ in karms], fontsize=8.3)
axk2.set_ylabel("fine-tune tok/s (thousands)", fontsize=10)
axk2.set_ylim(0, 50)
axk2.set_title("What each arm buys the trainer", fontsize=11, color=GRY)
axk2.grid(alpha=0.25, axis="y")

fig3.suptitle(
    "Gating below cuBLAS: a CUPTI callback inside the driver pauses EVERY kernel "
    "launch of the trainer while decode is\nmid-step — no trainer code changes, no "
    "per-library wrappers (16.7M launches intercepted per run; LD_PRELOAD-style\n"
    "interposition, Orion's mechanism, sees none of cuBLAS's). Same b26 setup, "
    "MPS on in all arms.", fontsize=10.5, y=1.10)
fig3.tight_layout()
fig3.savefig("figures/coloc_kgate_arms.png", dpi=140, bbox_inches="tight",
             facecolor="white")
print("wrote figures/coloc_kgate_arms.png")
