#!/usr/bin/env python3
"""KV-slot study: resident vs streamed KV per decode step (output/kv_vram/bench_stream.json).

    .venv/bin/python scripts/plots/plot_kv_stream.py  -> figures/kv_stream_bench.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
d = json.load(open(REPO / "output" / "kv_vram" / "bench_stream.json"))
rows = d["rows"]
B = [r["batch"] for r in rows]
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(B, [r["resident_ms"] for r in rows], "o-", color="#2a78d6", lw=2.4, ms=7, label="KV resident in HBM (what vLLM does)")
ax.plot(B, [r["copy_ms"] for r in rows], "s:", color="#9a9a95", lw=1.4, ms=5, label="copy alone: B × 0.78 GiB at 14.47 GB/s")
ax.plot(B, [r["stream_d0_ms"] for r in rows], "^-", color="#eb6834", lw=1.8, ms=6, label="streamed per layer, no prefetch (d = 0)")
ax.plot(B, [r["stream_d1_ms"] for r in rows], "v--", color="#b3261e", lw=1.8, ms=6, label="streamed, prefetch d = 1 (d = 2, 4, 8 identical)")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(B)
ax.set_xticklabels([str(b) for b in B])
ax.set_xlabel("requests in the batch (6,400-token context each)")
ax.set_ylabel("one decode step (ms, log)")
ax.set_title("A decode step with its KV streamed from host memory instead of resident\n"
             "(Llama-3-8B shapes, RTX 5090, PCIe Gen4 ×8; median of 10 steps)")
for b, r in zip(B, rows):
    ax.annotate(f"{r['stream_d1_ms'] / r['resident_ms']:.0f}×", (b, r["stream_d1_ms"]), textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=9, color="#b3261e")
ax.grid(color="#ececea", lw=0.6, which="both")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left", fontsize=9.5)
fig.savefig(REPO / "figures" / "kv_stream_bench.png", dpi=150, bbox_inches="tight")
print("-> figures/kv_stream_bench.png")
