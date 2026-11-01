#!/usr/bin/env python3
"""Comparison figure for the lmcache-backend sweep, in the house style of plot_case_a's
headline figure: 2x2 grid (columns = access pattern, top = TTFT log with p50/p95 band,
bottom = throughput), native OffloadingConnector in grey vs LMCacheConnectorV1 in blue.

    .venv/bin/python scripts/plots/plot_lmcache.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
OUT, FIGS = REPO / "output", REPO / "figures"

BLUE, RED, GREY = "#1f6feb", "#c1440e", "#57606a"

# Explicit precedence, later files override: the native floor configs were re-run under
# their own tags and those rows -- hung, but measured -- are the ones the original report
# published. The pilot summaries share config names with a different workload and must
# not be read at all.
SOURCES = ["summary_main.csv", "summary_b24_restore.csv", "summary_uniform_off.csv",
           "summary_zipf_low.csv", "summary_lmcache.csv"]


def load():
    rows = {}
    for name in SOURCES:
        f = OUT / name
        if not f.exists():
            continue
        for r in csv.DictReader(open(f)):
            if r.get("arm") == "offload" and r.get("gpu_kv_gib"):
                rows[r["name"]] = r
    return rows


def series(rows, skew, backend):
    sel = [r for n, r in rows.items()
           if n.startswith(f"{skew}_b") and n.endswith("_lmcache") == (backend == "lmcache")]
    sel.sort(key=lambda r: float(r["budget_gib"]))
    return sel


def cols(sel, *keys):
    return [[float(r[k]) for r in sel] for k in keys]


def rings(ax, sel, ys, label=None):
    hung = [i for i, r in enumerate(sel) if r.get("hung") == "True"]
    if hung:
        ax.scatter([float(sel[i]["budget_gib"]) for i in hung], [ys[i] for i in hung],
                   s=170, facecolors="none", edgecolors=RED, lw=2, zorder=5, label=label)


rows = load()
fig, axgrid = plt.subplots(2, 2, figsize=(17, 7.5), sharex="col",
                           gridspec_kw={"height_ratios": [1.9, 1], "hspace": 0.12,
                                        "wspace": 0.14})

for ax, skew in zip(axgrid[0], ("zipf", "uniform")):
    for backend, color, tag in (("native", GREY, "native"), ("lmcache", BLUE, "lmcache")):
        sel = series(rows, skew, backend)
        x, p50, p95 = cols(sel, "budget_gib", "ttft_p50_ms", "ttft_p95_ms")
        ax.plot(x, p50, color=color, marker="o", lw=2, label=f"{tag} p50")
        ax.plot(x, p95, color=color, marker="^", ls="--", lw=1.2, alpha=0.7,
                label=f"{tag} p95")
        ax.fill_between(x, p50, p95, color=color, alpha=0.10)
        rings(ax, sel, p50,
              label="engine stalled (run aborted)" if backend == "native" else None)

    ax.axhline(530, color=GREY, lw=1, ls=":")
    ax.annotate("recompute ≈ 530 ms", xy=(30, 530), xytext=(4, 4),
                textcoords="offset points", fontsize=8, color=GREY, ha="left")
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.invert_xaxis()
    ax.set_ylabel("TTFT (ms, log)")
    ax.set_title(f"{skew} access", fontsize=11)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="upper left")
axgrid[0][1].set_ylabel("")

for ax, skew in zip(axgrid[1], ("zipf", "uniform")):
    for backend, color in (("native", GREY), ("lmcache", BLUE)):
        sel = series(rows, skew, backend)
        x, tps = cols(sel, "budget_gib", "output_tok_per_s")
        ax.plot(x, tps, color=color, marker="o", lw=2)
        rings(ax, sel, tps)
    ax.axhline(256.0, color=GREY, lw=1, ls="--")
    ax.annotate("offered load: 2 req/s × 128 tok = 256 tok/s", xy=(30, 256),
                xytext=(4, -13), textcoords="offset points", fontsize=8, color=GREY,
                ha="left")
    ax.set_ylim(0, 300)
    # sharex="col" propagates the top row's inversion; inverting twice cancels out.
    ax.set_xlabel("VRAM budget given to vLLM (GiB)   —   tighter to the right")
    ax.grid(alpha=0.25)
axgrid[1][0].set_ylabel("output throughput (tok/s)")

fig.suptitle("Native OffloadingConnector vs LMCache as the DRAM tier — same sweep, "
             "Llama-3-8B, 24 GiB KV working set, RTX 5090 (PCIe Gen4 ×8)",
             fontsize=13, y=0.98)
dest = FIGS / "lmcache_offload.png"
fig.savefig(dest, dpi=140, bbox_inches="tight")
print(f"wrote {dest}")
