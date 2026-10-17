#!/usr/bin/env python3
"""Eighth study: a Qwen2.5-3B prefill on the 8B's decode GPU (scripts/inf_pf_coloc/).

Three figures from output/summary_pfc.csv and the raw records:

  pfc_rate.png        vs the 3B's request rate (0 = 8B alone): (a) 8B throughput,
                      (b) 8B decode-engine queue wait, (c) 3B TTFT p50, (d) GPU1
                      SM-active. P1 = prefill only, P2 = prefill + KV to DRAM (sync save).
  pfc_saturation.png  8B per-request time per output token after token #1, over the
                      measurement window, at each 3B rate (P1): the queue grows.
  pfc_async.png       3B TTFT with LMCache's synchronous save vs the async-save patch,
                      and the share of P3 decode requests that found their whole KV.

    .venv/bin/python scripts/plots/plot_pfc_results.py
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
RAW, FIGS = REPO / "output" / "raw", REPO / "figures"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "legend.fontsize": 12})
C_P1, C_P2, C_SOLO, C_ASYNC = "#1f6feb", "#cf222e", "#8c959f", "#2e7d4f"

ROWS = {r["name"]: r for r in csv.DictReader(open(REPO / "output" / "summary_pfc.csv"))}


def f(name, col):
    v = ROWS[name][col]
    return float(v) if v not in ("", "None") else np.nan


RATES = [("0", None), ("0.5", "_q0.5"), ("1", "_q1"), ("2", "")]


def cell(pl, q):
    return "pfc_base" if q is None else f"pfc_{pl}{q}"


def bars(ax, vals, fmt, colors, labels):
    """Grouped bars over the 3B rates, one group per rate, value on every bar."""
    xi = np.arange(len(RATES))
    w = 0.8 / len(vals)
    for k, (v, c, lab) in enumerate(zip(vals, colors, labels)):
        xs = xi + (k - (len(vals) - 1) / 2) * w
        b = ax.bar(xs, v, w, color=c, label=lab)
        for rect, y in zip(b, v):
            if np.isfinite(y):
                ax.text(rect.get_x() + rect.get_width() / 2, y, fmt(y), ha="center",
                        va="bottom", fontsize=10)
    ax.set_xticks(xi)
    ax.set_xticklabels([r for r, _ in RATES])
    ax.set_xlabel("3B load (requests / s)")
    ax.grid(alpha=0.3, axis="y")


def rate_figure():
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    cols, labs = [C_P1, C_P2], ["3B prefill only (P1)", "3B prefill + KV to DRAM (P2)"]
    get = lambda col: [[f(cell(pl, q), col) for _, q in RATES] for pl in ("p1", "p2")]

    bars(ax[0], get("big_tok_per_s"), lambda y: f"{y:.0f}", cols, labs)
    ax[0].set(ylabel="8B output tokens / s", ylim=(0, 285),
              title="(a) 8B serves less as the 3B load grows")
    q = [np.array(v) / 1000 for v in get("big_decode_queue_ms")]
    bars(ax[1], q, lambda y: f"{y:.1f}", cols, labs)
    ax[1].set(ylabel="seconds", ylim=(0, 31),
              title="(b) 8B requests wait longer for the decode GPU")
    bars(ax[2], get("gpu1_sm_active_mean"), lambda y: f"{y:.2f}", cols, labs)
    ax[2].set(ylabel="fraction of time the SMs are busy", ylim=(0, 0.8),
              title="(c) The decode GPU (GPU1) is busier")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Placements P1 and P2: 3B prefill on the 8B's decode GPU", y=1.06,
                 fontsize=15, fontweight="bold")
    fig.text(0.5, -0.04, "8B: split, 2 requests/s, decode GPU budget 22 GiB.  3B: a fresh "
             "6144-token prompt per request, prefilled on the 8B's decode GPU.  "
             "(b) = mean time an 8B request waits in the decode engine's queue.",
             ha="center", fontsize=11, color="#444")
    fig.tight_layout()
    fig.savefig(FIGS / "pfc_rate.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def saturation_figure():
    """Per 8B request: the time it spent waiting rather than decoding, over the window.
    Waiting = (end - first token) - 127 tokens x the run's median gap between tokens; the
    median gap is the same 14.4-14.5 ms in every cell, so the rest is queueing."""
    cells = [("pfc_base", "no 3B", "k"), ("pfc_p1_q0.5", "3B at 0.5 req/s", "#7aa7e8"),
             ("pfc_p1_q1", "3B at 1 req/s", C_P1), ("pfc_p1", "3B at 2 req/s", "#0b3d91")]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for nm, lab, c in cells:
        rec = json.loads((RAW / f"{nm}.json").read_text())
        gap = rec["big_summary"]["itl_ms"]["p50"] / 1000
        rs = sorted((r for r in rec["big_records"] if r.get("e2e") and r.get("ttft")),
                    key=lambda r: r["t_submit"])
        t = np.array([r["t_submit"] - rs[0]["t_submit"] for r in rs])
        wait = np.array([max(0.0, r["e2e"] - r["ttft"] - 127 * gap) for r in rs])
        k = 15
        ax.plot(t[k - 1:], np.convolve(wait, np.ones(k) / k, mode="valid"), color=c, lw=2.2,
                label=lab)
    ax.set(xlabel="when the 8B request arrived (s into the run)",
           ylabel="seconds the request waited\nfor the decode GPU",
           title="P1: with the 3B at 1-2 req/s the 8B's queue never drains", ylim=(0, None))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    fig.text(0.5, -0.03, "8B requests, P1 cells; rolling mean of 15 requests.",
             ha="center", fontsize=11, color="#444")
    fig.tight_layout()
    fig.savefig(FIGS / "pfc_saturation.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def async_figure():
    pairs = [("3B alone\nP2", "pfc_s2", "pfc_s2_async"),
             ("3B alone\nP3", "pfc_s3_v2", "pfc_s3_async_v2"),
             ("with 8B\nP2", "pfc_p2", "pfc_p2_async_r2"),
             ("with 8B\nP3", "pfc_p3_l1x4", "pfc_p3_async_l1x4")]
    ref = f("pfc_s1", "sm_ttft_p50_ms")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [2.2, 1]})
    xi = np.arange(len(pairs))
    w = 0.38
    for j, (col, c, lab) in enumerate(((1, C_SOLO, "stock LMCache: copy before the token"),
                                        (2, C_ASYNC, "patch: copy in the background"))):
        p50 = [f(p[col], "sm_ttft_p50_ms") for p in pairs]
        p95 = [f(p[col], "sm_ttft_p95_ms") for p in pairs]
        b = ax[0].bar(xi + (j - 0.5) * w, p50, w, color=c, label=lab)
        ax[0].errorbar(xi + (j - 0.5) * w, p50, yerr=[np.zeros(len(p50)), np.array(p95) - p50],
                       fmt="none", ecolor="k", capsize=3, lw=1)
        for rect, v in zip(b, p50):
            ax[0].text(rect.get_x() + rect.get_width() / 2, v * 0.55, f"{v:.0f}", ha="center",
                       va="center", fontsize=11, color="white", fontweight="bold")
    ax[0].axhline(ref, color=C_P1, ls="--", lw=1.2,
                  label=f"prefill only, 3B alone ({ref:.0f} ms)")
    ax[0].set(xticks=xi, xticklabels=[p[0] for p in pairs], ylabel="3B time to first token (ms)\nbar = median, line = p95",
              ylim=(0, 2300), title="(a) Writing the KV to DRAM no longer delays the first token")
    ax[0].grid(alpha=0.3, axis="y")

    hit = []
    for nm in ("pfc_s3_v", "pfc_s3_v2", "pfc_s3_async", "pfc_s3_async_v2", "pfc_p3_async_l1x4"):
        d = json.loads((RAW / f"{nm}.json").read_text()).get("decode_hits") or {}
        hit.append(100 * d["full_hits"] / d["n"])
    lab = ["stock\nrun 1", "stock\nrun 2", "patch\nrun 1", "patch\nrun 2", "patch\nwith 8B"]
    ax[1].bar(range(5), hit, color=[C_SOLO, C_SOLO, C_ASYNC, C_ASYNC, C_ASYNC])
    for i, v in enumerate(hit):
        ax[1].text(i, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=11)
    ax[1].set(xticks=range(5), xticklabels=lab, ylim=(0, 112),
              ylabel="% of decode requests",
              title="(b) P3: decode found all its KV")
    ax[1].grid(alpha=0.3, axis="y")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Placements P2 and P3: stock synchronous KV save vs the async-save patch",
                 y=1.05, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "pfc_async.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rate_figure()
    saturation_figure()
    async_figure()
    print("wrote figures/pfc_rate.png, pfc_saturation.png, pfc_async.png")
