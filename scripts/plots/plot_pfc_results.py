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


C_BASE = "#57606a"


def bars(ax, vals, fmt, colors, labels, base):
    """x = 3B load. At 0 a single grey bar = the 8B alone (the baseline), carried across
    the panel as a dashed line; at 0.5/1/2 one bar per placement, value on every bar."""
    xi = np.arange(len(RATES))
    w = 0.8 / len(vals)
    ax.axhline(base, color=C_BASE, ls="--", lw=1.4, zorder=0)
    b0 = ax.bar(0, base, 0.5, color=C_BASE, label="8B alone (baseline)")
    ax.text(0, base, fmt(base), ha="center", va="bottom", fontsize=10, fontweight="bold")
    for k, (v, c, lab) in enumerate(zip(vals, colors, labels)):
        xs = xi[1:] + (k - (len(vals) - 1) / 2) * w
        bb = ax.bar(xs, v[1:], w, color=c, label=lab)
        for rect, y in zip(bb, v[1:]):
            if np.isfinite(y):
                ax.text(rect.get_x() + rect.get_width() / 2, y, fmt(y), ha="center",
                        va="bottom", fontsize=10,
                        bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.85))
    ax.set_xticks(xi)
    ax.set_xticklabels(["8B\nalone"] + [r for r, _ in RATES[1:]])
    ax.set_xlabel("3B load (requests / s)")
    ax.grid(alpha=0.3, axis="y")


def rate_figure():
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    cols, labs = [C_P1, C_P2], ["3B prefill only (P1)", "3B prefill + KV to DRAM (P2)"]
    get = lambda col: [[f(cell(pl, q), col) for _, q in RATES] for pl in ("p1", "p2")]

    base = lambda col: f("pfc_base", col)
    bars(ax[0], get("big_tok_per_s"), lambda y: f"{y:.0f}", cols, labs, base("big_tok_per_s"))
    ax[0].set(ylabel="8B output tokens / s", ylim=(0, 285),
              title="(a) 8B throughput: output tokens / s")
    q = [np.array(v) / 1000 for v in get("big_decode_queue_ms")]
    bars(ax[1], q, lambda y: f"{y:.1f}", cols, labs, base("big_decode_queue_ms") / 1000)
    ax[1].set(ylabel="seconds (mean per request)", ylim=(0, 31),
              title="(b) 8B queue wait in the decode engine")
    bars(ax[2], get("gpu1_sm_active_mean"), lambda y: f"{y:.2f}", cols, labs,
         base("gpu1_sm_active_mean"))
    ax[2].set(ylabel="fraction of time (mean over the run)", ylim=(0, 0.8),
              title="(c) GPU1 SM-active (DCGM)")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Placements P1 and P2: 3B prefill on the 8B's decode GPU", y=1.06,
                 fontsize=15, fontweight="bold")
    fig.text(0.5, -0.04, "8B: split, 2 requests/s, decode GPU budget 22 GiB.  3B: a fresh "
             "6144-token prompt per request, prefilled on the 8B's decode GPU.  "
             "(b) = vLLM request_queue_time on the 8B decode engine: arrival there → first scheduled (not TPOT).",
             ha="center", fontsize=11, color="#444")
    fig.tight_layout()
    fig.savefig(FIGS / "pfc_rate.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def saturation_figure():
    """Per 8B request: the time it spent waiting rather than decoding, over the window.
    Waiting = (end - first token) - 127 tokens x the run's median gap between tokens; the
    median gap is the same 14.4-14.5 ms in every cell, so the rest is queueing."""
    cells = [("pfc_base", "8B alone (baseline)", "k"), ("pfc_p1_q0.5", "3B at 0.5 req/s", "#7aa7e8"),
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
           ylabel="seconds waited (per request)",
           title="P1: 8B wait for the decode GPU, per request, over the run", ylim=(0, None))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    fig.text(0.5, -0.03, "wait = (end − first token) − 127 × median gap between tokens "
             "(14.4–14.5 ms in every run); rolling mean of 15 requests.",
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
              ylim=(0, 2300), title="(a) 3B TTFT (time to first token)")
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
              title="(b) P3: decode requests that found\nthe whole KV in DRAM (%)")
    ax[1].grid(alpha=0.3, axis="y")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Placements P2 and P3: stock synchronous KV save vs the async-save patch",
                 y=1.05, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "pfc_async.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def p3_figure():
    """P3: 3B prefill on GPU1 -> DRAM -> 3B decode on GPU0. Beside the 8B only with the
    8B's pinned memory cut to 8 GiB, so its baseline is the 8B alone at 8 GiB."""
    fig, ax = plt.subplots(2, 3, figsize=(18, 9.5))
    ax = ax.ravel()

    def panel(a, names, labels, col, colors, fmt, title, ylabel, scale=1.0, base=None):
        v = [f(n, col) * scale for n in names]
        b = a.bar(range(len(v)), v, 0.6, color=colors)
        for rect, y in zip(b, v):
            a.text(rect.get_x() + rect.get_width() / 2, y, fmt(y), ha="center", va="bottom",
                   fontsize=10.5)
        if base is not None:
            a.axhline(v[base], color=C_BASE, ls="--", lw=1.4, zorder=0)
        a.set_xticks(range(len(v)))
        a.set_xticklabels(labels, fontsize=11)
        a.set(title=title, ylabel=ylabel, ylim=(0, max(v) * 1.18))
        a.grid(alpha=0.3, axis="y")

    big = ["pfc_base_l1x4", "pfc_p3_l1x4", "pfc_p3_async_l1x4"]
    big_lab = ["8B alone\n(baseline)", "+ 3B P3\nstock save", "+ 3B P3\npatch"]
    big_c = [C_BASE, C_SOLO, C_ASYNC]
    panel(ax[0], big, big_lab, "big_tok_per_s", big_c, lambda y: f"{y:.0f}",
          "(a) 8B throughput: output tokens / s", "tokens / s", base=0)
    panel(ax[1], big, big_lab, "big_decode_queue_ms", big_c, lambda y: f"{y:.1f}",
          "(b) 8B queue wait in the decode engine", "seconds (mean per request)",
          scale=1e-3, base=0)
    panel(ax[2], big, big_lab, "big_ttft_p50_ms", big_c, lambda y: f"{y:.0f}",
          "(c) 8B TTFT p50 (its prefill shares GPU0)", "ms", base=0)

    sm = ["pfc_s3_v2", "pfc_s3_async_v2", "pfc_p3_l1x4", "pfc_p3_async_l1x4"]
    sm_lab = ["3B alone\nstock", "3B alone\npatch", "with 8B\nstock", "with 8B\npatch"]
    sm_c = [C_SOLO, C_ASYNC, C_SOLO, C_ASYNC]
    panel(ax[3], sm, sm_lab, "sm_ttft_p50_ms", sm_c, lambda y: f"{y:.0f}",
          "(d) 3B TTFT p50", "ms")
    panel(ax[4], sm, sm_lab, "sm_tpot_p50_ms", sm_c, lambda y: f"{y:.1f}",
          "(e) 3B TPOT p50 (decode on GPU0)", "ms per token")

    cells = ["pfc_base_l1x4", "pfc_s3_async_v2", "pfc_p3_l1x4", "pfc_p3_async_l1x4"]
    lab = ["8B alone", "3B alone\npatch", "8B + P3\nstock", "8B + P3\npatch"]
    xi = np.arange(len(cells))
    for k, (g, c) in enumerate((("gpu0", "#bf8700"), ("gpu1", "#8250df"))):
        v = [f(n, f"{g}_sm_active_mean") for n in cells]
        b = ax[5].bar(xi + (k - 0.5) * 0.38, v, 0.38, color=c, label=g.upper())
        for rect, y in zip(b, v):
            ax[5].text(rect.get_x() + rect.get_width() / 2, y, f"{y:.2f}", ha="center",
                       va="bottom", fontsize=10)
    ax[5].set_xticks(xi)
    ax[5].set_xticklabels(lab, fontsize=11)
    ax[5].set(title="(f) SM-active per GPU (DCGM)", ylabel="fraction of time (mean)",
              ylim=(0, 0.85))
    ax[5].legend(frameon=False, loc="upper left", ncol=2)
    ax[5].grid(alpha=0.3, axis="y")

    fig.suptitle("Placement P3: 3B prefill on GPU1 → DRAM → 3B decode on GPU0, beside the 8B "
                 "split", fontsize=15, fontweight="bold", y=1.0)
    fig.text(0.5, -0.03, "3B at 2 requests/s.  With the 8B the 8B's pinned host memory is 8 GiB "
             "(12 GiB elsewhere): it does not fit host RAM otherwise; the 8B-alone bar is at "
             "8 GiB too.\nCells with the 8B are swap-backed (2.3–2.8 GB swapped in): the 8B "
             "numbers are an upper bound on its cost.  Dark grey + dashed line = 8B alone; "
             "light grey = stock synchronous save; green = async-save patch.", ha="center", fontsize=11, color="#444")
    fig.tight_layout()
    fig.savefig(FIGS / "pfc_p3.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rate_figure()
    saturation_figure()
    async_figure()
    p3_figure()
    print("wrote figures/pfc_rate.png, pfc_saturation.png, pfc_async.png, pfc_p3.png")
