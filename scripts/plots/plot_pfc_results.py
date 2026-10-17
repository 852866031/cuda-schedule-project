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


def rate_figure():
    rates = ["_q0.5", "_q1", ""]
    x = {pl: [0.0] + [f(f"pfc_{pl}{q}", "sm_req_per_s") for q in rates] for pl in ("p1", "p2")}
    xs = {pl: [f(f"pfc_{pl}{q}", "sm_req_per_s") for q in rates] for pl in ("s1", "s2")}
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.8))

    for pl, c, lab in (("p1", C_P1, "+ 3B prefill only (P1)"),
                       ("p2", C_P2, "+ 3B prefill, KV to DRAM (P2)")):
        tok = [f("pfc_base", "big_tok_per_s")] + [f(f"pfc_{pl}{q}", "big_tok_per_s") for q in rates]
        qw = [f("pfc_base", "big_decode_queue_ms")] + [f(f"pfc_{pl}{q}", "big_decode_queue_ms") for q in rates]
        sm = [f("pfc_base", "gpu1_sm_active_mean")] + [f(f"pfc_{pl}{q}", "gpu1_sm_active_mean") for q in rates]
        ax[0].plot(x[pl], tok, "o-", color=c, lw=2, label=lab)
        ax[1].plot(x[pl], np.array(qw) / 1000, "o-", color=c, lw=2)
        ax[3].plot(x[pl], sm, "o-", color=c, lw=2)
        ttft = [f(f"pfc_{pl}{q}", "sm_ttft_p50_ms") for q in rates]
        ax[2].plot(x[pl][1:], ttft, "o-", color=c, lw=2)
    for pl, c in (("s1", C_P1), ("s2", C_P2)):
        ttft = [f(f"pfc_{pl}{q}", "sm_ttft_p50_ms") for q in rates]
        ax[2].plot(xs[pl], ttft, "o--", color=c, lw=1.5, mfc="white")
        sm = [f(f"pfc_{pl}{q}", "gpu1_sm_active_mean") for q in rates]
        ax[3].plot(xs[pl], sm, "o--", color=c, lw=1.5, mfc="white")
    ax[2].plot([], [], "k--", lw=1.5, label="3B alone (dashed, open)")

    ts = [r["t_submit"] for r in json.loads((RAW / "pfc_base.json").read_text())["big_records"]]
    off = 300 * 128 / (max(ts) - min(ts))      # offered tok/s over the arrival span
    ax[0].axhline(off, color="k", ls=":", lw=1)
    ax[0].text(2.15, off + 2, "offered", ha="right", va="bottom", fontsize=11)
    ax[0].set(ylabel="8B output tok/s", ylim=(190, 275), title="(a) 8B throughput")
    ax[1].set(ylabel="8B decode queue wait, mean (s)", yscale="log", ylim=(0.5, 60),
              title="(b) 8B decode-engine queue")
    ax[2].set(ylabel="3B TTFT p50 (ms)", ylim=(0, 950), title="(c) 3B time to first token")
    ax[3].set(ylabel="GPU1 SM-active (mean)", ylim=(0, 0.75), title="(d) GPU1 compute")
    for a in ax:
        a.set_xlabel("3B request rate (req/s)")
        a.set_xlim(-0.1, 2.3)
        a.grid(alpha=0.3)
    h, l = [], []
    for a in (ax[0], ax[2]):
        hh, ll = a.get_legend_handles_labels()
        h += hh; l += ll
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("8B split (decode budget 22 GiB, 2 QPS) + Qwen2.5-3B prefill on the 8B's decode GPU",
                 y=1.09, fontsize=15)
    fig.subplots_adjust(wspace=0.38)
    fig.savefig(FIGS / "pfc_rate.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def saturation_figure():
    cells = [("pfc_base", "8B alone", "k"), ("pfc_p1_q0.5", "+ 3B 0.5 req/s", "#7aa7e8"),
             ("pfc_p1_q1", "+ 3B 1 req/s", C_P1), ("pfc_p1", "+ 3B 2 req/s", "#0b3d91")]
    fig, ax = plt.subplots(figsize=(9, 5))
    for nm, lab, c in cells:
        recs = json.loads((RAW / f"{nm}.json").read_text())["big_records"]
        recs = sorted((r for r in recs if r.get("e2e") and r.get("ttft")), key=lambda r: r["t_submit"])
        t0 = recs[0]["t_submit"]
        t = np.array([r["t_submit"] - t0 for r in recs])
        y = np.array([(r["e2e"] - r["ttft"]) / 127 * 1000 for r in recs])
        k = 15
        ys = np.convolve(y, np.ones(k) / k, mode="valid")
        ax.plot(t[k - 1:], ys, color=c, lw=2, label=f"{lab}  (tok/s {f(nm, 'big_tok_per_s'):.0f})")
    ax.set(xlabel="8B request arrival time in the window (s)",
           ylabel="ms per output token after token #1\n(rolling mean of 15 requests)",
           yscale="log", title="8B requests wait longer as the window goes on (P1)")
    ax.set_ylim(18, 1000)
    ax.set_yticks([20, 50, 100, 200, 400])
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "pfc_saturation.png", dpi=130)
    plt.close(fig)


def async_figure():
    pairs = [("3B alone\nP2", "pfc_s2", "pfc_s2_async"),
             ("3B alone\nP3", "pfc_s3_v2", "pfc_s3_async_v2"),
             ("with 8B\nP2", "pfc_p2", "pfc_p2_async_r2"),
             ("with 8B\nP3 (L1 4 GiB)", "pfc_p3_l1x4", "pfc_p3_async_l1x4")]
    ref = f("pfc_s1", "sm_ttft_p50_ms")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [2.2, 1]})
    xi = np.arange(len(pairs))
    w = 0.38
    for j, (col, c, lab) in enumerate(((1, C_SOLO, "synchronous save (stock LMCache)"),
                                        (2, C_ASYNC, "async-save patch"))):
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
    ax[0].set(xticks=xi, xticklabels=[p[0] for p in pairs], ylabel="3B TTFT (ms): bar p50, whisker p95",
              ylim=(0, 2300), title="(a) 3B time to first token, 2 req/s")
    ax[0].grid(alpha=0.3, axis="y")

    hit = []
    for nm in ("pfc_s3_v", "pfc_s3_v2", "pfc_s3_async", "pfc_s3_async_v2", "pfc_p3_async_l1x4"):
        d = json.loads((RAW / f"{nm}.json").read_text()).get("decode_hits") or {}
        hit.append(100 * d["full_hits"] / d["n"])
    lab = ["sync\nrun 1", "sync\nrun 2", "async\nrun 1", "async\nrun 2", "async\nwith 8B"]
    ax[1].bar(range(5), hit, color=[C_SOLO, C_SOLO, C_ASYNC, C_ASYNC, C_ASYNC])
    for i, v in enumerate(hit):
        ax[1].text(i, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=11)
    ax[1].set(xticks=range(5), xticklabels=lab, ylim=(0, 112),
              ylabel="decode requests that found\ntheir whole prompt in DRAM (%)",
              title="(b) P3: KV ready for the decode leg")
    ax[1].grid(alpha=0.3, axis="y")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(FIGS / "pfc_async.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rate_figure()
    saturation_figure()
    async_figure()
    print("wrote figures/pfc_rate.png, pfc_saturation.png, pfc_async.png")
