#!/usr/bin/env python3
"""N homogeneous models on one GPU, under MPS: how per-model latency and aggregate
throughput move as N grows, for the four serving cells (decode-only / full x fits /
offload). Reads output/summary_mc_<cohort>.csv (rebuilt from raw by multi_sweep.py).

  (a) per-model TPOT p50 (mean over models; whiskers = worst model)
  (b) per-model TTFT p50 (log; decode-only TTFT = queue + one-token step, no prefill)
  (c) aggregate throughput vs the offered rate (dashed), GPU1 SM-active on the right

    .venv/bin/python scripts/plots/plot_mc_scaling.py [--cohort small] [--arm mps]
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
OUT, FIGS = REPO / "output", REPO / "figures"
CELLS = [("dfits", "decode-only · fits", "#2e7d4f", "o", "-"),
         ("doff", "decode-only · offload", "#2e7d4f", "s", "--"),
         ("ffits", "full · fits", "#8250df", "o", "-"),
         ("foff", "full · offload", "#8250df", "s", "--")]
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "legend.fontsize": 12})


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(cohort, arm):
    rows = list(csv.DictReader(open(OUT / f"summary_mc_{cohort}.csv")))
    # canonical run per (cell, N); if it failed and a plain rerun (_r2) is clean, the
    # failure did not reproduce (e.g. the one client-side broken pipe at N=8) and the
    # rerun stands in. A failure that reproduces keeps its failed marker.
    by = {r["name"]: r for r in rows if r["arm"] == arm}
    out = {}
    for name, r in by.items():
        if name != f"mc_{cohort}_{r['cell']}_n{r['n']}_{arm}":
            continue
        rerun = by.get(name + "_r2")
        if r["ok"] != "True" and rerun and rerun["ok"] == "True":
            r = rerun
        out.setdefault(r["cell"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: int(r["n"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="small")
    ap.add_argument("--arm", default="mps")
    ap.add_argument("--offered", type=float, default=256.0,
                    help="offered output tok/s per model (2 QPS x 128 tokens)")
    args = ap.parse_args()
    data = load(args.cohort, args.arm)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    ax_tpot, ax_ttft, ax_tp = axes
    for cell, label, color, marker, ls in CELLS:
        rows = data.get(cell)
        if not rows:
            continue
        ns = [int(r["n"]) for r in rows]
        ok = [r["ok"] == "True" for r in rows]
        tpot = [f(r["tpot_p50_mean_ms"]) for r in rows]
        tpot_max = [f(r["tpot_p50_max_ms"]) for r in rows]
        ttft = [f(r["ttft_p50_mean_ms"]) for r in rows]
        tp = [f(r["tok_per_s_agg"]) for r in rows]
        kw = dict(color=color, marker=marker, ls=ls, lw=2, ms=8, label=label)
        ax_tpot.plot(ns, tpot, **kw)
        ax_tpot.vlines(ns, tpot, tpot_max, color=color, lw=1.2, alpha=0.6)
        ax_ttft.plot(ns, ttft, **kw)
        ax_tp.plot(ns, tp, **kw)
        for n, t, good, y in zip(ns, ttft, ok, tp):     # failed cells: hollow red ring
            if not good:
                ax_ttft.plot(n, t, "o", ms=16, mfc="none", mec="#cf222e", mew=2)
                ax_tp.plot(n, y, "o", ms=16, mfc="none", mec="#cf222e", mew=2)

    allns = sorted({int(r["n"]) for v in data.values() for r in v})
    ax_tp.plot(allns, [args.offered * n for n in allns], color="#57606a", ls=":", lw=1.5,
               label="offered (2 QPS/model)")
    sm = {int(r["n"]): f(r["gpu1_sm_active_mean"]) for r in data.get("dfits", [])}
    if sm:
        ax2 = ax_tp.twinx()
        ax2.plot(list(sm), list(sm.values()), color="#bc4c00", lw=1.5, ls="-.",
                 marker="^", ms=6, label="GPU1 SM-active (decode-only·fits)")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("GPU1 SM-active (fraction)", color="#bc4c00")
        ax2.tick_params(axis="y", colors="#bc4c00")
        ax2.legend(loc="upper left", frameon=False, fontsize=11)

    ax_tpot.set(title="(a) per-model TPOT p50", xlabel="N models on GPU1",
                ylabel="ms (log; whisker = worst model)", yscale="log")
    ax_ttft.set(title="(b) per-model TTFT p50", xlabel="N models on GPU1", ylabel="ms (log)",
                yscale="log")
    ax_tp.set(title="(c) aggregate throughput", xlabel="N models on GPU1",
              ylabel="output tok/s (all models)")
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    for ax, ticks in ((ax_tpot, [1, 2, 5, 10, 20, 50]), (ax_ttft, [10, 30, 100, 1e3, 1e4, 1e5])):
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        fmt = ScalarFormatter()
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)
    for ax in axes:
        ax.set_xticks(allns)
        ax.grid(alpha=0.3)
        if ax is ax_tp:
            ax.set_ylim(bottom=0)
    h, l = ax_tp.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIGS / f"mc_scaling_{args.cohort}_{args.arm}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
