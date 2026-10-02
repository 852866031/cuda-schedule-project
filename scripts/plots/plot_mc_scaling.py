#!/usr/bin/env python3
"""N homogeneous models on one GPU, under MPS: how per-model latency and aggregate
throughput move as N grows. One figure per working-set GROUP:
  fits     decode-only·fits + full·fits (KV resident)
  offload  decode-only·offload + full·offload (KV 3x over the grant, streams from DRAM)
Each cell's thread-capped variant (OMP_NUM_THREADS=4, name suffix _omp4), where it
exists, is drawn as its own dashed, hollow-marker series next to the stock one. Reads
output/summary_mc_<cohort>.csv (rebuilt from raw by multi_sweep.py).

  (a) per-model TPOT p50 (mean over models; whiskers = worst model)
  (b) per-model TTFT p50 (log; decode-only TTFT = queue + one-token step, no prefill)
  (c) aggregate throughput vs the offered rate (dotted), GPU1 SM-active on the right, and
      GPU1 memory used as a filled band at the bottom on its own offset axis -- the three
      quantities are given disjoint vertical bands via axis limits, so nothing overlaps

    .venv/bin/python scripts/plots/plot_mc_scaling.py --cohort small --group fits
    .venv/bin/python scripts/plots/plot_mc_scaling.py --cohort small --group offload
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
OUT, FIGS = REPO / "output", REPO / "figures"
GPU_GIB = 31.35
GROUPS = {"fits": ("dfits", "ffits"), "offload": ("doff", "foff")}
LABEL = {"dfits": "decode-only", "ffits": "full", "doff": "decode-only", "foff": "full"}
COLOR = {"dfits": "#2e7d4f", "doff": "#2e7d4f", "ffits": "#8250df", "foff": "#8250df"}
CAP = "_omp4"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "legend.fontsize": 12})


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(cohort, arm):
    """{(cell, variant): [rows by N]}; variant "" = stock threads, CAP = thread-capped.
    Canonical run per (cell, N, variant); if it failed and a plain rerun (_r2) is clean,
    the failure did not reproduce (e.g. the one client-side broken pipe at N=8) and the
    rerun stands in. A failure that reproduces keeps its failed marker."""
    rows = list(csv.DictReader(open(OUT / f"summary_mc_{cohort}.csv")))
    by = {r["name"]: r for r in rows if r["arm"] == arm}
    out = {}
    canon = {}
    for name, r in by.items():
        base = f"mc_{cohort}_{r['cell']}_n{r['n']}_{arm}"
        if name not in (base, base + CAP):
            continue
        variant = name[len(base):]
        canon.setdefault((r["cell"], variant), []).append(r)
        rerun = by.get(name + "_r2")
        if r["ok"] != "True" and rerun and rerun["ok"] == "True":
            r = rerun
        out.setdefault((r["cell"], variant), []).append(r)
    for d in (out, canon):
        for v in d.values():
            v.sort(key=lambda r: int(r["n"]))
    return out, canon


def log_ticks(ax, lo, hi):
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    cands = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5]
    ticks = [t for t in cands if lo / 1.5 <= t <= hi * 1.5] or cands
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="small")
    ap.add_argument("--arm", default="mps")
    ap.add_argument("--group", choices=list(GROUPS), required=True)
    ap.add_argument("--offered", type=float, default=256.0,
                    help="offered output tok/s per model (2 QPS x 128 tokens)")
    args = ap.parse_args()
    data, canon = load(args.cohort, args.arm)
    data = {k: v for k, v in data.items() if k[0] in GROUPS[args.group]}
    has_cap = any(var == CAP for _, var in data) and any(var == "" for _, var in data)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    ax_tpot, ax_ttft, ax_tp = axes
    lat = {"tpot": [], "ttft": []}
    for cell in GROUPS[args.group]:
      for variant in ("", CAP):
        rows = data.get((cell, variant))
        if not rows:
            continue
        color = COLOR[cell]
        capped = variant == CAP
        label = LABEL[cell] + (" · thread-capped" if capped and has_cap
                               else " · stock threads" if has_cap else "")
        marker = "s" if cell.startswith("f") else "o"
        ns = [int(r["n"]) for r in rows]
        ok = [r["ok"] == "True" for r in rows]
        tpot = [f(r["tpot_p50_mean_ms"]) for r in rows]
        tpot_max = [f(r["tpot_p50_max_ms"]) for r in rows]
        ttft = [f(r["ttft_p50_mean_ms"]) for r in rows]
        tp = [f(r["tok_per_s_agg"]) for r in rows]
        lat["tpot"] += [x for x in tpot + tpot_max if x]
        lat["ttft"] += [x for x in ttft if x]
        kw = dict(color=color, marker=marker, ls="--" if capped else "-", lw=2, ms=8,
                  label=label, mfc="white" if capped else color)
        ax_tpot.plot(ns, tpot, **kw)
        ax_tpot.vlines(ns, tpot, tpot_max, color=color, lw=1.2, alpha=0.6)
        ax_ttft.plot(ns, ttft, **kw)
        ax_tp.plot(ns, tp, **kw)
        for n, t, good, y in zip(ns, ttft, ok, tp):     # failed cells: hollow red ring
            if not good:
                ax_ttft.plot(n, t, "o", ms=16, mfc="none", mec="#cf222e", mew=2)
                ax_tp.plot(n, y, "o", ms=16, mfc="none", mec="#cf222e", mew=2)

    allns = sorted({int(r["n"]) for v in data.values() for r in v})
    main_cell = GROUPS[args.group][0]
    ax_tp.plot(allns, [args.offered * n for n in allns], color="#57606a", ls=":", lw=1.5,
               label="offered (2 QPS/model)")
    # SM-active of the decode-only cell, from the CANONICAL run (a GPU counter is not
    # invalidated by one client-side failure, while the N=8 rerun's transient CPU spin
    # storm depresses its mean): stock where measured, capped beyond -- one line
    sm = {int(r["n"]): f(r["gpu1_sm_active_mean"]) for r in canon.get((main_cell, CAP), [])}
    sm.update({int(r["n"]): f(r["gpu1_sm_active_mean"])
               for r in canon.get((main_cell, ""), [])})
    sm = dict(sorted(sm.items()))
    vram = {int(r["n"]): f(r["gpu1_fb_used_max_gib"]) for r in canon.get((main_cell, CAP), [])}
    vram.update({int(r["n"]): f(r["gpu1_fb_used_max_gib"])
                 for r in canon.get((main_cell, ""), [])})
    vram = dict(sorted(vram.items()))

    # Disjoint vertical bands in panel (c): memory fill in the bottom BAND_MEM of the
    # height; throughput and SM-active start at BAND_TOP0 (their zero) and use the rest.
    BAND_MEM, BAND_TOP0 = 0.25, 0.30
    tp_max = max(max(f(r["tok_per_s_agg"]) or 0 for r in v) for v in data.values())
    tp_max = max(tp_max, args.offered * max(allns)) * 1.06
    ax_tp.set_ylim(-BAND_TOP0 / (1 - BAND_TOP0) * tp_max, tp_max)
    ax_tp.set_yticks([t for t in ax_tp.get_yticks() if 0 <= t <= tp_max])
    if sm:
        ax2 = ax_tp.twinx()
        ax2.plot(list(sm), list(sm.values()), color="#bc4c00", lw=1.5, ls="-.",
                 marker="^", ms=6, label=f"GPU1 SM-active (decode-only·{args.group})")
        span = 1.05 / (1 - BAND_TOP0)
        ax2.set_ylim(-BAND_TOP0 * span, 1.05)
        ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2.set_ylabel("GPU1 SM-active", color="#bc4c00")
        ax2.yaxis.set_label_coords(1.14, 0.65)
        ax2.tick_params(axis="y", colors="#bc4c00")
        ax2.legend(loc="upper left", frameon=False, fontsize=11)
    if vram:
        # memory shares the LEFT spine: its ticks live only in the bottom band, where the
        # throughput axis (zero at BAND_TOP0) has none -- no third spine, no collisions
        ax3 = ax_tp.twinx()
        ax3.yaxis.tick_left()
        ax3.yaxis.set_label_position("left")
        for side in ("right", "top"):
            ax3.spines[side].set_visible(False)
        ns_v, gib = list(vram), list(vram.values())
        ax3.fill_between(ns_v, 0, gib, color="#1f6feb", alpha=0.18, lw=0)
        ax3.plot(ns_v, gib, color="#1f6feb", lw=1.2, label="GPU1 memory used")
        ax3.axhline(GPU_GIB, color="#1f6feb", ls=":", lw=1.0)
        ax3.text(max(allns), GPU_GIB, f"capacity {GPU_GIB:g} GiB ", color="#1f6feb",
                 fontsize=10, va="bottom", ha="right")
        ax3.set_ylim(0, 32 / BAND_MEM)
        ax3.set_yticks([0, 15, 30])
        ax3.set_ylabel("GiB", color="#1f6feb")
        ax3.yaxis.set_label_coords(-0.13, 0.12)
        ax3.tick_params(axis="y", colors="#1f6feb")
        ax3.legend(loc="upper left", frameon=False, fontsize=11,
                   bbox_to_anchor=(0.0, BAND_MEM + 0.01))

    # log only when the data spans >10x (offload's collapse); linear otherwise
    for ax, key, title in ((ax_tpot, "tpot", "(a) per-model TPOT p50"),
                           (ax_ttft, "ttft", "(b) per-model TTFT p50")):
        lo, hi = min(lat[key]), max(lat[key])
        if hi / lo > 10:
            ax.set_yscale("log")
            log_ticks(ax, lo, hi)
            unit = "ms (log)"
        else:
            ax.set_ylim(0, hi * 1.12)
            unit = "ms"
        extra = "; whisker = worst model" if key == "tpot" else ""
        ax.set(title=title, xlabel="N models on GPU1", ylabel=unit + extra)
    ax_tp.yaxis.set_label_coords(-0.13, 0.65)
    ax_tp.set(title="(c) throughput · SM-active · GPU1 memory", xlabel="N models on GPU1",
              ylabel="tok/s (all models)")

    for ax in axes:
        ax.set_xticks(allns)
        ax.grid(alpha=0.3)

    h, l = ax_tp.get_legend_handles_labels()
    title = {"fits": "KV resident (fits)", "offload": "KV 3× over the grant (offload)"}
    fig.suptitle(f"{args.cohort} cohort · {title[args.group]}", y=0.995, fontsize=15)
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = FIGS / f"mc_scaling_{args.cohort}_{args.arm}_{args.group}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
