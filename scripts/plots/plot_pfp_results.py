#!/usr/bin/env python3
"""Ninth study: 8B-decode vs 3B-prefill priority on the shared GPU (scripts/inf_pf_prio/).

From output/summary_pfp.csv. Three configurations:
  A  bf16 3B + 8B decode b23   (suffix _b23)
  B  fp8  3B + 8B decode b23   (suffix _fp8b23; fair at 4 and 8 req/s only)
  C  fp8  3B + 8B decode b25   (suffix _fp8b25)

  pfp_prio_<cfg>_8b.png   8B priority: fair vs mps8b vs gate8b
  pfp_prio_<cfg>_3b.png   3B priority: fair vs mps3b vs gate3b
  pfp_configs.png         fair sharing, configs A, B, C

Each priority figure, x = 3B load (req/s): (a) 8B throughput, (b) 8B TPOT p50, (c) 8B
queue wait in its decode engine, (d) 3B TTFT p50, (e) 3B throughput, (f) GPU1
tensor-pipe active. Grey bar + dashed line = the model alone (baseline); for the 3B the
baseline is the 3B alone at the same rate. For gate3b the 8B decode runs eager: its own
baseline (8B alone, eager) is drawn as a dotted line.

    .venv/bin/python scripts/plots/plot_pfp_results.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
FIGS = REPO / "figures"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                     "legend.fontsize": 12})
ROWS = {r["name"]: r for r in csv.DictReader(open(REPO / "output" / "summary_pfp.csv"))}
RATES = (2, 4, 8)
CFGS = {"A": ("_b23", "config A: bf16 3B, 8B decode 23 GiB"),
        "C": ("_fp8b25", "config C: fp8 3B, 8B decode 25 GiB")}
# config B: the fp8 3B beside the 23 GiB 8B -- fair sharing at 4 and 8 req/s only
CROSS = ("_fp8b23", "config B: fp8 3B, 8B decode 23 GiB")
C_BASE = "#57606a"
ARM_STYLE = {"fair": ("#bf8700", "fair (MPS default)"),
             "mps8b": ("#1f6feb", "8B priority: MPS client priority"),
             "gate8b": ("#0b3d91", "8B priority: gating"),
             "gate8bt": ("#8fb8ff", "8B priority: gating, tight backlog (supplement)"),
             "mps3b": ("#2e9d4f", "3B priority: MPS client priority"),
             "gate3b": ("#145a2b", "3B priority: gating")}


def pick(name):
    """A cell whose window swapped > 0.5 GB was rerun (`_r2`); use whichever run swapped
    less (DECISIONS_PF_PRIO.md: rule set before the reruns)."""
    runs = [n for n in (name, name + "_r2", name + "_r3") if n in ROWS]
    if not runs:
        return None
    return min(runs, key=lambda n: float(ROWS[n].get("swap_in_pages") or 0))


def v(name, col, scale=1.0):
    r = ROWS.get(pick(name) or "")
    if not r or r.get(col) in (None, "", "None"):
        return np.nan
    return float(r[col]) * scale


def cell(cfg, kind, arm, q=None):
    sfx = CFGS[cfg][0]
    return f"pfp_{kind}_{arm}{'' if q is None else f'_q{q:g}'}{sfx}"


PANELS = [  # (column, scale, ylabel, title, which baseline)
    ("big_tok_per_s", 1, "tokens / s", "(a) 8B throughput", "big"),
    ("big_tpot_p50_ms", 1, "ms", "(b) 8B TPOT p50", "big"),
    ("big_decode_queue_ms", 1e-3, "seconds (mean per request)",
     "(c) 8B queue wait in its decode engine", "big"),
    ("sm_ttft_p50_ms", 1, "ms", "(d) 3B TTFT p50", "small"),
    ("sm_req_per_s", 1, "requests / s", "(e) 3B throughput", "small"),
    ("gpu1_tensor_active_mean", 1, "fraction of time (mean)",
     "(f) GPU1 tensor-pipe active (DCGM)", "big"),
]


def label(ax, rects, vals, fmt):
    for r, y in zip(rects, vals):
        if np.isfinite(y):
            ax.text(r.get_x() + r.get_width() / 2, y, fmt(y), ha="center", va="bottom",
                    fontsize=8.5, rotation=90 if fmt(y).__len__() > 4 else 0)


def prio_figure(cfg, arms, tag, title):
    fig, axes = plt.subplots(2, 3, figsize=(19, 9.5))
    axes = axes.ravel()
    xi = np.arange(len(RATES))
    w = 0.8 / (len(arms) + 1)
    for ax, (col, sc, yl, tt, base) in zip(axes, PANELS):
        fmt = (lambda y: f"{y:.2f}") if sc != 1 or col.startswith(("gpu", "sm_req")) else \
              (lambda y: f"{y:.0f}")
        if col == "gpu1_tensor_active_mean":   # reference: what both alone add up to
            ssum = [v(cell(cfg, "base", "fair"), col) + v(cell(cfg, "s2", "fair", q), col)
                    for q in RATES]
            ax.scatter(xi - 0.4 + w / 2, ssum, marker="_", s=500, color="k", zorder=5, linewidths=2.5,
                       label="8B alone + 3B alone (sum), drawn over the grey bar")
        if base == "big":
            b = v(cell(cfg, "base", "fair"), col, sc)
            ax.axhline(b, color=C_BASE, ls="--", lw=1.3, zorder=0, label="model alone (baseline)")
            if "gate3b" in arms and col.startswith("big_"):
                be = v(cell(cfg, "base", "eager8b"), col, sc)
                ax.axhline(be, color=ARM_STYLE["gate3b"][0], ls=":", lw=1.5, zorder=0,
                           label="8B alone, eager decode (gate3b's baseline)")
            vals = [b] * len(RATES)
        else:
            vals = [v(cell(cfg, "s2", "fair", q), col, sc) for q in RATES]
        rb = ax.bar(xi - 0.4 + w / 2, vals, w, color=C_BASE)
        label(ax, rb, vals, fmt)
        for k, arm in enumerate(arms):
            vals = [v(cell(cfg, "p2", arm, q), col, sc) for q in RATES]
            r = ax.bar(xi - 0.4 + w * (k + 1.5), vals, w, color=ARM_STYLE[arm][0],
                       label=ARM_STYLE[arm][1])
            label(ax, r, vals, fmt)
        ax.set_xticks(xi)
        ax.set_xticklabels([f"{q} req/s" for q in RATES])
        ax.set(xlabel="3B load", ylabel=yl, title=tt)
        top = np.nanmax([p.get_height() for p in ax.patches] + [0]
                        + [y for c in ax.collections for y in c.get_offsets()[:, 1]])
        low = np.nanmin([p.get_height() for p in ax.patches if p.get_height() > 0] + [top])
        if top > 50 * low:            # a saturated cell would flatten the rest: log axis
            ax.set_yscale("log")
            ax.set_ylim(low * 0.5, top * 3)
            ax.set_ylabel(yl + " (log)")
        else:
            ax.set_ylim(0, top * 1.22 if top > 0 else 1)
        ax.grid(alpha=0.3, axis="y")
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    # two legend groups: what is measured (one bar colour per arm) and the references
    arm_lab = {ARM_STYLE[a][1] for a in arms}
    ha = [hh for hh, ll in zip(h, l) if ll in arm_lab]
    la = [ll for ll in l if ll in arm_lab]
    ref_names = {"model alone (baseline)": "model alone (dashed line)",
                 "8B alone, eager decode (gate3b's baseline)": "8B alone, eager decode (dotted; gate3b's baseline)",
                 "8B alone + 3B alone (sum), drawn over the grey bar": "8B alone + 3B alone, summed (black tick, panel f)"}
    hr = [plt.Rectangle((0, 0), 1, 1, color=C_BASE)]
    lr = ["alone (grey bar; 3B: alone at that rate)"]
    for hh, ll in zip(h, l):
        if ll in ref_names:
            hr.append(hh); lr.append(ref_names[ll])
    fig.legend(ha, la, loc="upper left", ncol=1, frameon=True, title="colocated, by priority arm",
               title_fontproperties={"weight": "bold"}, bbox_to_anchor=(0.12, 1.11))
    fig.legend(hr, lr, loc="upper right", ncol=1, frameon=True, title="references (run alone)",
               title_fontproperties={"weight": "bold"}, bbox_to_anchor=(0.88, 1.11))
    fig.suptitle(f"P2, {title} — {CFGS[cfg][1]}", fontsize=15, fontweight="bold", y=1.16)
    fig.text(0.5, -0.02, "8B: split, 2 requests/s, reference workload.  3B: fresh 2048-token "
             "prompt per request, prefill on GPU1 (the 8B's decode GPU), KV to DRAM "
             "(async save).  All under MPS.", ha="center", fontsize=11, color="#444")
    fig.tight_layout()
    fig.savefig(FIGS / f"pfp_prio_{cfg}_{tag}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def config_figure():
    fig, axes = plt.subplots(2, 3, figsize=(19, 9.5))
    axes = axes.ravel()
    xi = np.arange(len(RATES))
    cols = {"A": "#bf8700", "X": "#d4a0a0", "C": "#8250df"}
    for ax, (col, sc, yl, tt, base) in zip(axes, PANELS):
        for k, cfg in enumerate(("A", "X", "C")):
            if base == "big" and cfg != "X":
                ax.axhline(v(cell(cfg, "base", "fair"), col, sc), color=cols[cfg], ls="--",
                           lw=1.2, zorder=0)
            if cfg == "X":
                vals = [v(f"pfp_p2_fair_q{q}{CROSS[0]}", col, sc) for q in RATES]
                lab = CROSS[1]
            else:
                vals = [v(cell(cfg, "p2", "fair", q), col, sc) for q in RATES]
                lab = CFGS[cfg][1]
            r = ax.bar(xi + (k - 1) * 0.27, vals, 0.27, color=cols[cfg], label=lab)
            label(ax, r, vals, (lambda y: f"{y:.2f}") if sc != 1 or col.startswith(("gpu", "sm_req"))
                  else (lambda y: f"{y:.0f}"))
        ax.set_xticks(xi)
        ax.set_xticklabels([f"{q} req/s" for q in RATES])
        ax.set(xlabel="3B load", ylabel=yl, title=tt)
        top = np.nanmax([p.get_height() for p in ax.patches] + [0])
        ax.set_ylim(0, top * 1.22 if top > 0 else 1)
        ax.grid(alpha=0.3, axis="y")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("P2, fair sharing: configs A, B, C (dashed = 8B alone at 23 GiB / 25 GiB)",
                 fontsize=15, fontweight="bold", y=1.06)
    fig.tight_layout()
    fig.savefig(FIGS / "pfp_configs.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for cfg in CFGS:
        prio_figure(cfg, ["fair", "mps8b", "gate8b", "gate8bt"], "8b", "8B priority")
        prio_figure(cfg, ["fair", "mps3b", "gate3b"], "3b", "3B priority")
    config_figure()
    print("wrote figures/pfp_prio_{A,C}_{8b,3b}.png, pfp_configs.png")
