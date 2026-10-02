#!/usr/bin/env python3
"""The N=4 decode-only-offload collapse, decomposed in time: stock vs CPU-thread-capped.

Same cell (4 x Qwen2.5-0.5B on GPU1, MPS, KV 3.2x over the grant -> every request
streams its prefix from DRAM), two runs that differ only in OMP_NUM_THREADS:
  stock   each engine's torch/OpenMP pool sized to all 32 cores  (mc_..._trace)
  capped  OMP_NUM_THREADS=4                                       (mc_..._omp4)

  (a) host CPU cores used by the 4 EngineCores (stores stay ~0 in both runs)
  (b) GPU1 SM-active
  (c) cumulative completed requests (all 4 models) vs offered arrivals

    .venv/bin/python scripts/plots/plot_mc_collapse.py
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
RAW, MON, FIGS = REPO / "output" / "raw", REPO / "output" / "gpumon", REPO / "figures"
# (label, raw cell, monitor tag, color, annotation offset/alignment)
RUNS = [("stock (default threads)", "mc_small_doff_n4_mps_trace", "mc_small_n4_mps_trace",
         "#cf222e", (-6, 14), "right", "bottom"),
        ("OMP_NUM_THREADS=4", "mc_small_doff_n4_mps_omp4", "mc_small_n4_mps_omp4",
         "#2e7d4f", (10, -6), "left", "top")]
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "legend.fontsize": 12})


def series(path, t0, t1, key_col, key, val_col, filt=None):
    out = []
    for r in csv.DictReader(open(path)):
        try:
            t = float(r["ts"])
        except (KeyError, ValueError):
            continue
        if t0 <= t <= t1 and r[key_col] == key:
            out.append((t - t0, float(r[val_col])))
    return out


def completions(rec):
    """(seconds since measure start, cumulative completed) over all models."""
    done = []
    for m in rec["models"]:
        c = m["client"]
        recs = [r for r in c.get("records", []) if r.get("t_submit") is not None]
        if not recs:
            continue
        base = min(r["t_submit"] for r in recs)
        start = c["t_start_epoch"]
        for r in recs:
            # completed = streamed to the end; a request that got some tokens and then
            # hit the 300 s client timeout has ttft+e2e set too, so exclude e2e >= 299.9 s
            if (r.get("e2e") is not None and r.get("ttft") is not None
                    and r["e2e"] < 299.9):
                done.append(start + (r["t_submit"] - base) + r["e2e"] - rec["t_measure0"])
    done.sort()
    return done


def main():
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax_cpu, ax_sm, ax_done = axes
    tmax = 0
    for label, raw, mon, color, off, ha, va in RUNS:
        rec = json.load(open(RAW / f"{raw}.json"))
        t0, t1 = rec["t_measure0"], rec["t_measure1"]
        tmax = max(tmax, t1 - t0)
        cpu = series(MON / f"{mon}_host.csv", t0, t1, "role", "enginecore", "cores")
        ax_cpu.plot(*zip(*cpu), color=color, lw=2, label=label)
        sm = series(MON / f"{mon}.csv", t0, t1, "gpu", "1", "sm_active")
        ax_sm.plot(*zip(*sm), color=color, lw=2, label=label)
        done = completions(rec)
        ax_done.step(done, range(1, len(done) + 1), where="post", color=color, lw=2,
                     label=label)
        nfail = sum((m["client"].get("summary") or {}).get("n_failed") or 0
                    for m in rec["models"])
        ax_done.annotate(f"{len(done)} completed, {nfail} timed out",
                         (done[-1], len(done)), xytext=off, textcoords="offset points",
                         color=color, fontsize=12, ha=ha, va=va)
    ax_cpu.axhline(32, color="#57606a", ls=":", lw=1.2)
    ax_cpu.text(2, 30.3, "all 32 host cores", color="#57606a", fontsize=11, va="top")
    ax_cpu.set(ylabel="cores", title="(a) host CPU used by the 4 EngineCores",
               ylim=(0, 34))
    ax_sm.set(ylabel="fraction", title="(b) GPU1 SM-active", ylim=(0, 1))
    ax_done.plot([0, 150], [0, 1200], color="#57606a", ls=":", lw=1.2,
                 label="offered (4 x 2 QPS)")
    ax_done.set(ylabel="requests", xlabel="seconds since measurement start",
                title="(c) cumulative completed requests, all 4 models", ylim=(0, 1300))
    for ax in axes:
        ax.grid(alpha=0.3)
    ax_done.set_xlim(0, tmax)
    h, l = ax_done.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGS / "mc_collapse_small_n4.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
