#!/usr/bin/env python3
"""Figures for the Case A inference sweep. Reads every output/summary_*.csv into one frame.

    python scripts/plot_case_a.py
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT, FIGS = REPO / "output", REPO / "figures"

WEIGHTS_GIB = 14.96
BLUE, RED, PURPLE, GREY = "#1f6feb", "#c1440e", "#8250df", "#57606a"


def load():
    rows = []
    for p in sorted(OUT.glob("summary_*.csv")):
        if any(t in p.name for t in ("smoke", "pilot", "rebuilt")):
            continue
        for r in csv.DictReader(open(p)):
            if not r.get("gpu_kv_gib"):
                continue
            rows.append(r)
    # Later files win: a config re-run under a new tag supersedes the earlier attempt.
    dedup = {}
    for r in rows:
        dedup[r["name"]] = r
    return list(dedup.values())


def series(rows, skew, arm):
    s = [r for r in rows if r["skew"] == skew and r["arm"] == arm]
    s.sort(key=lambda r: float(r["budget_gib"]))
    return s


def fnum(r, k):
    v = r.get(k)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def style_budget_axis(ax, title, ylabel):
    ax.set_xlabel("VRAM budget given to vLLM (GiB)   —   tighter to the right")
    ax.invert_xaxis()
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, which="both")


def fig_main(rows, cal):
    """The headline: TTFT and throughput vs budget, both access patterns."""
    fig, axgrid = plt.subplots(2, 2, figsize=(17, 7.5), sharex="col",
                               gridspec_kw={"height_ratios": [1.9, 1], "hspace": 0.12,
                                            "wspace": 0.14})
    axes = axgrid[0]
    for ax, skew in zip(axes, ("zipf", "uniform")):
        off = series(rows, skew, "offload")
        x = [fnum(r, "budget_gib") for r in off]
        p50 = [fnum(r, "ttft_p50_ms") for r in off]
        p95 = [fnum(r, "ttft_p95_ms") for r in off]
        hung = [i for i, r in enumerate(off) if r.get("hung") == "True"]

        ax.plot(x, p50, color=BLUE, marker="o", lw=2, label="TTFT p50")
        ax.plot(x, p95, color=BLUE, marker="^", ls="--", lw=1.2, alpha=0.7, label="TTFT p95")
        ax.fill_between(x, p50, p95, color=BLUE, alpha=0.10)
        if hung:
            ax.scatter([x[i] for i in hung], [p50[i] for i in hung], s=170,
                       facecolors="none", edgecolors=RED, lw=2, zorder=5,
                       label="engine stalled (run aborted)")

        if cal:
            t = cal["derived"]["t_load_8k_prefix_ms"] * 0.75  # a 6144-token prefix
            ax.axhline(t, color=GREY, lw=1, ls=":")
            ax.annotate(f"PCIe fetch of one prefix ≈ {t:.0f} ms", xy=(x[-1], t),
                        xytext=(4, 4), textcoords="offset points", fontsize=8, color=GREY,
                        ha="left")
        ax.axhline(530, color=GREY, lw=1, ls=":")
        ax.annotate("recompute ≈ 530 ms", xy=(x[-1], 530), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color=GREY, ha="left")

        ax.set_yscale("log")
        style_budget_axis(ax, f"{skew} access", "TTFT (ms, log)")
        ax.legend(fontsize=8, loc="upper left")
    axes[1].set_ylabel("")

    # Throughput row. Flat at the offered rate everywhere the server keeps up -- which is the
    # point: below saturation, throughput measures the load generator, not the server.
    for ax, skew in zip(axgrid[1], ("zipf", "uniform")):
        off = series(rows, skew, "offload")
        x = [fnum(r, "budget_gib") for r in off]
        tps = [fnum(r, "output_tok_per_s") for r in off]
        ax.plot(x, tps, color=BLUE, marker="o", lw=2)
        ax.axhline(256.0, color=GREY, lw=1, ls="--")
        ax.annotate("offered load: 2 req/s × 128 tok = 256 tok/s", xy=(x[-1], 256),
                    xytext=(4, -13), textcoords="offset points", fontsize=8, color=GREY,
                    ha="left")
        hung = [i for i, r in enumerate(off) if r.get("hung") == "True"]
        if hung:
            ax.scatter([x[i] for i in hung], [tps[i] for i in hung], s=170,
                       facecolors="none", edgecolors=RED, lw=2, zorder=5)
        ax.set_ylim(0, 300)
        # No invert_xaxis() here: sharex="col" already propagates the top row's inversion,
        # and inverting a shared axis twice cancels out.
        ax.set_xlabel("VRAM budget given to vLLM (GiB)   —   tighter to the right")
        ax.grid(alpha=0.25)
    axgrid[1][0].set_ylabel("output throughput (tok/s)")
    for ax in axes:
        ax.set_xlabel("")

    fig.suptitle("Time to first token vs VRAM budget, with DRAM KV offloading — Llama-3-8B, 24 GiB KV working set, "
                 "RTX 5090 (PCIe Gen4 ×8)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = FIGS / "case_a_ttft.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_tiers(rows):
    """Where the KV comes from, and the preemptions that mark the concurrency wall."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, skew in zip(axes, ("zipf", "uniform")):
        off = series(rows, skew, "offload")
        x = [fnum(r, "budget_gib") for r in off]
        ax.plot(x, [(fnum(r, "gpu_hit_rate") or 0) * 100 for r in off],
                color=BLUE, marker="o", label="served from GPU cache")
        ax.plot(x, [(fnum(r, "dram_hit_rate") or 0) * 100 for r in off],
                color=PURPLE, marker="^", ls="--", label="served from DRAM tier")
        style_budget_axis(ax, f"{skew} — where prefix hits come from", "share of queries (%)")

        ax2 = ax.twinx()
        pre = [fnum(r, "preemptions") or 0 for r in off]
        ax2.bar(x, pre, width=0.6, color=RED, alpha=0.20)
        ax2.set_ylabel("preemptions (bars)", color=RED, fontsize=9)
        ax.legend(fontsize=8, loc="center left")
    fig.suptitle("The DRAM tier absorbs the shrinking GPU cache — until preemption starts",
                 fontsize=12)
    fig.tight_layout()
    p = FIGS / "case_a_tiers.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_walls(rows):
    """TTFT against GPU KV, with the computed concurrency requirement marked."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for skew, c, m in (("zipf", BLUE, "o"), ("uniform", PURPLE, "s")):
        off = series(rows, skew, "offload")
        ax.plot([fnum(r, "gpu_kv_gib") for r in off], [fnum(r, "ttft_p50_ms") for r in off],
                color=c, marker=m, lw=2, label=f"{skew} — offload")

    ax.axvspan(0, 5.36, color=RED, alpha=0.07)
    ax.axvline(5.36, color=RED, lw=1.5, ls="--")
    ax.annotate("concurrency wall: 5.36 GiB\n(Poisson p95 of 7 in flight × 0.766 GiB)",
                xy=(5.36, 3000), xytext=(5.7, 3000), fontsize=9, color=RED,
                va="center", ha="left")
    ax.axvline(12.75, color=GREY, lw=1, ls=":")
    ax.annotate("zipf hot set 12.75 GiB", xy=(12.75, 6000), xytext=(-6, 0),
                textcoords="offset points", fontsize=8, color=GREY,
                rotation=90, va="center", ha="right")

    ax.set_yscale("log")
    ax.invert_xaxis()  # match the other figures: generous VRAM on the left, tightest on the right
    ax.set_xlabel("GPU KV cache (GiB)   —   tighter to the right")
    ax.set_ylabel("TTFT p50 (ms, log)")
    ax.set_title("The wall is concurrency, not cache capacity or bandwidth", fontsize=11)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = FIGS / "case_a_walls.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main():
    FIGS.mkdir(exist_ok=True)
    rows = load()
    cal_p = OUT / "calibration_pcie.json"
    cal = json.loads(cal_p.read_text()) if cal_p.exists() else None
    print(f"{len(rows)} configs loaded")
    for p in (fig_main(rows, cal), fig_tiers(rows), fig_walls(rows)):
        print(f"wrote {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
