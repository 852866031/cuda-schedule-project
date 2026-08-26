#!/usr/bin/env python3
"""Same-card vs split-card, uniform access, identical Case A workload.

    python scripts/plots/plot_disagg.py

Same-card numbers come from the inference study's uniform arm (summary_uniform_off.csv);
split-card from summary_disagg.csv. Both ran the same 300-request Poisson stream at 2 QPS
over the same 32-session / 24 GiB working set, so the x axis means the same thing on both
curves: the VRAM budget given to the card that decodes.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
OUT, FIGS = REPO / "output", REPO / "figures"

WEIGHTS_GIB = 14.96
BLUE, RED, PURPLE, GREY = "#1f6feb", "#c1440e", "#8250df", "#57606a"


def fnum(r, k):
    try:
        return float(r.get(k))
    except (TypeError, ValueError):
        return None


def read(path):
    p = OUT / path
    return list(csv.DictReader(open(p))) if p.exists() else []


def load():
    same = [r for r in read("summary_uniform_off.csv") if r.get("skew") == "uniform"]
    split = read("summary_disagg.csv")
    for s in (same, split):
        s.sort(key=lambda r: float(r["budget_gib"]))
    return same, split


def healthy(r):
    """A point is a measurement only if the server stayed up and served every request."""
    if str(r.get("ok")).lower() != "true":
        return False
    if str(r.get("hung")).lower() == "true":
        return False
    return (fnum(r, "n_failed") or 0) == 0


def split_series(rows, key):
    """(budgets, values) for healthy points, and the same for failed ones."""
    ok = [(fnum(r, "budget_gib"), fnum(r, key)) for r in rows if healthy(r)]
    bad = [(fnum(r, "budget_gib"), fnum(r, key)) for r in rows if not healthy(r)]
    f = lambda s: ([a for a, b in s if b is not None], [b for a, b in s if b is not None])
    return f(ok), f(bad)


def draw(ax, rows, key, colour, label, marker="o"):
    (bx, by), (fx, fy) = split_series(rows, key)
    if bx:
        ax.plot(bx, by, marker=marker, color=colour, label=label, lw=1.8, ms=5)
    if fx:
        # Hollow markers: the server failed here, so the value is an artefact of the
        # failure, not a measurement of it.
        ax.plot(fx, fy, marker="x", color=colour, ls=":", lw=1.2, ms=7, alpha=0.75,
                label=f"{label} (failed)")
    return bx, by


def style(ax, ylabel, title, log=False):
    ax.set_xlabel("VRAM budget on the decoding card (GiB)  —  tighter to the right")
    ax.invert_xaxis()
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both")


def main():
    same, split = load()
    if not split:
        raise SystemExit("no output/summary_disagg.csv yet -- run scripts/decode/disagg_sweep.py")
    FIGS.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0][0]
    draw(ax, same, "ttft_p95_ms", BLUE, "same card (prefill+decode)")
    draw(ax, split, "ttft_p95_ms", RED, "split (GPU0 prefill → GPU1 decode)", marker="s")
    style(ax, "TTFT p95 (ms)", "Tail latency", log=True)
    ax.legend(fontsize=8)

    ax = axes[0][1]
    draw(ax, same, "output_tok_per_s", BLUE, "same card")
    draw(ax, split, "output_tok_per_s", RED, "split", marker="s")
    # 2 QPS x 128 tokens: the offered rate. An open loop cannot exceed it, so a curve
    # sitting on this line is healthy and one below it is losing requests.
    ax.axhline(2.0 * 128, color=GREY, ls="--", lw=1,
               label="offered load (2 QPS × 128 tok)")
    style(ax, "output tokens/s", "Throughput against the offered rate")
    ax.legend(fontsize=8)

    ax = axes[1][0]
    draw(ax, same, "itl_p50_ms", BLUE, "same card")
    draw(ax, split, "itl_p50_ms", RED, "split", marker="s")
    style(ax, "ITL p50 (ms)", "Per-token latency once decoding starts")
    ax.legend(fontsize=8)

    # What the budget actually has to hold. On one card the KV tier competes with a 24 GiB
    # working set of reusable prefixes; a decode-only card holds only what it is decoding.
    ax = axes[1][1]
    for rows, colour, label, marker in ((same, BLUE, "same card", "o"),
                                        (split, RED, "split", "s")):
        bx = [fnum(r, "budget_gib") for r in rows if fnum(r, "gpu_kv_gib")]
        by = [fnum(r, "gpu_kv_gib") for r in rows if fnum(r, "gpu_kv_gib")]
        if bx:
            ax.plot(bx, by, marker=marker, color=colour, label=f"{label}: KV tier", lw=1.8, ms=5)
    ax.axhline(24.0, color=PURPLE, ls="--", lw=1.2, label="working set (32 × 6144 tok = 24 GiB)")
    style(ax, "GPU KV tier (GiB)", "KV capacity on the decoding card")
    ax.legend(fontsize=8)

    fig.suptitle("Uniform access, identical Case A workload: one card vs two", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGS / "disagg_vs_same_card.png", dpi=150)
    print(f"wrote {FIGS / 'disagg_vs_same_card.png'}")

    # Second figure: the split setup's own validity check. If GPU0 saturates, the curve is
    # measuring the prefill node, not GPU1's VRAM.
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes2[0]
    draw(ax, split, "prefill_prompt_tok_s", PURPLE, "prefill node prompt tok/s", marker="s")
    style(ax, "prompt tokens/s on GPU0", "Validity check: is the prefill node the bottleneck?")
    ax.legend(fontsize=8)

    ax = axes2[1]
    draw(ax, same, "preemptions", BLUE, "same card")
    draw(ax, split, "preemptions", RED, "split", marker="s")
    style(ax, "preemptions", "Scheduler preemptions")
    ax.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(FIGS / "disagg_validity.png", dpi=150)
    print(f"wrote {FIGS / 'disagg_validity.png'}")


if __name__ == "__main__":
    main()
