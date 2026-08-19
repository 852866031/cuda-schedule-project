#!/usr/bin/env python3
"""Two figures explaining the access distributions the sweep uses.

The whole locality argument rests on how requests are spread over the 32 sessions, so this
draws it from the real generator rather than describing it.

    python scripts/plot_workload.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from workload import build_workload

REPO = Path(__file__).resolve().parent.parent
FIGS = REPO / "figures"

BLUE, PURPLE, RED, GREY = "#1f6feb", "#8250df", "#c1440e", "#57606a"
PREFIX_GIB = 0.75            # KV held by one session's prefix
BUDGETS = [(13.77, "30 GiB budget"), (5.77, "22 GiB"), (1.77, "18 GiB")]


def main():
    FIGS.mkdir(exist_ok=True)
    wls = {s: build_workload(skew=s, num_requests=300, seed=0) for s in ("zipf", "uniform")}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- left: how many requests each session receives ------------------------------
    for skew, color in (("zipf", BLUE), ("uniform", PURPLE)):
        counts = np.bincount([r.session_id for r in wls[skew].requests],
                             minlength=wls[skew].num_sessions)
        counts = np.sort(counts)[::-1]                     # rank-ordered
        ax1.bar(np.arange(1, len(counts) + 1) + (0.2 if skew == "uniform" else -0.2),
                counts, width=0.4, color=color, alpha=0.85,
                label=f"{skew}  (busiest session: {counts[0]} requests)")
    ax1.set_xlabel("session, ranked by popularity (32 sessions, 0.75 GiB of KV each)")
    ax1.set_ylabel("requests received (of 300)")
    ax1.set_title("How 300 requests spread over 32 sessions", fontsize=11)
    ax1.grid(alpha=0.25, axis="y")
    ax1.legend(fontsize=9)

    # --- right: what fraction of requests the GPU tier can serve ---------------------
    for skew, color in (("zipf", BLUE), ("uniform", PURPLE)):
        counts = np.bincount([r.session_id for r in wls[skew].requests],
                             minlength=wls[skew].num_sessions)
        counts = np.sort(counts)[::-1]
        cum = np.cumsum(counts) / counts.sum() * 100
        gib = np.arange(1, len(counts) + 1) * PREFIX_GIB   # KV needed to hold the top-k
        ax2.plot(gib, cum, color=color, marker="o", ms=4, lw=2, label=skew)

    for x, label in BUDGETS:
        ax2.axvline(x, color=GREY, lw=1, ls=":")
        ax2.annotate(label, xy=(x, 8), xytext=(-4, 0), textcoords="offset points",
                     fontsize=8, color=GREY, rotation=90, va="bottom", ha="right")

    ax2.set_xlabel("GPU KV cache (GiB) — enough to hold the top-k most popular sessions")
    ax2.set_ylabel("share of requests that hit GPU cache (%)")
    ax2.set_title("Why skew decides what offloading is worth", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9, loc="lower right")

    fig.suptitle("The two access patterns: 32 sessions, 24 GiB of KV, 300 requests",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = FIGS / "workload_access.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p.relative_to(REPO)}")

    for skew in ("zipf", "uniform"):
        counts = np.sort(np.bincount([r.session_id for r in wls[skew].requests],
                                     minlength=32))[::-1]
        cum = np.cumsum(counts) / counts.sum() * 100
        for gib, _ in BUDGETS:
            k = int(gib / PREFIX_GIB)
            print(f"  {skew:8s} at {gib:5.2f} GiB GPU KV (top {k:2d} sessions): "
                  f"{cum[min(k, 31) - 1]:5.1f}% of requests could hit GPU")


if __name__ == "__main__":
    main()
