#!/usr/bin/env python3
"""QPS sweep: the tenant's real cost is capacity headroom the 2-QPS point hides.

At 2 QPS the open loop pins throughput at the offered rate, so the incumbent looks ~7%
down. Sweeping the 8B's QPS reveals the capacity ceiling: 8B alone saturates near
~325 tok/s, but with the decode-only tenant on GPU1 (no MPS) it saturates near ~245 —
the tenant consumes ~25% of the 8B's serving capacity. Run from the repo root.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
RAW, FIGS = REPO / "output" / "raw", REPO / "figures"
GRY, GRN, PUR = "#57606a", "#2e7d4f", "#8250df"

ALONE = {2: "split_lmcache_zipf_b26_ctrl", 3: "split_lmcache_zipf_b26_q3",
         4: "split_lmcache_zipf_b26_q4"}
TENANT = {2: "infc_B_fits", 3: "infc_B_fits_q3", 4: "infc_B_fits_q4"}


def s(f):
    d = json.load(open(RAW / f"{f}.json"))["summary"]
    return d["output_tok_per_s"], d["ttft_ms"]["p50"]


def main():
    qs = [2, 3, 4]
    offered = [q * 128 for q in qs]
    a_tput, a_ttft = zip(*(s(ALONE[q]) for q in qs))
    t_tput, t_ttft = zip(*(s(TENANT[q]) for q in qs))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # left: achieved vs offered, with the y=x ideal
    ax.plot([200, 540], [200, 540], color=GRY, ls=":", lw=1.2, label="offered = achieved")
    ax.plot(offered, a_tput, "-o", color=GRN, lw=2, ms=8, label="8B alone")
    ax.plot(offered, t_tput, "-s", color=PUR, lw=2, ms=8, label="8B + 0.5B tenant (no MPS)")
    ax.axhline(max(a_tput), color=GRN, ls="--", lw=0.9, alpha=0.6)
    ax.axhline(max(t_tput), color=PUR, ls="--", lw=0.9, alpha=0.6)
    ax.annotate(f"8B-alone capacity ≈ {max(a_tput):.0f} tok/s", (384, max(a_tput)),
                textcoords="offset points", xytext=(0, 6), fontsize=8.5, color=GRN, ha="center")
    ax.annotate(f"with tenant ≈ {max(t_tput):.0f} tok/s  (−{(1 - max(t_tput) / max(a_tput)) * 100:.0f}%)",
                (384, max(t_tput)), textcoords="offset points", xytext=(0, -14),
                fontsize=8.5, color=PUR, ha="center")
    for xo, y in zip(offered, a_tput):
        ax.annotate(f"{y:.0f}", (xo, y), textcoords="offset points", xytext=(5, 7),
                    fontsize=8, color=GRN)
    for xo, y in zip(offered, t_tput):
        ax.annotate(f"{y:.0f}", (xo, y), textcoords="offset points", xytext=(5, 6),
                    fontsize=8, color=PUR)
    ax.set_xticks(offered, [f"{q} QPS\n({o})" for q, o in zip(qs, offered)], fontsize=9)
    ax.set_xlabel("offered load, tok/s", fontsize=10)
    ax.set_ylabel("achieved 8B throughput, tok/s", fontsize=10)
    ax.set_ylim(190, 555)
    ax.set_title("The tenant lowers the 8B's capacity ceiling ~25%", fontsize=11, color=GRY)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25)

    # right: TTFT vs QPS
    ax2.plot(qs, a_ttft, "-o", color=GRN, lw=2, ms=8, label="8B alone")
    ax2.plot(qs, t_ttft, "-s", color=PUR, lw=2, ms=8, label="8B + tenant")
    for xq, y in zip(qs, a_ttft):
        ax2.annotate(f"{y:.0f}", (xq, y), textcoords="offset points", xytext=(4, 6),
                     fontsize=8, color=GRN)
    for xq, y in zip(qs, t_ttft):
        ax2.annotate(f"{y:.0f}", (xq, y), textcoords="offset points", xytext=(4, -12),
                     fontsize=8, color=PUR)
    ax2.set_xticks(qs, [f"{q} QPS" for q in qs], fontsize=9)
    ax2.set_xlabel("8B offered QPS (tenant fixed at 2 QPS)", fontsize=10)
    ax2.set_ylabel("8B TTFT p50, ms", fontsize=10)
    ax2.set_title("TTFT climbs faster with the tenant present", fontsize=11, color=GRY)
    ax2.legend(fontsize=8.5, loc="upper left")
    ax2.grid(alpha=0.25)

    fig.suptitle("QPS sweep — pricing the tenant's cost as lost 8B capacity (b26, B/fits)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "inf_coloc_qps.png", dpi=140, bbox_inches="tight", facecolor="white")
    print("wrote figures/inf_coloc_qps.png")


if __name__ == "__main__":
    main()
