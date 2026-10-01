#!/usr/bin/env python3
"""The §6 results figure, re-drawn with MPS on — and overlaid on the no-MPS points.

Same layout as plot_inf_inf_coloc.py (left: 8B incumbent cost as TTFT/TPOT ratios to the
8B-alone baseline, log y, with absolute throughput on the twin axis; right: what the Qwen
tenant keeps) so the two figures are read side by side. Here the solid markers are the
+MPS runs and the faded 'x' markers are the no-MPS runs from the same cells, so one figure
is both "results under MPS" and "with vs without MPS". Cells with an MPS run: A/fits,
B/fits, B/offload. Run from the repo root.
"""

import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
RAW, FIGS = REPO / "output" / "raw", REPO / "figures"
GRY, BLU, ORG, GRN, AMB = "#57606a", "#1f6feb", "#c1440e", "#2e7d4f", "#9a6700"

# no-MPS cells (fits are means of 4 repeats), and their +MPS counterparts
NOMPS = {
    "A_fits": ["infc_A_fits", "infc_A_fits_r2", "infc_A_fits_r3", "infc_A_fits_r4"],
    "B_fits": ["infc_B_fits", "infc_B_fits_iso", "infc_B_fits_legs", "infc_B_fits_cpu"],
    "B_offload": ["infc_B_offload"],
}
MPS = {"A_fits": "infc_A_fits_mps", "B_fits": "infc_B_fits_mps",
       "B_offload": "infc_B_offload_mps"}
ORDER = ["A_fits", "B_fits", "B_offload"]
LABELS = ["A\nfits", "B\nfits", "B\noffload"]


def g(name, metric):
    s = json.load(open(RAW / f"{name}.json"))["summary"]
    return (s["ttft_ms"]["p50"] if metric == "ttft"
            else s["tpot_ms"]["p50"] if metric == "tpot"
            else s["output_tok_per_s"])


def qtp(name):
    d = json.load(open(RAW / f"{name}.json"))
    return d.get("qwen", {}).get("summary", {}).get("output_tok_per_s")


def main():
    base = json.load(open(RAW / "split_lmcache_zipf_b26_ctrl.json"))["summary"]
    b_ttft, b_tpot, b_tput = base["ttft_ms"]["p50"], base["tpot_ms"]["p50"], base["output_tok_per_s"]

    no = {m: [st.mean([g(n, m) for n in NOMPS[k]]) for k in ORDER] for m in ("ttft", "tpot", "tput")}
    mp = {m: [g(MPS[k], m) for k in ORDER] for m in ("ttft", "tpot", "tput")}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.7),
                                  gridspec_kw={"width_ratios": [1.15, 1]})
    xp = np.arange(len(ORDER) + 1)
    xticks = ["8B alone\n(baseline)"] + LABELS

    # ---- left: incumbent cost ratios (log), MPS solid vs no-MPS faded ----
    ttft_mp = [1.0] + [v / b_ttft for v in mp["ttft"]]
    tpot_mp = [1.0] + [v / b_tpot for v in mp["tpot"]]
    ttft_no = [1.0] + [v / b_ttft for v in no["ttft"]]
    tpot_no = [1.0] + [v / b_tpot for v in no["tpot"]]

    # no-MPS: faded x markers, no connecting line (the "before")
    ax.plot(xp, ttft_no, "x", color=ORG, ms=9, mew=2, alpha=0.4, zorder=2)
    ax.plot(xp, tpot_no, "x", color=BLU, ms=9, mew=2, alpha=0.4, zorder=2)
    # MPS: solid lines (the "after")
    ax.plot(xp, ttft_mp, "-^", color=ORG, lw=1.9, ms=8, zorder=3,
            label="TTFT p50 / baseline  (GPU0 prefill leg)")
    ax.plot(xp, tpot_mp, "-o", color=BLU, lw=1.9, ms=7, zorder=3,
            label="TPOT p50 / baseline  (GPU1 decode leg)")
    # drop arrows from no-MPS to +MPS on the offload cell (the big win)
    for i, (yn, ym) in enumerate(zip(tpot_no, tpot_mp)):
        if xp[i] and yn / ym > 1.6:
            ax.annotate("", (xp[i], ym * 1.05), (xp[i], yn * 0.95),
                        arrowprops=dict(arrowstyle="-|>", color=GRY, lw=1.1, alpha=0.6))
    ax.axhline(1.0, color=GRY, lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 120)
    ax.set_xticks(xp, xticks, fontsize=9)
    ax.set_ylabel("latency, × the 8B-alone baseline (log)", fontsize=10)
    ax.set_title("What the 8B incumbent pays — MPS on (× = no-MPS)", fontsize=11, color=GRY)
    ax.set_xlim(-0.4, len(ORDER) + 0.4)
    ax.grid(alpha=0.25, which="both")
    # annotate offload ratios so the collapse is readable
    ax.annotate(f"{ttft_mp[-1]:.0f}×", (xp[-1], ttft_mp[-1]), textcoords="offset points",
                xytext=(6, 2), fontsize=8, color=ORG)
    ax.annotate(f"{ttft_no[-1]:.0f}× no-MPS", (xp[-1], ttft_no[-1]), textcoords="offset points",
                xytext=(-6, 4), fontsize=7.5, color=ORG, ha="right", alpha=0.7)

    ax_t = ax.twinx()
    ax_t.plot(xp, [b_tput] + mp["tput"], "-s", color=GRN, lw=1.6, ms=5.5, alpha=0.9)
    ax_t.plot(xp, [b_tput] + no["tput"], "s", color=GRN, ms=5, alpha=0.35)
    ax_t.set_ylim(0, 280)
    ax_t.set_ylabel("8B throughput, tok/s", fontsize=10, color=GRN)
    ax_t.tick_params(axis="y", labelcolor=GRN)
    ax.plot([], [], "-s", color=GRN, ms=5.5, label="8B throughput, +MPS (right axis)")
    ax.plot([], [], "x", color=GRY, ms=8, mew=2, alpha=0.5, label="no-MPS (same cell)")
    ax.legend(fontsize=8.1, loc="center left", bbox_to_anchor=(0.30, 0.74),
              handlelength=2.2, handletextpad=0.7, borderpad=0.5, framealpha=0.95)

    # ---- right: Qwen tenant keeps, +MPS vs no-MPS ----
    ceil = {"fits": qtp("infc_solo_fits"), "offload": qtp("infc_solo_offload")}
    cvals = [ceil["fits" if "fits" in k else "offload"] for k in ORDER]
    no_q = [st.mean([qtp(n) for n in NOMPS[k]]) for k in ORDER]
    mp_q = [qtp(MPS[k]) for k in ORDER]
    xq = np.arange(len(ORDER))
    ax2.bar(xq - 0.27, cvals, 0.26, color="#e8ddcf", edgecolor=AMB, lw=1.1,
            label="solo (ceiling)")
    ax2.bar(xq, no_q, 0.26, color="#c9b89a", label="beside 8B, no MPS")
    ax2.bar(xq + 0.27, mp_q, 0.26, color=AMB, label="beside 8B, + MPS")
    for i, (c, v) in enumerate(zip(cvals, mp_q)):
        ax2.text(i + 0.27, v + 4, f"{v / c * 100:.0f}%", ha="center", fontsize=8.5, color=AMB)
    ax2.set_xticks(xq, LABELS, fontsize=9)
    ax2.set_ylabel("Qwen decode throughput, tok/s", fontsize=10)
    ax2.set_ylim(0, 310)
    ax2.set_title("What the 0.5B tenant keeps", fontsize=11, color=GRY)
    ax2.legend(fontsize=8.3, loc="lower left")
    ax2.grid(alpha=0.25, axis="y")

    fig.suptitle("Results under MPS, overlaid on no-MPS — Qwen2.5-0.5B + 8B split-decode "
                 "(b26, 2 QPS)", fontsize=11.5, y=1.02)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "inf_coloc_result_mps.png", dpi=140, bbox_inches="tight",
                facecolor="white")
    print("wrote figures/inf_coloc_result_mps.png")


if __name__ == "__main__":
    main()
