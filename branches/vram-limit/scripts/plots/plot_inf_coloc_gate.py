#!/usr/bin/env python3
"""Two ways to share GPU1 under MPS: fair (MPS alone) vs prioritized (MPS + idle-window gate).

The tenant is treated as best-effort: the 8B decode publishes its GPU-busy window and the
Qwen engine is gated (CUPTI, COLOC_ROLE=be, K=8) to the 8B's idle gaps. At 2 QPS both
decodes have slack, so the gate ~= MPS. Under saturation (4 QPS) the gate recovers the 8B's
capacity (298 -> 315 tok/s) and decode latency (TPOT 241 -> 202 ms) toward the 8B-alone
ceiling, while the tenant keeps full throughput (decode leaves GPU1 idle time to fill).

Left: 8B decode TPOT. Right: 8B throughput (tenant throughput annotated). Grouped by 8B
offered load (2 and 4 QPS; tenant fixed at 2 QPS). Run from the repo root.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
RAW, FIGS = REPO / "output" / "raw", REPO / "figures"
GRY, GRN, PUR = "#57606a", "#2e7d4f", "#8250df"

# (load label) -> {arm: raw name}. alone = 8B with no tenant at that QPS.
ARMS = {
    "2 QPS": {"alone": "split_lmcache_zipf_b26_ctrl", "mps": "infc_B_fits_mps",
              "gate": "infc_B_fits_gate"},
    "4 QPS": {"alone": "split_lmcache_zipf_b26_q4", "mps": "infc_B_fits_mpsq4",
              "gate": "infc_B_fits_gateq4"},
}


def g(name, metric):
    d = json.load(open(RAW / f"{name}.json"))
    s = d["summary"]
    if metric == "qwen":
        return d.get("qwen", {}).get("summary", {}).get("output_tok_per_s")
    return (s["tpot_ms"]["p50"] if metric == "tpot" else s["output_tok_per_s"])


def main():
    loads = list(ARMS)
    x = np.arange(len(loads))
    w = 0.26
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # ---- left: 8B decode TPOT ----
    for i, (arm, col, lab) in enumerate([("alone", GRY, "8B alone"),
                                         ("mps", GRN, "+ tenant, MPS (fair)"),
                                         ("gate", PUR, "+ tenant, MPS + gate (priority)")]):
        vals = [g(ARMS[L][arm], "tpot") for L in loads]
        b = ax.bar(x + (i - 1) * w, vals, w, color=col, label=lab, zorder=3)
        for bb, v in zip(b, vals):
            ax.annotate(f"{v:.0f}", (bb.get_x() + bb.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=9, color=col)
    ax.set_xticks(x, loads, fontsize=11)
    ax.set_ylabel("8B decode TPOT p50, ms", fontsize=10.5)
    ax.set_ylim(0, 285)
    ax.set_title("Decode latency — the gate helps under load", fontsize=11.5, color=GRY)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.annotate("at 2 QPS: slack,\nso gate ≈ MPS", (0, 95), ha="center", fontsize=8.5,
                color=GRY)

    # ---- right: 8B throughput, with tenant throughput annotated ----
    for i, (arm, col, lab) in enumerate([("alone", GRY, "8B alone (ceiling)"),
                                         ("mps", GRN, "+ tenant, MPS (fair)"),
                                         ("gate", PUR, "+ tenant, MPS + gate")]):
        vals = [g(ARMS[L][arm], "tput") for L in loads]
        b = ax2.bar(x + (i - 1) * w, vals, w, color=col, label=lab, zorder=3)
        for bb, v in zip(b, vals):
            ax2.annotate(f"{v:.0f}", (bb.get_x() + bb.get_width() / 2, v),
                         textcoords="offset points", xytext=(0, 3), ha="center",
                         fontsize=9, color=col)
        if arm != "alone":                      # tenant throughput sits unharmed
            for xi, L in zip(x, loads):
                qt = g(ARMS[L][arm], "qwen")
                if qt:
                    ax2.annotate(f"Q{qt:.0f}", (xi + (i - 1) * w, 20), ha="center",
                                 fontsize=7.6, color=col, rotation=90, va="bottom")
    ax2.set_xticks(x, loads, fontsize=11)
    ax2.set_ylabel("8B throughput, tok/s", fontsize=10.5)
    ax2.set_ylim(0, 395)
    ax2.set_title("Capacity — gate recovers it, tenant unharmed (Q = Qwen tok/s)",
                  fontsize=11.5, color=GRY)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(axis="y", alpha=0.25, zorder=0)
    ax2.annotate("gate recovers\n298→315 (−9%→−4%)", (1, 352), ha="center", fontsize=8.5,
                 color=PUR, fontweight="bold")

    fig.suptitle("Fair vs prioritized GPU1 sharing (both under MPS) — 8B + Qwen decode-only, "
                 "b26, tenant best-effort", fontsize=12.5, y=1.0)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "inf_coloc_gate.png", dpi=140, bbox_inches="tight", facecolor="white")
    print("wrote figures/inf_coloc_gate.png")


if __name__ == "__main__":
    main()
