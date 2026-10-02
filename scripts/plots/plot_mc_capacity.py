#!/usr/bin/env python3
"""Capacity: how much load can each of N co-located models take before it stops keeping up?

Experiment E4 (DECISIONS_MULTI_COLOC.md): decode-only fits, thread cap on, per-model QPS
swept at fixed N. All the other figures are at 2 QPS per model, where every cell serves
its offered load; this one prices sharing as capacity, the sixth study's lesson.

Rows = cohorts. (a) delivered / offered throughput per model vs per-model QPS, one line
per N (offered from each client's actual arrival span); dashed = 95%. (b) per-model TPOT
p50 vs per-model QPS. A point "keeps up" if >= 95% delivered and <= 1% failed requests.

2-QPS points come from the main sweep. For the small cohort at N=1/4 there is no capped
2-QPS run, so the stock run is used (the cap is neutral there: within noise at N=1-3,
and stock N=4 dfits is healthy).

    .venv/bin/python scripts/plots/plot_mc_capacity.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
RAW, FIGS = REPO / "output" / "raw", REPO / "figures"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "legend.fontsize": 12})
COHORTS = {
    "small": dict(ns=(1, 4, 8), qps=(2, 3, 4, 6, 8), sfx="_omp4", label="Qwen2.5-0.5B"),
    "medium": dict(ns=(1, 2, 4), qps=(2, 2.5, 3, 4), sfx="", label="Qwen2.5-3B"),
}
COLORS = ["#2e7d4f", "#1f6feb", "#cf222e"]


def name(cohort, n, q, sfx):
    qt = "" if q == 2 else f"_q{q:g}"
    return f"mc_{cohort}_dfits_n{n}_mps{qt}{sfx}"


def point(cohort, n, q, sfx):
    """(delivered fraction, TPOT p50 mean, failed fraction) or None."""
    for nm in (name(cohort, n, q, sfx), name(cohort, n, q, "")):
        p = RAW / f"{nm}.json"
        if p.exists():
            break
    else:
        return None
    r = json.loads(p.read_text())
    if "models" not in r:
        return None
    fr, tp, failed, total = [], [], 0, 0
    for m in r["models"]:
        c = m["client"]
        s = c.get("summary") or {}
        recs = [x for x in c.get("records", []) if x.get("t_submit") is not None]
        if not recs or not s.get("duration_s"):
            return None
        span = max(x["t_submit"] for x in recs) - min(x["t_submit"] for x in recs)
        offered = len(recs) * 128 / span
        fr.append(s["output_tok_per_s"] / offered)
        if s.get("tpot_ms", {}).get("p50"):
            tp.append(s["tpot_ms"]["p50"])
        failed += s.get("n_failed") or 0
        total += s.get("n_requests") or 0
    return (sum(fr) / len(fr), sum(tp) / len(tp) if tp else None,
            failed / max(1, total))


def main():
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    caps = {}
    for row, (cohort, c) in enumerate(COHORTS.items()):
        ax_f, ax_t = axes[row]
        for n, col in zip(c["ns"], COLORS):
            pts = [(q, point(cohort, n, q, c["sfx"])) for q in c["qps"]]
            pts = [(q, p) for q, p in pts if p]
            if not pts:
                continue
            qs = [q for q, _ in pts]
            ax_f.plot(qs, [p[0] * 100 for _, p in pts], color=col, marker="o", lw=2,
                      ms=8, label=f"N={n}")
            ax_t.plot([q for q, p in pts if p[1]], [p[1] for _, p in pts if p[1]],
                      color=col, marker="o", lw=2, ms=8, label=f"N={n}")
            ok = [q for q, p in pts if p[0] >= 0.95 and p[2] <= 0.01]
            caps[(cohort, n)] = (max(ok) if ok else None, max(qs))
            for q, p in pts:
                if p[2] > 0.01:               # failed point: hollow ring
                    ax_f.plot(q, p[0] * 100, "o", ms=16, mfc="none", mec="#cf222e", mew=2)
        ax_f.axhline(95, color="#57606a", ls="--", lw=1.2)
        ax_f.text(max(c["qps"]), 95.5, "95% ", ha="right", va="bottom", color="#57606a",
                  fontsize=11)
        lf, lt = "abcd"[2 * row], "abcd"[2 * row + 1]
        ax_f.set(title=f"({lf}) {c['label']}: delivered / offered",
                 xlabel="offered QPS per model", ylabel="% of offered tokens",
                 ylim=(0, 110))
        ax_t.set(title=f"({lt}) {c['label']}: per-model TPOT p50", xlabel="offered QPS per model",
                 ylabel="ms", ylim=(0, None))
        for ax in (ax_f, ax_t):
            ax.set_xticks(c["qps"])
            ax.grid(alpha=0.3)
            ax.legend(frameon=False, loc="lower left" if ax is ax_f else "upper left")
    fig.tight_layout()
    out = FIGS / "mc_capacity.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(out)
    for (cohort, n), (cap, top) in caps.items():
        print(f"{cohort} N={n}: keeps up to {cap} QPS/model (swept to {top})")


if __name__ == "__main__":
    main()
