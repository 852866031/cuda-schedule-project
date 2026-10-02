#!/usr/bin/env python3
"""Capacity: how much load can each of N co-located models take before it stops keeping up?

Experiment E4 (DECISIONS_MULTI_COLOC.md): decode-only fits, thread cap on, per-model QPS
swept at fixed N. All the other figures are at 2 QPS per model, where every cell serves
its offered load; this one prices sharing as capacity, the sixth study's lesson.

Rows = cohorts. (a) aggregate delivered output tok/s vs per-model QPS, one line per N,
with each N's offered rate dotted. (b) per-model TPOT p50 vs per-model QPS. (c) the same TPOT against TOTAL offered QPS -- at equal total
load, more engines means smaller batches each (the cost of colocation). A point
"keeps up" if the engines' mean scheduler queue wait is < 50 ms and <= 1% of requests
failed. (Delivered/offered is NOT the test: with 300 requests per client, at high QPS the
fixed drain tail of the window alone pulls it to ~92-94% with zero queueing -- see
DECISIONS_MULTI_COLOC.md.)

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
    "small": dict(ns=(1, 4, 8), qps=(2, 3, 4, 6, 8, 12, 16, 24, 32), sfx="_omp4",
                  label="Qwen2.5-0.5B"),
    "medium": dict(ns=(1, 2, 4), qps=(2, 2.5, 3, 4, 6, 8, 12, 16), sfx="",
                   label="Qwen2.5-3B"),
}
COLORS = ["#2e7d4f", "#1f6feb", "#cf222e"]


def name(cohort, n, q, sfx):
    qt = "" if q == 2 else f"_q{q:g}"
    return f"mc_{cohort}_dfits_n{n}_mps{qt}{sfx}"


def point(cohort, n, q, sfx):
    """(agg delivered tok/s, TPOT p50 mean, failed fraction, mean queue wait ms) or None."""
    for nm in (name(cohort, n, q, sfx), name(cohort, n, q, "")):
        p = RAW / f"{nm}.json"
        if p.exists():
            break
    else:
        return None
    r = json.loads(p.read_text())
    if "models" not in r:
        return None
    fr, tp, failed, total, qs, qc, agg = [], [], 0, 0, 0.0, 0.0, 0.0
    for m in r["models"]:
        d = m.get("metrics_delta", {})
        qs += d.get("vllm:request_queue_time_seconds_sum", 0.0)
        qc += d.get("vllm:request_queue_time_seconds_count", 0.0)
        c = m["client"]
        s = c.get("summary") or {}
        recs = [x for x in c.get("records", []) if x.get("t_submit") is not None]
        if not recs or not s.get("duration_s"):
            return None
        span = max(x["t_submit"] for x in recs) - min(x["t_submit"] for x in recs)
        offered = len(recs) * 128 / span
        fr.append(s["output_tok_per_s"] / offered)
        agg += s["output_tok_per_s"]
        if s.get("tpot_ms", {}).get("p50"):
            tp.append(s["tpot_ms"]["p50"])
        failed += s.get("n_failed") or 0
        total += s.get("n_requests") or 0
    return (agg, sum(tp) / len(tp) if tp else None, failed / max(1, total),
            qs / qc * 1e3 if qc else 0.0)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    caps = {}
    for row, (cohort, c) in enumerate(COHORTS.items()):
        ax_f, ax_t, ax_a = axes[row]
        for n, col in zip(c["ns"], COLORS):
            pts = [(q, point(cohort, n, q, c["sfx"])) for q in c["qps"]]
            pts = [(q, p) for q, p in pts if p]
            if not pts:
                continue
            qs = [q for q, _ in pts]
            ax_f.plot(qs, [p[0] for _, p in pts], color=col, marker="o", lw=2,
                      ms=8, label=f"N={n}")
            ax_f.plot(qs, [n * q * 128 for q in qs], color=col, ls=":", lw=1.2)
            ax_t.plot([q for q, p in pts if p[1]], [p[1] for _, p in pts if p[1]],
                      color=col, marker="o", lw=2, ms=8, label=f"N={n}")
            # same data against TOTAL offered QPS: equal total load, different N
            ax_a.plot([q * n for q, p in pts if p[1]], [p[1] for _, p in pts if p[1]],
                      color=col, marker="o", lw=2, ms=8, label=f"N={n}")
            ok = [q for q, p in pts if p[3] < 50 and p[2] <= 0.01]
            caps[(cohort, n)] = (max(ok) if ok else None, max(qs))
            for q, p in pts:
                if p[2] > 0.01 or p[3] >= 50:     # not keeping up: hollow red ring
                    ax_f.plot(q, p[0], "o", ms=16, mfc="none", mec="#cf222e", mew=2)
                print(f"  {cohort} N={n} q={q:g}: agg {p[0]:.0f} tok/s, TPOT {p[1]:.2f} ms, "
                      f"queue {p[3]:.1f} ms, failed {p[2]*100:.1f}%")
        lf, lt, la = "abcdef"[3 * row:3 * row + 3]
        ax_f.set(title=f"({lf}) {c['label']}: aggregate throughput (dotted = offered)",
                 xlabel="offered QPS per model", ylabel="output tok/s, all models",
                 ylim=(0, None))
        ax_t.set(title=f"({lt}) {c['label']}: per-model TPOT p50", xlabel="offered QPS per model",
                 ylabel="ms", ylim=(0, None))
        ax_a.set(title=f"({la}) {c['label']}: TPOT at equal TOTAL load",
                 xlabel="offered QPS, all models", ylabel="ms", ylim=(0, None))
        ax_a.grid(alpha=0.3)
        ax_a.legend(frameon=False, loc="lower right")
        for ax in (ax_f, ax_t):
            ax.set_xticks([q for q in c["qps"] if q == int(q)])
            ax.grid(alpha=0.3)
            ax.legend(frameon=False, loc="lower right" if ax is ax_f else "center right")
    fig.tight_layout()
    out = FIGS / "mc_capacity.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(out)
    for (cohort, n), (cap, top) in caps.items():
        print(f"{cohort} N={n}: keeps up to {cap} QPS/model (swept to {top})")


if __name__ == "__main__":
    main()
