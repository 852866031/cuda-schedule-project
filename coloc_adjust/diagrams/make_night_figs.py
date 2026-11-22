#!/usr/bin/env python3
"""Figures of the vLLM-engine runs of the night of Nov 21/22 (reports/night_2026-11-22*.md), from ../results_n21/night_runs.csv
(analyze_night.py). SVG, rendered to ../report/figures/night_*.png.
  python3 diagrams/make_night_figs.py [name ...]
Every bar is a change against the engine alone on the same night (pooled baselines); a dot per run where a setting was repeated."""
import csv, os, statistics as st, sys
import make_c1c2_figs as F
from make_c1c2_figs import T, R, P, add, INK, INK2, MUTED, LINE, LINE2, BLUE, ORANGE, GREEN, RED, GRAY
RES = os.path.join(F.HERE, "..", "results_n21")
def groups():
    g = {}
    for r in csv.DictReader(open(os.path.join(RES, "night_runs.csv"))):
        if r["run"].startswith("adjust_"): continue
        g.setdefault(r["group"], []).append({k: float(v) for k, v in r.items() if k not in ("run", "group") and v not in ("", None)})
    return g
def panel(b, rows, key, x0, w, y0, title, vmax, unit="%", color=BLUE, hl=None):
    """horizontal bars: mean of the group, a dot per run"""
    b.append(T(x0, y0 - 12, title, "d-b"))
    for i, (lab, runs) in enumerate(rows):
        y = y0 + i * 30; v = [r[key] for r in runs if key in r]
        if not v: continue
        m = st.mean(v); c = RED if (hl and lab in hl) else color
        b.append(R(x0, y, max(m, 0) / vmax * w, 18, c))
        for x in v: b.append(f'<circle cx="{x0 + max(x, 0) / vmax * w:.1f}" cy="{y + 9}" r="3" fill="#fff" stroke="{INK}" stroke-width="1.2"/>') if len(v) > 1 else None
        b.append(T(x0 + max(max(v), 0) / vmax * w + 7, y + 14, (f"{m:+.1f} %" if unit == "%" else f"{m:,.0f}"), "d-s"))
def build():
    G = groups()
    # ------------------------------------------------------------------ the methods at about equal tenant work
    order = [("c1_fix", "usual tenant, pause after the step"), ("v_step", "only the output layer changed, pause after the step"), ("v_w12", "only the output layer changed, slices"),
             ("c2a", "compiled as a whole, pause after the step (Nov 16 best)"), ("cvfm_step", "new tenant, pause after the step"), ("cvfm_w12", "new tenant, slices")]
    rows = [(lab, G[k]) for k, lab in order if k in G]
    if rows:
        b = []; x0 = 330; y0 = 44; n = len(rows)
        for i, (lab, runs) in enumerate(rows): b.append(T(x0 - 12, y0 + i * 30 + 14, lab, "d-s", "end"))
        panel(b, rows, "mean", x0, 90, y0, "mean token gap", 10)
        panel(b, rows, "s90", x0 + 150, 90, y0, "single step p90", 10)
        panel(b, rows, "s99", x0 + 300, 90, y0, "single step p99", 30, color=ORANGE)
        panel(b, rows, "tok", x0 + 440, 60, y0, "tenant tok/s", 5000, unit="", color=GREEN)
        add("night_methods", "Decode next to each tenant on the vLLM engine (0.5 and 1 req/s): change against the engine alone", 880, y0 + n * 30 + 8, "".join(b))
    # ------------------------------------------------------------------ SLO-aware budget, with and without slicing
    order = [("c1_slow5", "usual tenant, target 5 %"), ("cvfm_slow5_step", "new tenant, target 5 %, pause after the step"), ("cvfm_slow5_w12", "new tenant, target 5 %, slices"),
             ("c1_slow15", "usual tenant, target 15 %"), ("cvfm_slow15_step", "new tenant, target 15 %, pause after the step"), ("cvfm_slow15_w12", "new tenant, target 15 %, slices")]
    rows = [(lab, G[k]) for k, lab in order if k in G]
    if rows:
        b = []; x0 = 330; y0 = 44; n = len(rows)
        for i, (lab, runs) in enumerate(rows): b.append(T(x0 - 12, y0 + i * 30 + 14, lab, "d-s", "end"))
        panel(b, rows, "mean", x0, 90, y0, "mean token gap", 10)
        panel(b, rows, "s90", x0 + 150, 90, y0, "single step p90", 10)
        panel(b, rows, "s99", x0 + 300, 90, y0, "single step p99", 30, color=ORANGE)
        panel(b, rows, "tok", x0 + 440, 60, y0, "tenant tok/s", 7000, unit="", color=GREEN)
        add("night_slo", "The budget set automatically from decode's measured slowdown (0.5 and 1 req/s): change against the engine alone", 880, y0 + n * 30 + 8, "".join(b))
    # ------------------------------------------------------------------ decode's cost against tenant progress, and against the data the tenant moves
    MB = {"c1_fix": 13.95, "c2a": 8.39, "v_step": 8.54, "v_w12": 8.54, "cvfm_step": 5.05, "cvfm_w12": 5.05, "cvfm_w12_half": 5.05, "cvfm_w12_x2": 5.05}   # MB per token, tenant alone (profile); the new tenant mostly in its 2-batch shape
    kinds = (("c1_fix", GRAY, "usual tenant"), ("v_", RED, "only the output layer changed"), ("c2a", ORANGE, "compiled as a whole (Nov 16 best)"), ("cvfm", BLUE, "new tenant (all three changes)"))
    have = [k for k in MB if k in G]
    if len(have) >= 5:
        b = []; y0, y1, ym = 44, 300, 8
        for pi, (title, xm, xt, fx, xl) in enumerate((("against the tenant's progress", 8000, 2000, lambda k, r: r["tok"], "tenant progress, tokens per second"),
                                                       ("against the data it moves", 60, 10, lambda k, r: r["tok"] * MB[k] / 1e3, "data the tenant moves, GB per second"))):
            x0 = 70 + pi * 420; w = 330
            b.append(T(x0, y0 - 22, "Decode's mean token gap " + title, "d-b"))
            for v in range(0, ym + 1, 2): y = y1 - v / ym * (y1 - y0); b += [P(f"M{x0} {y:.1f}H{x0 + w}", LINE, 1), T(x0 - 8, y + 4, f"+{v} %", "d-m", "end")]
            for v in range(0, xm + 1, xt): x = x0 + v / xm * w; b += [P(f"M{x:.1f} {y1}v5", INK2, 1), T(x, y1 + 19, f"{v:,}", "d-m", "middle")]
            b.append(T(x0 + w / 2, y1 + 40, xl, "d-m", "middle"))
            line = sorted((st.mean(fx(k, r) for r in G[k]), st.mean(r["mean"] for r in G[k])) for k in ("cvfm_w12_half", "cvfm_w12", "cvfm_w12_x2") if k in G)
            if len(line) >= 2: b.append(P("M" + "L".join(f"{x0 + a / xm * w:.1f} {y1 - c / ym * (y1 - y0):.1f}" for a, c in line), BLUE, 1.6))
            for k in have:
                c = next(col for pre, col, _ in kinds if k.startswith(pre))
                for r in G[k]: b.append(f'<circle cx="{x0 + fx(k, r) / xm * w:.1f}" cy="{y1 - r["mean"] / ym * (y1 - y0):.1f}" r="4.5" fill="{c}" stroke="#fff" stroke-width="1"/>')
        ly = y1 + 62; lx = 70
        for (pre, col, lab), lx in zip(kinds, (70, 190, 420, 660)): b += [f'<circle cx="{lx + 5}" cy="{ly - 4}" r="5" fill="{col}"/>', T(lx + 16, ly, lab, "d-s")]
        add("night_tradeoff", "What the tenant's training costs decode on the vLLM engine (0.5 and 1 req/s; one dot per run)", 880, ly + 14, "".join(b))
if __name__ == "__main__":
    build(); F.render(set(sys.argv[1:]) or None)
