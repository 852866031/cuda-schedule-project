#!/usr/bin/env python3
"""Figures of reports/3_c1_c2.md (C1 byte allowance, C2 lighter tenant), drawn as SVG and rendered to ../report/figures/c1c2_*.png.
  python3 diagrams/make_c1c2_figs.py        (needs google-chrome and Pillow; network for the web font)
Bar values are typed in from results_o3/{r1_table,r2_bytes,r3_table,r4_summary,probe2_summary,r6_floor_summary}.md.
The two time-series figures are computed here from the raw traces of the night (results_o3/<scheme>_ft.csv, _controller.csv, _t0.txt,
bytes_per_token.json; the traces are not in git): tokens finished per 10-s window x MB per token of the shape in use."""
import csv, json, os, statistics as st, subprocess, sys, tempfile
from PIL import Image, ImageChops
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "..", "report", "figures"); RES = os.path.join(HERE, "..", "results_o3")
INK, INK2, MUTED, LINE, LINE2, TINT = "#18212f", "#465166", "#77818f", "#dde2ea", "#b9c1cd", "#eaf1fb"
BLUE, ORANGE, GREEN, RED, GOOD, WARN, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#cf3a3a", "#178a58", "#9a6a00", "#77818f"
CSS = f'''svg{{background:#fff}} text{{font-family:"IBM Plex Sans","Segoe UI",Helvetica,Arial,sans-serif}}
.d-box{{fill:#ffffff;stroke:{LINE2};stroke-width:1.4}}.d-soft{{fill:{TINT};stroke:{BLUE};stroke-width:1.2}}.d-good{{fill:#e7f5ee;stroke:{GOOD};stroke-width:1.2}}
.d-warn{{fill:#fbf3dc;stroke:{WARN};stroke-width:1.2}}.d-ten{{fill:#fdeee6;stroke:{ORANGE};stroke-width:1.2}}.d-bad{{fill:#fbecec;stroke:{RED};stroke-width:1.4}}
.d-zone{{fill:none;stroke:{LINE2};stroke-width:1.2;stroke-dasharray:5 4}}
.d-t{{fill:{INK};font-size:13px}}.d-s{{fill:{INK};font-size:12px}}.d-b{{fill:{INK};font-size:14px;font-weight:600}}.d-h{{fill:{INK};font-size:15.5px;font-weight:600}}
.d-m{{fill:{MUTED};font-size:12px}}.d-g{{fill:{GOOD};font-size:12.5px;font-weight:600}}.d-r{{fill:{RED};font-size:12.5px;font-weight:600}}.d-ln{{stroke:{INK2};stroke-width:1.5;fill:none}}'''
ARR = f'<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L8 4L0 8z" fill="{INK2}"/></marker></defs>'


def T(x, y, s, cls="d-t", anchor=None, style=None):
    a = f' text-anchor="{anchor}"' if anchor else ""; stl = f' style="{style}"' if style else ""
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}"{a}{stl}>{s}</text>'


def R(x, y, w, h, fill, op=None, rx=3):
    o = f' opacity="{op}"' if op else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0.5):.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{o}/>'


def P(d, stroke=LINE2, sw=1, dash=None, marker=False):
    return f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="none"' + (f' stroke-dasharray="{dash}"' if dash else "") + (' marker-end="url(#arr)"' if marker else "") + "/>"


p1 = lambda v: f"{v:+.1f} %"
p0 = lambda v: f"{v:+.0f} %"
FIGS = {}


def add(name, title, w, h, body):
    FIGS[name] = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h + 40}" width="{w}" height="{h + 40}"><style>{CSS}</style>'
                  f'<rect width="{w}" height="{h + 40}" fill="#fff"/>{ARR}{T(10, 24, title, "d-h")}<g transform="translate(0,40)">{body}</g></svg>\n')


# ------------------------------------------------------------------ measured series of the real-engine runs
def load_series(W=10.0):
    bpt = {int(k): v for k, v in json.load(open(os.path.join(RES, "bytes_per_token.json")))["base"].items()}
    out = {}
    for sch in ("tpot_slo15", "c1_fix", "c1_slow15", "c1_slow5"):
        t0 = float(open(os.path.join(RES, f"{sch}_t0.txt")).read().split()[0]); nb = int(660 / W); gb = [0.0] * nb
        for r in csv.DictReader(open(os.path.join(RES, f"{sch}_ft.csv"))):
            if not r["dt_ms"]: continue
            te = float(r["ts"]) - t0; dt = float(r["dt_ms"]) / 1e3; a, b = max(te - dt, 0), min(te, 660)   # the step computes during [te - dt, te]
            k = int(a // W)
            while b > a and k < nb and k * W < b:
                ov = min(b, (k + 1) * W) - max(a, k * W)
                if ov > 0: gb[k] += 4096 * ov / dt * bpt.get(int(r["cfg"]), 13.47) / 1e3
                k += 1
        loop = [[] for _ in range(nb)]
        for r in csv.DictReader(open(os.path.join(RES, f"{sch}_controller.csv"))):
            t = float(r["t"]) - t0
            if 0 <= t < 660 and r.get("gbps_loop"): loop[int(t // W)].append(float(r["gbps_loop"]))
        out[sch] = dict(t=[(i + 0.5) * W for i in range(nb)], gbps=[x / W for x in gb], loop=[st.median(v) if v else None for v in loop])
    return out


PH = [0, 90, 180, 270, 360, 420, 480, 510, 540, 570, 600, 660]; RATE = ["1", "2", "3", "1", "4", "0.5", "2", "3", "2", "1", "1"]


def bands(x0, w, y0, y1, labels=True):
    o = [T(x0 - 8, y0 - 6, "arrivals, req/s", "d-m", "end")] if labels else []
    for i in range(len(RATE)):
        xa, xb = x0 + PH[i] / 660 * w, x0 + PH[i + 1] / 660 * w
        if i % 2 == 0: o.append(f'<rect x="{xa:.1f}" y="{y0}" width="{xb - xa:.1f}" height="{y1 - y0}" fill="{TINT}" opacity=".6"/>')
        if labels: o.append(T((xa + xb) / 2, y0 - 6, RATE[i], "d-m", "middle"))
    return o


def xaxis(x0, w, y):
    o = [P(f"M{x0} {y}H{x0 + w}", INK2)]
    for m in range(12):
        x = x0 + m * 60 / 660 * w; o.append(P(f"M{x:.1f} {y}v5", INK2)); o.append(T(x, y + 19, f"{m} min" if m else "0", "d-m", "middle"))
    return o


def series_panel(b, x0, w, y0, y1, lines, top=160):
    Y = lambda v: y1 - min(v, top) / top * (y1 - y0)
    for v in (0, 50, 100, 150): b.append(P(f"M{x0} {Y(v):.1f}H{x0 + w}", LINE)); b.append(T(x0 - 8, Y(v) + 4, str(v), "d-m", "end"))
    for ts, vs, col, sw, dash in lines:
        pts = " ".join(f"{x0 + t / 660 * w:.1f},{Y(v):.1f}" for t, v in zip(ts, vs) if v is not None)
        b.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linejoin="round"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
    return Y


def build():
    S = load_series()

    # ---- route
    b = []
    def box(x, y, w, h, cls, head, lines):
        o = [f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>', T(x + 12, y + 22, head, "d-b")]
        o += [T(x + 12, y + 42 + 18 * k, ln, "d-m") for k, ln in enumerate(lines)]
        return "".join(o)
    b += [box(10, 10, 210, 84, "d-soft", "Starting point", ["decode pays for the tenant's bytes;", "an SM limit controls them weakly"]),
          box(260, 10, 240, 84, "d-warn", "C1: a byte allowance", ["a GB/s allowance for the tenant;", "it runs, then pauses"]),
          box(540, 10, 240, 84, "d-good", "C1, conclusion", ["decode's cost halves (about 6 %),", "the tail disappears"]),
          P("M220 52H258", INK2, 1.5, marker=True), P("M500 52H538", INK2, 1.5, marker=True),
          box(260, 108, 240, 46, "d-box", "Pause more finely?", []), T(272, 146, "nothing more on the real engine", "d-m"),
          box(540, 108, 240, 46, "d-box", "Who sets the allowance?", []), T(552, 146, "a slow loop: holds, but too careful", "d-m"),
          P("M660 156V176H115V190", INK2, 1.5, marker=True), T(388, 171, "with the limit in place decode is still about 6 % slower", "d-m", "middle"),
          box(10, 192, 210, 84, "d-soft", "What is left to lower", ["the allowance cannot go to zero;", "each token has to cost less"]),
          box(260, 192, 240, 84, "d-warn", "C2: a lighter tenant", ["the same training, fewer bytes;", "compiled: 41 % fewer per token"]),
          box(540, 192, 240, 84, "d-good", "C2, conclusion", ["at the same tenant work, decode", "is down to about 4 %; needs C1"]),
          P("M220 234H258", INK2, 1.5, marker=True), P("M500 234H538", INK2, 1.5, marker=True),
          box(260, 290, 240, 46, "d-box", "Shapes with offload", []), T(272, 328, "cost more than their bytes", "d-m"),
          box(540, 290, 240, 46, "d-box", "The floor", []), T(552, 328, "3–5 % even with no bytes moved", "d-m")]
    add("c1c2_route", "The route of this report", 790, 350, "".join(b).replace('height="46" rx="8"/>', 'height="50" rx="8"/>'))

    # ---- the two benches
    b = []
    for k, (zone, dec, d1, d2, m1, m2) in enumerate([("Synthetic bench · GPU 1", "Synthetic decode step", "replays one Llama-3-8B step", "138 SMs · batch 1 / 4 / 8", "Measured: the time of every decode step", "20 s per point, run forwards and backwards"),
                                                      ("Real-engine bench · GPU 0", "vLLM + kvcached", "Llama-3-8B, real requests", "11 minutes · 0.5–4 req/s", "Measured: each request's TPOT, token gaps", "one run per scheme, against the engine alone")]):
        x = 10 + k * 392
        b += [f'<rect class="d-zone" x="{x}" y="6" width="376" height="248" rx="12"/>', T(x + 14, 30, zone, "d-b"),
              f'<rect class="d-soft" x="{x + 14}" y="44" width="208" height="72" rx="8"/>', T(x + 118, 68, dec, "d-b", "middle"), T(x + 118, 88, d1, "d-m", "middle"), T(x + 118, 105, d2, "d-m", "middle"),
              f'<rect class="d-ten" x="{x + 232}" y="44" width="130" height="72" rx="8"/>', T(x + 297, 68, "Tenant" if k == 0 else "The same tenant", "d-b", "middle"), T(x + 297, 88, "LoRA fine-tune", "d-m", "middle"), T(x + 297, 105, "32 SMs", "d-m", "middle"),
              P(f"M{x + 118} 116V136", INK2, 1.5, marker=True), P(f"M{x + 297} 116V136", INK2, 1.5, marker=True),
              f'<rect class="d-bad" x="{x + 14}" y="138" width="348" height="38" rx="8"/>', T(x + 188, 162, "path between the SMs and GPU memory · shared", "d-r", "middle"),
              f'<rect class="d-box" x="{x + 14}" y="194" width="348" height="50" rx="8"/>', T(x + 188, 215, m1, "d-t", "middle"), T(x + 188, 234, m2, "d-m", "middle")]
    add("c1c2_benches", "Two test benches: the same tenant, a different decode side", 790, 262, "".join(b))

    # ---- Nov 9: tenant work against decode slowdown for each SM count (disjoint SMs, next to 4 requests; results/knee_bw2_summary.md)
    pts = [(8, 2750, 7.5), (16, 5122, 9.1), (32, 9198, 15.3), (64, 14949, 19.5), (96, 19213, 29.1)]
    x0, y0, w, h = 70, 250, 470, 220
    X = lambda v: x0 + v / 20000 * w; Y = lambda v: y0 - v / 30 * h
    b = []
    for v in (0, 10, 20, 30): b += [P(f"M{x0} {Y(v):.1f}H{x0 + w}", LINE), T(x0 - 8, Y(v) + 4, f"{v} %", "d-m", "end")]
    for v in (0, 5000, 10000, 15000, 20000): b += [P(f"M{X(v):.1f} {y0}v5", INK2), T(X(v), y0 + 19, f"{v // 1000}k" if v else "0", "d-m", "middle")]
    b += [P(f"M{x0} {y0}H{x0 + w}", INK2), T(x0 + w / 2, y0 + 38, "tenant work, tok/s", "d-m", "middle"), T(14, 12, "decode slowdown", "d-m"),
          P(f"M{x0} {y0}L{X(19213):.1f} {Y(29.1):.1f}", LINE2, 1.2, "4 4"),
          f'<polyline points="{" ".join(f"{X(t):.1f},{Y(sl):.1f}" for _, t, sl in pts)}" fill="none" stroke="{ORANGE}" stroke-width="2"/>']
    for sm, t, sl in pts:
        b += [f'<circle cx="{X(t):.1f}" cy="{Y(sl):.1f}" r="5.5" fill="{ORANGE}" stroke="#fff" stroke-width="2"/>', T(X(t) + (-34 if sm == 64 else 4), Y(sl) - 12, f"{sm} SMs", "d-b", "middle", "font-size:12.5px")]
    lx = 566
    b += [T(lx, 40, "From 96 SMs down to 8:", "d-b", style="font-size:13px"), T(lx, 62, "tenant work falls to 1/7", "d-t"), T(lx, 82, "(19.2 k → 2.8 k tok/s),", "d-m"),
          T(lx, 106, "decode slowdown only to 1/4", "d-t"), T(lx, 126, "(+29.1 % → +7.5 %).", "d-m"),
          P(f"M{lx} 158h26", LINE2, 1.2, "4 4"), T(lx + 32, 162, "slowdown in proportion", "d-m"), T(lx + 32, 178, "to tenant work", "d-m"),
          T(lx, 210, "The 8- and 16-SM points lie", "d-m"), T(lx, 228, "above that line: little training", "d-m"), T(lx, 246, "for the slowdown they cause.", "d-m")]
    add("c1c2_sm_cost", "Fewer SMs for the tenant: its work falls faster than decode's slowdown", 790, 296, "".join(b))

    # ---- C1, synthetic: three ways to limit
    meth = [("8 SMs, running all the time", [3.9, 6.9, 5.9], [5.4, 7.9, 8.2], GRAY), ("32 SMs, pause after each step (C1)", [2.5, 5.7, 5.5], [5.8, 11.8, 11.0], BLUE),
            ("all SMs shared, pause after each step", [4.6, 7.0, 6.5], [12.7, 20.0, 19.0], GRAY)]
    b = [R(250, 4, 14, 10, INK2), T(270, 14, "mean", "d-m"), R(320, 4, 14, 10, INK2, 0.4), T(340, 14, "p90 of single steps", "d-m")]; sc = 5.4
    for k, pn in enumerate(["next to 1 request", "4 requests", "8 requests"]):
        x0 = 250 + k * 180; b += [T(x0, 44, pn, "d-b", style="font-size:13px"), P(f"M{x0} 52V182")]
        for m, (name, mean, p90, col) in enumerate(meth):
            y = 62 + m * 40
            if k == 0: b.append(T(10, y + 16, name, "d-s"))
            b += [R(x0, y, mean[k] * sc, 11, col), T(x0 + mean[k] * sc + 5, y + 10, p1(mean[k]), "d-s"), R(x0, y + 13, p90[k] * sc, 11, col, 0.4), T(x0 + p90[k] * sc + 5, y + 23, p1(p90[k]), "d-m")]
    b.append(T(10, 200, "Tenant work is the same in all three rows: 2.4 / 2.8 / 2.9 k tok/s next to 1 / 4 / 8 requests.", "d-m"))
    add("c1c2_c1_synthetic", "The same tenant work limited three ways: decode slowdown per step (synthetic bench)", 790, 208, "".join(b))

    # ---- C1, synthetic: running against pausing
    run, pau = [8.1, 14.0, 13.5], [0.7, 3.0, 2.7]
    b = [R(170, 4, 14, 10, RED), T(190, 14, "while the tenant runs", "d-m"), R(340, 4, 14, 10, RED, 0.4), T(360, 14, "while the tenant pauses", "d-m"),
         P("M530 2v14", INK, 1.6, "4 3"), T(538, 14, "assumed in the proposal: 0 while it pauses", "d-m")]; sc = 30
    for k, pn in enumerate(["next to 1 request", "4 requests", "8 requests"]):
        y = 36 + k * 42
        b += [T(10, y + 17, pn, "d-t"), R(170, y, run[k] * sc, 12, RED), T(170 + run[k] * sc + 6, y + 11, p1(run[k]), "d-s"), R(170, y + 14, pau[k] * sc, 12, RED, 0.4), T(170 + pau[k] * sc + 6, y + 25, p1(pau[k]), "d-s")]
    b.append(P("M170 30V158", INK, 1.6, "4 3"))
    add("c1c2_c1_pause", "Decode slowdown while the tenant runs and while it pauses (synthetic bench, tenant on 32 SMs)", 790, 164, "".join(b))

    # ---- C1, real engine: what the tenant moved
    x0, w, y0, y1 = 96, 664, 54, 224
    b = bands(x0, w, y0, y1)
    Y = series_panel(b, x0, w, y0, y1, [(S["tpot_slo15"]["t"], S["tpot_slo15"]["gbps"], GRAY, 2, None), (S["c1_fix"]["t"], S["c1_fix"]["gbps"], BLUE, 2.6, None)])
    b += [T(x0 - 8, y0 - 24, "GB/s", "d-m", "end"), P(f"M{x0} {Y(50):.1f}H{x0 + w}", INK, 1.4, "6 4"),
          P("M96 10h26", GRAY, 2), T(128, 14, "tpot_slo15 (control): moved by the tenant", "d-m"), P("M400 10h26", BLUE, 2.6), T(432, 14, "C1: moved by the tenant", "d-m"),
          P("M600 10h26", INK, 1.4, "6 4"), T(632, 14, "C1's allowance, 50 GB/s", "d-m")] + xaxis(x0, w, y1)
    add("c1c2_c1_tenant_rate", "What the tenant moved, 10-s averages (real engine, 11-minute schedule)", 790, 250, "".join(b))

    # ---- real-engine bar pairs
    def real(name, title, A, B):
        b = [R(190, 4, 14, 10, A[1]), T(210, 14, A[0], "d-m"), R(490, 4, 14, 10, B[1]), T(510, 14, B[0], "d-m")]; sc = 8.0
        for k, pn in enumerate(["0.5 req/s", "1 req/s"]):
            x0 = 190 + k * 300; b += [T(x0, 46, pn, "d-b", style="font-size:13px"), T(x0 + 76, 46, f"tenant {A[2][k]:,} and {B[2][k]:,} tok/s", "d-m"), P(f"M{x0} 54V180")]
            for m, mn in enumerate(["mean gap between tokens", "TPOT per request, median", "TPOT per request, p90"]):
                y = 64 + m * 40; fm = p1 if m == 0 else p0; va, vb = A[3 + m][k], B[3 + m][k]
                if k == 0: b.append(T(10, y + 17, mn, "d-s"))
                b += [R(x0, y, va * sc, 12, A[1]), T(x0 + va * sc + 5, y + 11, fm(va), "d-s"), R(x0, y + 14, vb * sc, 12, B[1]), T(x0 + vb * sc + 5, y + 25, fm(vb), "d-s")]
        add(name, title, 790, 188, "".join(b))
    real("c1c2_c1_real", "Real engine: decode against the engine alone, at nearly the same tenant work",
         ("tpot_slo15 (control): all SMs, fast loop", GRAY, (3769, 3953), (12.2, 11.0), (7, 10), (28, 15)), ("C1: 32 SMs, fixed 50 GB/s", BLUE, (3524, 3670), (5.6, 6.3), (6, 6), (6, 4)))
    real("c1c2_c2_real", "Real engine: the same allowance setting, only the tenant differs",
         ("C1 + the usual tenant", BLUE, (3524, 3670), (5.6, 6.3), (6, 6), (6, 4)), ("C1 + the compiled tenant (C2)", GREEN, (3524, 3688), (4.3, 4.0), (5, 4), (4, 2)))

    # ---- slices
    b = [T(10, 14, "Synthetic bench, next to 4 requests: each decode step", "d-b", style="font-size:13px"), R(196, 26, 14, 10, BLUE), T(216, 36, "mean", "d-m"), R(262, 26, 14, 10, BLUE, 0.4), T(282, 36, "p90 of single steps", "d-m")]
    for m, (n, mean, p90) in enumerate([("pause after each step", 5.7, 11.8), ("slices of 48 blocks", 5.8, 11.8), ("slices of 8 blocks", 5.9, 10.3), ("slices of 1 block (2.4 ms)", 5.8, 5.5)]):
        y = 50 + m * 36; b += [T(10, y + 16, n, "d-s"), R(196, y, mean * 12, 11, BLUE), T(196 + mean * 12 + 5, y + 10, p1(mean), "d-s"), R(196, y + 13, p90 * 12, 11, BLUE, 0.4), T(196 + p90 * 12 + 5, y + 23, p1(p90), "d-m")]
    b += [P("M424 6V196", LINE2, 1, "4 4"), T(444, 14, "Real engine: p90 of single token gaps", "d-b", style="font-size:13px"),
          R(444, 26, 14, 10, BLUE), T(464, 36, "pause after each step", "d-m"), R(606, 26, 14, 10, GREEN), T(626, 36, "slices of 1 block", "d-m")]
    for k, (pn, a, c) in enumerate([("0.5 req/s", 8.2, 7.9), ("1 req/s", 14.7, 14.3)]):
        y = 54 + k * 44; b += [T(444, y + 17, pn, "d-s"), R(524, y, a * 11, 12, BLUE), T(524 + a * 11 + 5, y + 11, p1(a), "d-s"), R(524, y + 14, c * 11, 12, GREEN), T(524 + c * 11 + 5, y + 25, p1(c), "d-s")]
    b += [T(444, 158, "tenant work: 3,524 / 3,670 tok/s with pauses after a step,", "d-m"), T(444, 176, "3,195 / 3,348 tok/s with slices (9 % less)", "d-m")]
    add("c1c2_slices", "Slices inside a step: the tail of single steps on the synthetic bench, and on the real engine", 790, 200, "".join(b))

    # ---- slow loop, measured
    x0, w = 96, 664
    b = [P("M96 10h26", BLUE, 2.6), T(128, 14, "moved by the tenant (10-s averages)", "d-m"), P("M400 10h26", GRAY, 1.4), T(432, 14, "allowance given by the loop", "d-m")]
    for k, (key, name) in enumerate([("c1_slow15", "Target: decode at most 15 % slower"), ("c1_slow5", "Target: decode at most 5 % slower")]):
        y0 = 70 + k * 178; y1 = y0 + 124
        b += bands(x0, w, y0, y1, labels=(k == 0))
        series_panel(b, x0, w, y0, y1, [(S[key]["t"], S[key]["loop"], GRAY, 1.4, None), (S[key]["t"], S[key]["gbps"], BLUE, 2.6, None)])
        b += [T(x0, y0 - 24, name, "d-b", style="font-size:13px"), T(x0 - 8, y0 - 24, "GB/s", "d-m", "end")]
        b += xaxis(x0, w, y1) if k == 1 else [P(f"M{x0} {y1}H{x0 + w}", INK2)]
    add("c1c2_slow_loop", "The slow loop: what the tenant moved, and the allowance the loop gave (real engine)", 790, 400, "".join(b))

    # ---- torch.compile on one layer
    ops = [("to float32", "3.67 MB", 1), ("square", "3.67 MB", 1), ("mean", "4 KB", 0), ("add ε", "4 KB", 0), ("rsqrt", "4 KB", 0), ("multiply", "3.67 MB", 1), ("to bf16", "1.84 MB", 1), ("× weight", "1.84 MB", 1)]
    b = [T(10, 16, "Eager: Python issues 8 operators one by one; each writes its result to memory", "d-b")]
    for i, (n, sz, big) in enumerate(ops):
        x = 14 + i * 96; b += [f'<rect class="d-box" x="{x}" y="30" width="86" height="34" rx="6"/>', T(x + 43, 52, n, "d-s", "middle"), P(f"M{x + 43} 64V88", INK2, 2.4 if big else 1, marker=True)]
        if i < 7: b.append(P(f"M{x + 86} 47h10", INK2))
    b.append('<rect class="d-good" x="14" y="90" width="758" height="36" rx="6"/>')
    b += [T(14 + i * 96 + 43, 113, sz, "d-r" if big else "d-m", "middle") for i, (n, sz, big) in enumerate(ops)]
    b += [T(10, 144, "5 new full-size tensors in memory, 14.7 MB; 4 of them are only intermediate results.", "d-m"), T(10, 184, "Compiled: one kernel generated ahead of time", "d-b"),
          '<rect class="d-warn" x="14" y="198" width="758" height="50" rx="6"/>', T(393, 219, "to float32 · square · mean · add ε · rsqrt · multiply · to bf16 · × weight", "d-s", "middle"),
          T(393, 238, "intermediate values stay in registers", "d-m", "middle"), P("M729 248V272", INK2, 2.4, marker=True),
          '<rect class="d-good" x="14" y="274" width="758" height="36" rx="6"/>', T(729, 297, "1.84 MB", "d-r", "middle"), T(30, 297, "1 new full-size tensor in memory: the output (plus 4 KB kept for the backward pass)", "d-m")]
    add("c1c2_compile", "What torch.compile changes: the forward pass of one RMSNorm layer of the tenant (run on the CPU)", 790, 318, "".join(b))

    # ---- bytes per token
    x0, sc, clip = 270, 13.0, 25.0
    rows = [("usual tenant (4 micro-batches)", 13.47, ORANGE, None, "", ""), ("torch.compile (merged operators)", 7.95, GREEN, None, "−41 %", "same training, 24 % faster"),
            ("forward pass only", 3.73, LINE2, None, "−72 %", "nothing trained: a lower bound"), None,
            ("gradient checkpointing alone", 13.08, ORANGE, 0.45, "−3 %", "tenant 24 % slower"), ("2 micro-batches", 13.48, ORANGE, 0.45, "0 %", ""), ("8 micro-batches", 14.88, ORANGE, 0.45, "+10 %", ""),
            ("1 micro-batch", 15.58, ORANGE, 0.45, "+16 %", ""), ("offload: activations in host RAM", 15.19, ORANGE, 0.45, "+13 %", ""), ("offload: + 12 layers of weights", 16.57, ORANGE, 0.45, "+23 %", ""),
            ("offload: + 20 layers of weights", 17.61, ORANGE, 0.45, "+31 %", ""), ("attention not fused", 20.66, ORANGE, 0.45, "+53 %", "fused is the default"),
            ("output layer and loss in chunks", 42.23, RED, None, "+214 %", "a design error")]
    b = [T(10, 14, "How the tenant computes", "d-b"), T(x0, 14, "MB moved per token", "d-b"), P("M10 24H780")]; y = 34; ytop = y
    for r in rows:
        if r is None: b.append(T(10, y + 14, "Other variants tried", "d-m")); y += 24; continue
        n, v, col, op, d, note = r; wv = min(v, clip) * sc; lx = max(x0 + wv, x0 + 13.47 * sc) + 8
        b += [T(10, y + 14, n, "d-s"), R(x0, y, wv, 18, col, op), T(lx, y + 14, f"{v:.2f} MB", "d-s")]
        if v > clip: b.append(f'<path d="M{x0 + wv - 16} {y - 2}l8 22M{x0 + wv - 9} {y - 2}l8 22" stroke="#fff" stroke-width="3" fill="none"/>')
        if d: b.append(T(lx + 62, y + 14, d, "d-g" if d.startswith("−") else "d-m"))
        if note: b.append(T(lx + 112, y + 14, note, "d-m"))
        y += 26
    b.append(P(f"M{x0 + 13.47 * sc:.1f} {ytop - 4}V{y - 4}", INK, 1.2, "4 3"))
    add("c1c2_bytes_per_token", "Bytes per token of each tenant variant, measured on the GPU (tenant alone on 32 SMs)", 790, y + 2, "".join(b))

    # ---- C2 synthetic
    a, c = [8.1, 13.9, 13.7], [5.3, 9.7, 9.2]
    b = [R(170, 4, 14, 10, ORANGE), T(190, 14, "usual tenant", "d-m"), R(290, 4, 14, 10, GREEN), T(310, 14, "compiled tenant, paced to the same tokens/s", "d-m")]
    for k, (pn, tk) in enumerate([("next to 1 request", "8.0 k tok/s"), ("4 requests", "9.2 k tok/s"), ("8 requests", "9.4 k tok/s")]):
        y = 34 + k * 44
        b += [T(10, y + 12, pn, "d-t"), T(10, y + 28, "tenant " + tk, "d-m"), R(170, y, a[k] * 32, 13, ORANGE), T(170 + a[k] * 32 + 6, y + 11, p1(a[k]), "d-s"),
              R(170, y + 15, c[k] * 32, 13, GREEN), T(170 + c[k] * 32 + 6, y + 26, p1(c[k]), "d-s"), T(170 + c[k] * 32 + 62, y + 26, f"{round((1 - c[k] / a[k]) * 100)} % less", "d-g")]
    add("c1c2_c2_synthetic", "The same tenant work with fewer bytes per token: decode slowdown per step (synthetic bench)", 790, 166, "".join(b))

    # ---- shapes
    rows = [("4 micro-batches", 3701, 80, 6.0), ("8 micro-batches", 3321, 66, 7.5), ("8 micro-batches + activations in host RAM", 2086, 32, 10.0), ("+ 12 layers of weights in host RAM", 1818, 30, 14.0), ("+ 20 layers of weights in host RAM", 1670, 29, 13.0)]
    b = [T(10, 14, "Tenant shape", "d-b"), T(290, 14, "Tenant work", "d-b"), T(290, 31, "tok/s", "d-m"), T(450, 14, "Bytes while computing", "d-b"), T(450, 31, "GB/s", "d-m"),
         T(630, 14, "Decode, mean slowdown", "d-b"), T(630, 31, "token gap, against no tenant", "d-m"), P("M10 40H780")]
    for k, (n, tk, gb, sl) in enumerate(rows):
        y = 52 + k * 32
        b += [T(10, y + 14, n, "d-s"), R(290, y, tk * 0.03, 18, GREEN), T(290 + tk * 0.03 + 6, y + 14, f"{tk:,}", "d-s"), R(450, y, gb * 1.4, 18, ORANGE), T(450 + gb * 1.4 + 6, y + 14, str(gb), "d-s"),
              R(630, y, sl * 7, 18, RED), T(630 + sl * 7 + 6, y + 14, p1(sl), "d-s")]
    b.append(P("M10 109H780", LINE2, 1, "4 4"))
    add("c1c2_shapes", "Every tenant shape under the same 50 GB/s allowance (real engine, 1 req/s)", 790, 212, "".join(b))

    # ---- floor
    b = []
    for k, (h, lines, res) in enumerate([("Kernel launches", ["about 100 k launches/s of 16 KB each", "clock unchanged, under the power limit"], "decode +3–5 %"),
                                         ("The power limit", ["arithmetic on 32 SMs", "card at 500 W, clock down 2–7 %"], "decode +4–10 %"),
                                         ("The shared cache", ["64-MB working set next to 8 requests", "counter 6–8 % above decode's own traffic"], "decode +5–7 %")]):
        x = 10 + k * 262; b += [f'<rect class="d-box" x="{x}" y="6" width="250" height="112" rx="10"/>', T(x + 14, 30, h, "d-b"), P(f"M{x + 14} 40H{x + 236}")]
        b += [T(x + 14, 60 + i * 19, ln, "d-m") for i, ln in enumerate(lines)] + [T(x + 14, 106, res, "d-r")]
    add("c1c2_floor", "Why a job that moves no bytes still slows decode: three causes, each seen on its own", 790, 124, "".join(b))

    # ---- trade-off on the real engine
    x0, y0, w, h = 70, 258, 500, 220
    X = lambda v: x0 + v / 12500 * w; Y = lambda v: y0 - v / 15 * h
    b = []
    for v in (0, 5, 10, 15): b += [P(f"M{x0} {Y(v):.1f}H{x0 + w}", LINE), T(x0 - 8, Y(v) + 4, f"{v} %", "d-m", "end")]
    for v in (0, 2500, 5000, 7500, 10000, 12500): b += [P(f"M{X(v):.1f} {y0}v5", INK2), T(X(v), y0 + 19, f"{v / 1000:g}k" if v else "0", "d-m", "middle")]
    b += [P(f"M{x0} {y0}H{x0 + w}", INK2), T(x0 + w / 2, y0 + 38, "tenant work, tok/s", "d-m", "middle"), T(14, 12, "decode, mean slowdown", "d-m")]
    for xe, ye, lab in [(12500, 12.5, "1 %"), (7500, 15, "2 %"), (5000, 15, "3 %")]:
        b += [P(f"M{x0} {y0}L{X(xe):.1f} {Y(ye):.1f}", LINE2, 1, "3 4"), T(X(xe) + 5, Y(ye) + (4 if xe == 12500 else -4), lab, "d-m")]
    for col, p, q in [(GRAY, (3769, 12.2), (3953, 11.0)), (GRAY, (11468, 13.4), (9154, 13.8)), (BLUE, (3524, 5.6), (3670, 6.3)), (BLUE, (3195, 5.7), (3348, 5.9)), (BLUE, (3686, 5.5), (4167, 6.2)),
                      (BLUE, (2622, 3.5), (1442, 2.6)), (GREEN, (3524, 4.3), (3688, 4.0))]:
        b += [P(f"M{X(p[0]):.1f} {Y(p[1]):.1f}L{X(q[0]):.1f} {Y(q[1]):.1f}", col, 1.6), f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="5" fill="{col}" stroke="#fff" stroke-width="2"/>',
              f'<rect x="{X(q[0]) - 4.5:.1f}" y="{Y(q[1]) - 4.5:.1f}" width="9" height="9" fill="{col}" stroke="#fff" stroke-width="2"/>']
    b += [T(X(3953) + 12, Y(11.6) + 4, "tpot_slo15 (control)", "d-s"), T(X(10300), Y(13.6) + 24, "green_dyn2 (control)", "d-s", "middle"), T(X(4167) + 12, Y(6.2) + 1, "C1: fixed, sliced, slow loop +15 %", "d-s"),
          T(X(3688) + 12, Y(4.0) + 8, "C1 + compiled tenant (C2)", "d-g"), T(X(2622) + 4, Y(2.2) + 12, "C1: slow loop +5 %", "d-s")]
    lx = 618
    b += [f'<circle cx="{lx + 6}" cy="52" r="5" fill="{INK2}"/>', T(lx + 18, 56, "0.5 req/s", "d-m"), f'<rect x="{lx + 1.5}" y="69.5" width="9" height="9" fill="{INK2}"/>', T(lx + 18, 78, "1 req/s", "d-m"),
          R(lx, 96, 12, 10, GRAY), T(lx + 18, 105, "controls", "d-m"), R(lx, 116, 12, 10, BLUE), T(lx + 18, 125, "C1", "d-m"), R(lx, 136, 12, 10, GREEN), T(lx + 18, 145, "C1 + C2", "d-m"),
          T(lx, 178, "Better towards the lower", "d-m"), T(lx, 196, "right: more tenant work,", "d-m"), T(lx, 214, "less decode slowdown.", "d-m"),
          T(lx, 244, "Dashed: decode slowdown", "d-m"), T(lx, 262, "per 1 k tenant tok/s.", "d-m")]
    add("c1c2_tradeoff", "All schemes on the real engine: how much the tenant trains, and what decode pays", 790, 304, "".join(b))


def render(only=None):
    os.makedirs(FIG, exist_ok=True)
    for name, s in FIGS.items():
        if only and name not in only: continue
        open(os.path.join(HERE, name + ".svg"), "w", encoding="utf-8").write(s)
        tmp = tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8")
        tmp.write('<!doctype html><meta charset="utf-8"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap">'
                  '<body style="margin:0;background:#fff">' + s); tmp.close()
        out = os.path.join(FIG, name + ".png")
        subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2", "--window-size=900,900",
                        "--virtual-time-budget=8000", f"--screenshot={out}", "file://" + tmp.name], check=True, capture_output=True)
        os.unlink(tmp.name); im = Image.open(out).convert("RGB"); bb = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
        im.crop((0, 0, min(im.size[0], bb[2] + 24), min(im.size[1], bb[3] + 24))).save(out); print(name, Image.open(out).size)


if __name__ == "__main__":
    build(); render(set(sys.argv[1:]) or None)
