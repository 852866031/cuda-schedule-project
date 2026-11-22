#!/usr/bin/env python3
"""Figures of the tenant profile (reports/3_c1_c2.md, section 1), drawn as SVG and rendered to ../report/figures/prof_*.png.
  python3 diagrams/make_profile_figs.py [name ...]      (needs google-chrome and Pillow; network for the web font)
Every value is read from ../results_prof/ (segment_bytes.py: seg_<variant>.csv, _trace.csv, _marks.csv, seg_summary.csv), except the
per-kernel figures of the output layer, typed in from the Nsight Compute run of Nov 21 (results_prof/ncu_head_kernels.md)."""
import csv, os, sys
import make_c1c2_figs as F
from make_c1c2_figs import T, R, P, add, INK, INK2, MUTED, LINE, LINE2, TINT, BLUE, ORANGE, GREEN, RED, GRAY
RES = os.path.join(F.HERE, "..", "results_prof")
PURPLE = "#7a5bd0"
def rows(name): return list(csv.DictReader(open(os.path.join(RES, name))))
def segs(v): return [dict(r, total=float(r["total_gb"]), ms=float(r["ms"]), rd=float(r["read_gb"]), wr=float(r["write_gb"])) for r in rows(f"seg_{v}.csv")]
def summary(): return {r["variant"]: r for r in rows("seg_summary.csv") if r.get("step_gb")}
def trace(v):
    tr = [(float(r["t_ms"]), float(r["read_gbps"]) + float(r["write_gbps"])) for r in rows(f"seg_{v}_trace.csv")]
    mk = [(float(r["t_ms"]), r["label"]) for r in rows(f"seg_{v}_marks.csv")]
    return tr, mk
def kind(label):
    return "head" if label.startswith("head:") else "fwd" if label.endswith(":fwd") else "bwd" if label.endswith(":bwd") else "other"
def first_mb(mk):
    """marks of the first micro-batch: from the step's first checkpoint to its first 'mb_end'"""
    a = next(i for i, m in enumerate(mk) if m[1] == "step"); b = next(i for i, m in enumerate(mk) if m[1] == "mb_end")
    return mk[a:b + 1]
def yaxis(b, x0, x1, y0, y1, vmax, ticks, unit):
    for v in ticks:
        y = y1 - v / vmax * (y1 - y0); b.append(P(f"M{x0} {y:.1f}H{x1}", LINE, 1)); b.append(T(x0 - 6, y + 4, f"{v:,}", "d-m", "end"))
    b.append(T(x0 - 6, y0 - 10, unit, "d-m", "end"))
def area(pts, x0, w, t0, t1, y0, y1, vmax, fill):
    xs = lambda t: x0 + (t - t0) / (t1 - t0) * w; ys = lambda v: y1 - min(v, vmax) / vmax * (y1 - y0)
    use = [(t, v) for t, v in pts if t0 <= t <= t1]
    d = f"M{xs(use[0][0]):.1f} {y1}" + "".join(f"L{xs(t):.1f} {ys(v):.1f}" for t, v in use) + f"L{xs(use[-1][0]):.1f} {y1}Z"
    return f'<path d="{d}" fill="{fill}" opacity=".85"/>'

def build():
    S = summary(); base = segs("base"); split = segs("split")
    # ------------------------------------------------------------------ the trace of one micro-batch, and the output layer in close-up
    tr, mk = trace("base"); mb = first_mb(mk); t0, t1 = mb[0][0], mb[-1][0]
    x0, w, y0, y1, vmax = 70, 760, 40, 230, 1600; b = []
    yaxis(b, x0, x0 + w, y0, y1, vmax, [0, 400, 800, 1200, 1600], "GB/s")
    xs = lambda t: x0 + (t - t0) / (t1 - t0) * w
    hs = next(t for t, l in mb if l == "blk23:fwd"); he = next(t for t, l in mb if l == "blk23:bwd")
    b.append(f'<rect x="{xs(hs):.1f}" y="{y0}" width="{xs(he) - xs(hs):.1f}" height="{y1 - y0}" fill="#fdeee6"/>')
    b.append(area(tr, x0, w, t0, t1, y0, y1, vmax, BLUE))
    for t in range(0, int(t1 - t0) + 1, 10): b += [P(f"M{xs(t0 + t):.1f} {y1}v5", INK2, 1), T(xs(t0 + t), y1 + 19, str(t), "d-m", "middle")]
    b.append(T(x0 + w, y1 + 19, "ms", "d-m", "start", "transform:translate(8px,0)"))
    fw = [s for s in base[:60] if kind(s["label"]) == "fwd"][:23]; bw = [s for s in base[:60] if kind(s["label"]) == "bwd"][:24]; head = max(base[:60], key=lambda s: s["total"])
    b += [T((xs(t0) + xs(hs)) / 2, y0 + 16, "24 blocks, forward", "d-b", "middle"), T((xs(t0) + xs(hs)) / 2, y0 + 34, f"{sum(s['total'] for s in fw) / len(fw):.2f} GB and {sum(s['ms'] for s in fw) / len(fw):.1f} ms each", "d-m", "middle"),
          T((xs(he) + xs(t1)) / 2, y0 + 16, "24 blocks, backward", "d-b", "middle"), T((xs(he) + xs(t1)) / 2, y0 + 34, f"{sum(s['total'] for s in bw) / len(bw):.2f} GB and {sum(s['ms'] for s in bw) / len(bw):.1f} ms each", "d-m", "middle"),
          T((xs(hs) + xs(he)) / 2, y0 - 22, "output layer and loss", "d-b", "middle", f"fill:{ORANGE}"), T((xs(hs) + xs(he)) / 2, y0 - 6, f"{head['total']:.2f} GB in {head['ms']:.1f} ms, no checkpoint inside", "d-m", "middle")]
    # close-up: the split variant, which has a checkpoint between the operators
    tr2, mk2 = trace("split"); mb2 = first_mb(mk2); ha = next(t for t, l in mb2 if l == "head:linear"); hb = next(t for t, l in mb2 if l == "blk23:bwd")
    z0, z1 = 340, 530; b.append(T(x0, z0 - 36, "Close-up of the output layer and loss, with a checkpoint between its operators (variant 'split')", "d-b"))
    yaxis(b, x0, x0 + w, z0, z1, vmax, [0, 400, 800, 1200, 1600], "GB/s")
    zx = lambda t: x0 + (t - ha) / (hb - ha) * w
    b.append(area(tr2, x0, w, ha, hb, z0, z1, vmax, ORANGE))
    names = {"head:linear": "matrix product (the logits)", "head:cast": "logits to float32", "head:logsoftmax": "log-softmax", "head:nll": "target's value; zero the gradient",
             "head:logsoftmax_bwd": "log-softmax, backward", "head:cast_bwd": "gradient back to bf16", "head:linear_bwd": "matrix product, backward"}
    hm = [(t, l) for t, l in mb2 if ha <= t <= hb]; sp = {s["label"]: s for s in split[:60] if s["label"].startswith("head:")}
    for n, ((ta, la), (tb, _)) in enumerate(zip(hm, hm[1:]), 1):
        b.append(P(f"M{zx(ta):.1f} {z0 - 4}V{z1}", INK2, 1, "3 3")); xm = (zx(ta) + zx(tb)) / 2
        b.append(f'<circle cx="{xm:.1f}" cy="{z0 - 10}" r="9" fill="{INK}"/>'); b.append(T(xm, z0 - 6, str(n), "d-s", "middle", "fill:#fff;font-weight:600"))
        col, row = (n - 1) // 4, (n - 1) % 4; lx, ly = x0 + col * 400, z1 + 50 + row * 22
        b.append(f'<circle cx="{lx + 9}" cy="{ly - 4}" r="9" fill="{INK}"/>'); b.append(T(lx + 9, ly, str(n), "d-s", "middle", "fill:#fff;font-weight:600"))
        b.append(T(lx + 26, ly, f"{names[la]}", "d-s")); b.append(T(lx + 380, ly, f"{sp[la]['total']:.2f} GB · {sp[la]['ms']:.1f} ms", "d-b", "end"))
    for t in range(0, int(hb - ha) + 1, 2): b += [P(f"M{zx(ha + t):.1f} {z1}v5", INK2, 1), T(zx(ha + t), z1 + 19, str(t), "d-m", "middle")]
    b.append(T(x0 + w + 8, z1 + 19, "ms", "d-m"))
    add("prof_trace", "One micro-batch of the tenant alone on 32 SMs: traffic to and from GPU memory, sampled 10,000 times a second", 880, 690, "".join(b))

    g = lambda ss, k: sum(s["total"] for s in ss if kind(s["label"]) == k)
    # ------------------------------------------------------------------ the variants: bytes per step, largest piece, speed alone
    label = {"base": "usual tenant", "split": "same operators, checkpoints between them", "compile": "compiled (BWR-Compile)"}
    def lab(v):
        if v in label: return label[v]
        n = int(v.split("vocab")[1]); return ("compiled body, " if v.startswith("compile") else "") + f"vocabulary blocks of {n:,}"
    label.update({"cb_vocab4096": "changes one + two: and every layer compiled on its own", "cb_vocab4096f": "changes one + two + three: and fused kernels (new tenant)"})   # night report
    night = {"base": "usual tenant", "vocab4096": "change one: output layer over vocabulary blocks of 4,096", "compile": "for reference: compiled as a whole (Nov 16)"}
    plain = ("base", "split", "vocab32768", "vocab16384", "vocab8192", "vocab4096", "vocab2048", "vocab1024")
    for name, title, want in (("prof_variants", "The output layer over vocabulary blocks, tenant alone on 32 SMs: bytes per step by block size", plain),
                              ("prof_variants_all", "Every tenant variant alone on 32 SMs: bytes per step, largest piece that cannot be paused, speed", plain + ("compile", "compile_vocab4096", "compile_vocab2048")),
                              ("night_tenants", "The three changes to the tenant, alone on 32 SMs: data per step, largest piece that cannot be paused, speed", ("base", "vocab4096", "cb_vocab4096", "cb_vocab4096f", "compile"))):
        order = [v for v in want if v in S]; x0, w, vmax = (360, 190, 75) if name == "night_tenants" else (282, 250, 75); b = []; y = 34
        b += [T(x0, 18, "GB per step", "d-m"), T(x0 + w + 104, 18, "largest piece", "d-m", "end"), T(x0 + w + 200, 18, "MB per token", "d-m", "end"), T(x0 + w + 306, 18, "tokens/s, alone", "d-m", "end")]
        for v in order:
            r = S[v]; ss = segs(v); step = float(r["step_gb"])
            hd = sum(s["total"] for s in ss if s["total"] > 5) if v == "base" else float(r["head_gb"]) if r.get("head_gb") else None   # usual tenant: the four segments that hold the output layer
            b.append(T(x0 - 10, y + 15, (night.get(v) or label.get(v) or lab(v)) if name == "night_tenants" else lab(v), "d-s", "end"))
            if hd is None: b.append(R(x0, y, step / vmax * w, 20, PURPLE)); b.append(T(x0 + step / vmax * w + 6, y + 15, f"{step:.1f}", "d-b"))
            else:
                b.append(R(x0, y, (step - hd) / vmax * w - 1, 20, BLUE)); b.append(R(x0 + (step - hd) / vmax * w, y, hd / vmax * w, 20, ORANGE)); b.append(T(x0 + step / vmax * w + 6, y + 15, f"{step:.1f}", "d-b"))
            big = float(r["largest_gb"]); b.append(T(x0 + w + 104, y + 15, f"{big:.2f} GB", "d-b" if big > 1 else "d-s", "end", f"fill:{RED}" if big > 1 else None))
            b.append(T(x0 + w + 200, y + 15, r["mb_per_token"], "d-s", "end")); b.append(T(x0 + w + 306, y + 15, f"{int(r['tok_s']):,}", "d-s", "end"))
            y += 30
        lx = 200 if name == "night_tenants" else x0
        ly = y + 12; b += [R(lx, ly, 12, 10, BLUE), T(lx + 18, ly + 9, "the 24 layers and the rest" if name == "night_tenants" else "the 24 blocks and the rest", "d-m"), R(lx + 190, ly, 12, 10, ORANGE), T(lx + 208, ly + 9, "output layer and loss", "d-m")]
        if any(v == "compile" for v in order): b += [R(lx + 350, ly, 12, 10, PURPLE), T(lx + 368, ly + 9, "cannot be told apart (no place to pause)" if name == "night_tenants" else "not separable (no checkpoints)", "d-m")]
        add(name, title, 880, ly + 26, "".join(b))

    # ------------------------------------------------------------------ one micro-batch with the output layer over vocabulary blocks of 2,048, on the scale of the usual tenant
    V = "vocab2048"
    if V in S:
        tr2, mk2 = trace(V); m2 = first_mb(mk2); u0, u1 = m2[0][0], m2[-1][0]; trb, mkb = trace("base"); mbb = first_mb(mkb); c0 = mbb[0][0]
        x0, w, y0, y1, vmax = 70, 760, 60, 250, 1600; tmax = 110; b = []
        yaxis(b, x0, x0 + w, y0, y1, vmax, [0, 400, 800, 1200, 1600], "GB/s")
        xs = lambda t: x0 + t / tmax * w; ys = lambda v: y1 - min(v, vmax) / vmax * (y1 - y0)
        p1 = next(t for t, l in m2 if l == "head:pass1") - u0; p2 = next(t for t, l in m2 if l == "head:pass2") - u0; hb = next(t for t, l in m2 if l == "head:surrogate") - u0
        b.append(f'<rect x="{xs(p1):.1f}" y="{y0}" width="{xs(hb) - xs(p1):.1f}" height="{y1 - y0}" fill="#fdeee6"/>')
        b.append(P(f"M{xs(p2):.1f} {y0}V{y1}", ORANGE, 1, "3 3"))
        use = [(t - c0, v) for t, v in trb if c0 <= t <= mbb[-1][0]]
        b.append(P("M" + "L".join(f"{xs(t):.1f} {ys(v):.1f}" for t, v in use), GRAY, 1.2))
        b.append(area([(t - u0, v) for t, v in tr2 if u0 <= t <= u1], x0, w, 0, tmax, y0, y1, vmax, BLUE))
        for t in range(0, tmax + 1, 10): b += [P(f"M{xs(t):.1f} {y1}v5", INK2, 1), T(xs(t), y1 + 19, str(t), "d-m", "middle")]
        b.append(T(x0 + w + 24, y1 + 19, "ms", "d-m"))
        ss = segs(V); n = len(ss); q = [s for s in ss[: n // 4 + 2]]
        g1 = sum(s["total"] for s in ss if s["label"] == "head:pass1") / 4; g2 = sum(s["total"] for s in ss if s["label"] == "head:pass2") / 4
        b += [T((xs(p1) + xs(p2)) / 2, y0 - 26, "pass 1", "d-b", "middle", f"fill:{ORANGE}"), T((xs(p1) + xs(p2)) / 2, y0 - 10, f"{g1:.2f} GB", "d-m", "middle"),
              T((xs(p2) + xs(hb)) / 2, y0 - 26, "pass 2", "d-b", "middle", f"fill:{ORANGE}"), T((xs(p2) + xs(hb)) / 2, y0 - 10, f"{g2:.2f} GB", "d-m", "middle"),
              T(xs(p1) / 2, y0 - 10, "24 blocks, forward", "d-m", "middle"), T((xs(hb) + xs(u1 - u0)) / 2, y0 - 10, "24 blocks, backward", "d-m", "middle")]
        ly = y1 + 44
        b += [R(x0, ly, 14, 10, BLUE), T(x0 + 20, ly + 9, f"output layer over vocabulary blocks of 2,048: {float(S[V]['step_gb']) / 4:.1f} GB in {u1 - u0:.0f} ms, largest piece {float(S[V]['largest_gb']):.2f} GB", "d-s"),
              P(f"M{x0} {ly + 27}h14", GRAY, 1.6), T(x0 + 20, ly + 31, f"usual tenant: {float(S['base']['step_gb']) / 4:.1f} GB in {mbb[-1][0] - c0:.0f} ms, largest piece {float(S['base']['largest_gb']):.2f} GB", "d-s")]
        add("prof_trace_vocab2048", "One micro-batch with the output layer over vocabulary blocks of 2,048, on the scale of the usual tenant", 880, ly + 44, "".join(b))

    # ------------------------------------------------------------------ how the profile is made
    b = []; bx = lambda x, y, w, h, cls: f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>'
    b += [bx(20, 20, 250, 96, "d-ten"), T(34, 44, "Tenant, alone, as it runs", "d-b"), T(34, 64, "MPS, green context of 32 SMs", "d-s"), T(34, 82, "at every checkpoint: wait for the", "d-s"), T(34, 98, "GPU, then emit a mark (NVTX)", "d-s"),
          bx(20, 140, 250, 78, "d-box"), T(34, 164, "Nsight Systems, GPU metrics", "d-b"), T(34, 184, "memory read and write throughput,", "d-s"), T(34, 200, "10,000 samples a second, in %", "d-s"),
          bx(330, 60, 240, 116, "d-soft"), T(344, 84, "Per segment", "d-b"), T(344, 104, "integrate the samples between", "d-s"), T(344, 120, "two marks; average over the", "d-s"), T(344, 136, "steps", "d-s"), T(344, 160, "× GB/s per 1 %", "d-s"),
          bx(330, 200, 240, 62, "d-box"), T(344, 224, "Calibration", "d-b"), T(344, 244, "copies of a 1 GiB tensor", "d-s"),
          bx(630, 60, 230, 116, "d-good"), T(644, 84, "Table of a step", "d-b"), T(644, 104, "one row per segment:", "d-s"), T(644, 120, "bytes read, bytes written,", "d-s"), T(644, 136, "time, running total", "d-s"),
          bx(630, 200, 230, 62, "d-box"), T(644, 224, "Checks", "d-b"), T(644, 244, "per-kernel counters; old counter", "d-s"),
          P("M270 68H326", INK2, 1.5, None, True), P("M270 180H300V130H326", INK2, 1.5, None, True), P("M450 200V180", INK2, 1.5, None, True), P("M570 118H626", INK2, 1.5, None, True), P("M745 200V180", INK2, 1.5, None, True)]
    add("prof_method", "How the profile of a tenant step is measured", 880, 280, "".join(b))

if __name__ == "__main__":
    build(); F.render(set(sys.argv[1:]) or None)
