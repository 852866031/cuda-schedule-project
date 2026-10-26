#!/usr/bin/env python3
"""KV-slot study: what the decode engine's free KV space does to each request.

One column per cell (same workload, same QPS, different decode VRAM), one shared
time window (seconds since the measurement started). Rows:
  (a) the KV slots: every request that holds KV is a bar on a lane (orange = loading its
      0.75 GiB of KV from the host, blue = decoding); requests waiting for a slot are
      grey bars in the queue band above. The dashed line marks the whole-request slots
      the grant holds (c = floor(blocks / 400)).
  (b) free KV space in "requests that still fit" (= free blocks / 400) and the number of
      requests waiting, both in requests.
  (c) how long each decode step took: every step that overlaps a KV load (orange) stalls
      the whole batch.
One request present in both cells is followed and annotated.

    .venv/bin/python scripts/plots/plot_kv_timeline.py kv_base_b24_q2.5_tr kv_base_b28_q2.5_tr \
        [--t0 S --span S] [--follow INDEX] [--out figures/kv_slots_timeline.png]
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent.parent
TL = REPO / "output" / "kv_vram" / "timeline"
RAW = REPO / "output" / "raw"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "legend.fontsize": 11})
C_LOAD, C_DEC, C_WAIT, C_FREE, C_Q, C_STEP = "#eb6834", "#2a78d6", "#9a9a95", "#1baf7a", "#57606a", "#2a78d6"
REQ_BLOCKS = 400


def rd(name, kind):
    return list(csv.DictReader(open(TL / f"{name}_{kind}.csv")))


def f(x):
    return float(x) if x not in ("", None) else np.nan


def lanes(reqs):
    """Greedy lane assignment over [t_sched, t_fin)."""
    ends = []
    out = {}
    for r in sorted(reqs, key=lambda r: r["s"]):
        for i, e in enumerate(ends):
            if e <= r["s"]:
                ends[i] = r["e"]
                out[r["index"]] = i
                break
        else:
            ends.append(r["e"])
            out[r["index"]] = len(ends) - 1
    return out


def cell(name):
    rec = json.loads((RAW / f"{name}.json").read_text())
    base = rec["t_mono_measure0"]
    rq = []
    for r in rd(name, "req"):
        if r.get("t_sched") in ("", None) or r.get("t_fin") in ("", None):
            continue
        rq.append({"index": int(r["index"]), "a": f(r["t_arr"]) - base, "s": f(r["t_sched"]) - base,
                   "f": f(r["t_first"]) - base, "e": f(r["t_fin"]) - base,
                   "sub": f(r.get("t_submit")) - base if r.get("t_submit") else np.nan,
                   "free_arr": int(r["free_at_arr"]), "pre": bool(r.get("preempted"))})
    st = [(f(s["t"]) - base, int(s["run"]), int(s["wait"]), int(s["free"])) for s in rd(name, "state")]
    steps = [(f(s["t"]) - base, f(s["dt"])) for s in rd(name, "steps")]
    loads = [(f(l["t"]) - base, f(l["dt"])) for l in rd(name, "loads")]
    slots = rec["launch"]["decode"]["slots"]
    blocks = rec["launch"]["decode"]["kv_blocks"]
    return rec, rq, np.array(st), np.array(steps), np.array(loads), slots, blocks


def pick_window(rq, span):
    """Window centred on the longest queue wait."""
    worst = max(rq, key=lambda r: r["s"] - r["a"])
    t0 = max(0.0, worst["a"] - 0.4 * span)
    return t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+")
    ap.add_argument("--t0", type=float)
    ap.add_argument("--span", type=float, default=30.0)
    ap.add_argument("--follow", type=int)
    ap.add_argument("--out", default=str(REPO / "figures" / "kv_slots_timeline.png"))
    args = ap.parse_args()
    data = [cell(n) for n in args.cells]
    t0 = args.t0 if args.t0 is not None else pick_window(data[0][1], args.span)
    t1 = t0 + args.span
    if args.follow is None:
        cand = [r for r in data[0][1] if t0 + 0.15 * args.span < r["a"] < t0 + 0.6 * args.span]
        if len(data) > 1:       # the request whose wait differs most between the two cells
            w1 = {r["index"]: r["s"] - r["a"] for r in data[1][1]}
            args.follow = max(cand, key=lambda r: (r["s"] - r["a"]) - w1.get(r["index"], 0))["index"]
        else:
            args.follow = max(cand, key=lambda r: r["s"] - r["a"])["index"]

    ncol = len(data)
    fig, axes = plt.subplots(4, ncol, figsize=(7.2 * ncol, 11.6), sharex=True,
                             gridspec_kw={"height_ratios": [0.55, 3.2, 1.4, 1.4], "hspace": 0.42,
                                          "wspace": 0.12}, squeeze=False)
    ymax_lane = max(max(lanes(d[1]).values()) + 1 for d in data)
    qmax = 0
    for rec, rq, st, steps, loads, slots, blocks in data:
        m = (st[:, 0] > t0) & (st[:, 0] < t1)
        qmax = max(qmax, int(st[m, 2].max()) if m.any() else 0)
    q0 = ymax_lane + 0.8                                  # queue band starts above lanes
    nq = 0
    for j, (rec, rq, st, steps, loads, slots, blocks) in enumerate(data):
        r0, a, b, c = axes[0, j], axes[1, j], axes[2, j], axes[3, j]
        ln = lanes(rq)
        # (0) the arrival process alone: one tick per submitted request
        subs = np.array(sorted(r["sub"] for r in rq if not np.isnan(r["sub"])))
        win = [x for x in subs if t0 <= x <= t1]
        for x in win:
            r0.plot([x, x], [0, 1], color="black", lw=1.1)
        r0.set_ylim(-0.1, 1.5)
        r0.set_yticks([])
        if j == 0:
            r0.set_ylabel("arrivals")
        r0.text(0.99, 0.93, f"(0) {len(win)} arrivals in this window, one tick each ({rec['qps']:g} req/s mean)",
                transform=r0.transAxes, fontsize=8.5, va="top", ha="right")
        r0.set_title(f"decode {rec['budget_gib']:g} GiB: KV grant {rec['launch']['decode']['kv_gib']:.2f} GiB"
                     f" = {blocks:,} blocks, {rec['qps']:g} req/s")
        # (a) slots
        waiting = sorted([r for r in rq if r["s"] - r["a"] > 0.02 and r["s"] > t0 and r["a"] < t1],
                         key=lambda r: r["a"])
        qlane, qends = {}, []
        for r in waiting:
            for i, e in enumerate(qends):
                if e <= r["a"]:
                    qends[i] = r["s"]; qlane[r["index"]] = i; break
            else:
                qends.append(r["s"]); qlane[r["index"]] = len(qends) - 1
        nq = max(nq, len(qends))
        for r in rq:
            if r["e"] < t0 or r["s"] > t1:
                continue
            y = ln[r["index"]]
            fol = r["index"] == args.follow
            a.barh(y, r["f"] - r["s"], left=r["s"], height=0.72, color=C_LOAD,
                   edgecolor="black" if fol else "none", lw=1.6)
            a.barh(y, r["e"] - r["f"], left=r["f"], height=0.72, color=C_DEC,
                   edgecolor="black" if fol else "none", lw=1.6,
                   hatch="///" if r["pre"] else None)
        for r in waiting:
            y = q0 + qlane[r["index"]] * 0.55
            fol = r["index"] == args.follow
            a.barh(y, r["s"] - r["a"], left=r["a"], height=0.42, color=C_WAIT,
                   edgecolor="black" if fol else "none", lw=1.6)
        a.axhline(slots - 0.5, color="black", ls="--", lw=1)
        a.text(t1 - 0.2, slots - 0.5 + 0.12, f"{blocks / REQ_BLOCKS:.2f} requests fit", ha="right",
               va="bottom", fontsize=9, bbox=dict(fc="white", ec="none", pad=1))
        a.axhline(q0 - 0.45, color="#d0d0cc", lw=0.8)
        a.set_ylim(-0.7, q0 + max(1, nq) * 0.55 + 3.4)      # headroom for the annotation box
        a.set_yticks(list(range(ymax_lane)) + [q0 + (max(1, nq) - 1) * 0.275])
        a.set_yticklabels([str(i + 1) for i in range(ymax_lane)] + ["queue"], fontsize=9)
        if j == 0:
            a.set_ylabel("KV slot (lane)")
        a.text(0.01, 0.985, "(a) the same requests packed into the KV slots", transform=a.transAxes,
               fontsize=8.5, va="top")
        # follow annotation
        fr = next((r for r in rq if r["index"] == args.follow), None)
        if fr is not None:
            wait = fr["s"] - fr["a"]
            free_req = fr["free_arr"] / REQ_BLOCKS
            txt = (f"request #{args.follow}: arrives t={fr['a']:.1f} s, "
                   f"{fr['free_arr']} blocks free ({free_req:.1f} requests)\n"
                   r"$\bf{waits\ " + f"{wait * 1000:.0f}" + r"\ ms}$" + f", loads {1000 * (fr['f'] - fr['s']):.0f} ms, "
                   f"decodes {fr['e'] - fr['f']:.1f} s")
            ytxt = ln[args.follow] if wait < 0.02 else q0 + qlane.get(args.follow, 0) * 0.55
            a.annotate(txt, xy=(fr["a"], ytxt), xytext=(0.02, 0.985), textcoords="axes fraction",
                       va="top", fontsize=9.5, bbox=dict(fc="white", ec="#888", lw=0.6),
                       arrowprops=dict(arrowstyle="->", color="black", lw=0.9))
        # (b) free space and queue, in requests
        m = (st[:, 0] > t0 - 1) & (st[:, 0] < t1 + 1)
        b.fill_between(st[m, 0], 0, st[m, 2], step="post", color=C_WAIT, alpha=0.8, lw=0)
        b.set_ylim(0, max(qmax, 1) + 0.5)
        if j == 0:
            b.set_ylabel("requests\nwaiting")
        b.text(0.01, 0.95, "(b) how many requests are waiting for a slot", transform=b.transAxes,
               fontsize=8.5, va="top")
        # (c) step durations
        ms = (steps[:, 0] > t0) & (steps[:, 0] < t1) & ~np.isnan(steps[:, 1])
        c.plot(steps[ms, 0], 1000 * steps[ms, 1], ".", ms=2.5, color=C_STEP)
        own = None
        if fr is not None:
            own = next(((lt, ld) for lt, ld in loads if lt <= fr["s"] + 0.05 and lt + ld >= fr["s"]), None)
        for lt, ld in loads:
            if t0 < lt < t1:
                is_own = own is not None and lt == own[0]
                c.axvspan(lt, lt + ld, color=C_LOAD, alpha=0.6 if is_own else 0.22, lw=0)
        c.set_yscale("log")
        c.set_ylim(5, 800)
        c.text(0.01, 0.95, "(c) wall time of each engine step; orange = a KV load call inside the step",
               transform=c.transAxes, fontsize=8.5, va="top")
        if fr is not None:
            for ax_ in (b, c):
                ax_.axvline(fr["a"], color="black", lw=1, ls="--")
                ax_.axvline(fr["s"], color="black", lw=1, ls=":")
            b.text(fr["a"], b.get_ylim()[1] * 0.78, f" #{args.follow} arrives", fontsize=8, va="center")
            b.text(fr["s"], b.get_ylim()[1] * 0.6,
                   f" #{args.follow} admitted" if fr["s"] - fr["a"] > 0.3 else "  admitted at once",
                   fontsize=8, va="center")
        if j == 0:
            c.set_ylabel("engine step (ms)")
        c.set_xlabel("time since the run started (s)")
        c.set_xlim(t0, t1)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#ececea", lw=0.6)
    def row_legend(row, handles, ncol):
        y1 = axes[row, 0].get_position().y1
        fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=9.5,
                   bbox_to_anchor=(0.5, y1 + 0.003), bbox_transform=fig.transFigure)
    row_legend(1, [Patch(color=C_LOAD, label="loading its KV from host (the whole batch stops)"),
                   Patch(color=C_DEC, label="decoding (holds its 400 blocks)"),
                   Patch(fc=C_DEC, hatch="///", ec="white", label="decoding, was preempted once"),
                   Patch(color=C_WAIT, label="waiting for a slot"),
                   Line2D([], [], color="black", lw=1.6, label=f"outlined: request #{args.follow}")], 5)
    row_legend(2, [Patch(color=C_WAIT, alpha=0.8, label="requests waiting for a slot"),
                   Line2D([], [], color="black", lw=1, ls="--", label=f"#{args.follow} arrives"),
                   Line2D([], [], color="black", lw=1, ls=":", label=f"#{args.follow} admitted")], 3)
    row_legend(3, [Line2D([], [], color=C_STEP, marker=".", ls="", label="one engine step (one forward pass)"),
                   Patch(color=C_LOAD, alpha=0.22, label="a synchronous KV load call (start_load_kv)"),
                   Patch(color=C_LOAD, alpha=0.6, label=f"the call that loads #{args.follow}")], 3)
    fig.suptitle("Decode engine KV slots, queue and step time per request "
                 "(Llama-3-8B decode on GPU1, 6400-token requests)", y=0.95, fontsize=13)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("->", args.out, "window", round(t0, 1), "follow", args.follow)


if __name__ == "__main__":
    main()
