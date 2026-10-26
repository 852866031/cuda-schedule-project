#!/usr/bin/env python3
"""KV-slot study, D10/D11: the asynchronous load options against the synchronous compiled
baseline, per budget and rate. Per cell from the joined timelines:

  slot wait  = vLLM queue wait − the asynchronous prefetch (lookup asked → answered);
               the synchronous arms have no prefetch, so slot wait = queue wait
  residence  = admitted → last token
  throughput = output tokens / s (summary)

    .venv/bin/python scripts/plots/plot_kv_async.py [method ...]
        -> figures/kv_async.png (§6.3), figures/kv_mp.png (§6.4), output/kv_vram/async_summary.csv
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
TL = REPO / "output" / "kv_vram" / "timeline"
QPS = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
BUDGETS = [24, 26, 28]
ARMS = {"cops": ("synchronous (compiled kernels)", "#9a9a95", "o", "-"),
        "cops_async": ("enable_async_loading", "#2a78d6", "s", "-"),
        "mp18": ("MP connector (vLLM 0.18.1)", "#eb6834", "^", "--")}
SUFFIX = {"cops": ["_tr_r2", "_tr"], "cops_async": ["_tr_g"], "mp18": ["_tr_g", "_tr"]}


OK = {r["name"] for r in csv.DictReader(open(REPO / "output" / "summary_kv.csv")) if r["ok"] == "True"}
ALT = {"cops_async": ["cops_async", "cops_async_t30"]}      # D10: crashed cells rerun with a 30 s lookup timeout


def cell(method, b, q):
    for m in ALT.get(method, [method]):
        for suf in SUFFIX[method]:
            name = f"kv_{m}_b{b}_q{q:g}{suf}"
            p = TL / f"{name}_req.csv"
            if p.exists() and name in OK:
                return p
    return None


def summary_row(name):
    for r in csv.DictReader(open(REPO / "output" / "summary_kv.csv")):
        if r["name"] == name:
            return r
    return {}


def stats(p):
    rq = [r for r in csv.DictReader(open(p)) if r.get("resident")]
    f = lambda k: np.array([float(r[k]) for r in rq if r.get(k) not in ("", None)])
    wait = f("wait")
    pre = f("prefetch") if any(r.get("prefetch") for r in rq) else np.zeros(len(rq))
    slot = f("slot_wait") if any(r.get("slot_wait") for r in rq) else wait
    s = summary_row(p.name[:-len("_req.csv")])
    return {"wait_ms": wait.mean() * 1e3, "slot_wait_ms": slot.mean() * 1e3,
            "prefetch_ms": float(np.median(pre)) * 1e3 if len(pre) else 0.0,
            "resid_s": f("resident").mean(), "load_s": f("load").mean(),
            "tok_s": float(s.get("tok_per_s") or "nan"), "req_s": float(s.get("req_per_s") or "nan"),
            "tpot_ms": float(s.get("tpot_p50_ms") or "nan"), "preempt": s.get("preemptions", "")}


def main(methods):
    rows = []
    for m in methods:
        for b in BUDGETS:
            for q in QPS:
                p = cell(m, b, q)
                if p:
                    rows.append({"method": m, "budget": b, "qps": q, "cell": p.name[:-8], **stats(p)})
    out = REPO / "output" / "kv_vram" / "async_summary.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out.relative_to(REPO)} ({len(rows)} cells)")

    # one figure per load-path arm against the synchronous baseline, with the metrics of
    # report §4.1 (queue wait, throughput, end-user TPOT) plus residence
    for arm, fname, title in (("cops_async", "kv_async.png", "enable_async_loading"),
                              ("mp18", "kv_mp.png", "MP connector (vLLM 0.18.1)")):
        if arm in methods:
            plot_arm(rows, ["cops", arm], fname, title)
            plot_arm_3d(rows, arm, fname.replace(".png", "_3d.png"), title)


def plot_arm_3d(rows, arm, fname, title):
    """§4.1's surfaces (queue wait, throughput, TPOT) plus residence over budget x rate:
    the load-path arm as a coloured surface, the synchronous baseline as a grey wireframe.
    Cells that did not complete (async at 24 GiB, 4-5 req/s) are holes."""
    from matplotlib import cm
    panels = [("wait_ms", 1e-3, "mean decode queue wait (s)", "(a) decode queue wait", cm.Blues, False),
              ("tok_s", 1, "output tokens / s", "(b) throughput", cm.Greens, False),
              ("tpot_ms", 1, "TPOT p50, end-user (ms, log)", "(c) TPOT p50, end-user", cm.Oranges, True),
              ("resid_s", 1, "residence (s)", "(d) residence", cm.Purples, False)]
    B, Q = np.meshgrid(BUDGETS, QPS, indexing="ij")

    def grid(m, col, sc, log):
        Z = np.full(B.shape, np.nan)
        for r in rows:
            if r["method"] == m:
                i, j = BUDGETS.index(r["budget"]), QPS.index(r["qps"])
                Z[i, j] = r[col] * sc
        return np.log10(Z) if log else Z

    fig = plt.figure(figsize=(15, 12.5))
    lab, c, _, _ = ARMS[arm]
    for k, (col, sc, zl, ttl, cmap, log) in enumerate(panels):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        Zs, Za = grid("cops", col, sc, log), grid(arm, col, sc, log)
        ax.plot_surface(B, Q, np.ma.masked_invalid(Za), cmap=cmap, edgecolor="#333", lw=0.5, alpha=0.85,
                        rstride=1, cstride=1)
        ax.plot_wireframe(B, Q, Zs, color="#6b6b66", lw=1.3, rstride=1, cstride=1)
        for i, b in enumerate(BUDGETS):
            for j, q in enumerate(QPS):
                if not np.isnan(Za[i, j]):
                    ax.scatter([b], [q], [Za[i, j]], color=c, s=16, depthshade=False)
        if log:
            ticks = [15, 20, 30, 50, 100, 200]
            ax.set_zticks(np.log10(ticks))
            ax.set_zticklabels([str(t) for t in ticks])
        ax.set_xlabel("VRAM budget (GiB)", labelpad=8)
        ax.set_ylabel("request rate (req/s)", labelpad=8)
        ax.set_zlabel(zl, labelpad=10)
        ax.set_xticks(BUDGETS)
        ax.view_init(elev=24, azim=-55)
        ax.set_box_aspect(None, zoom=0.82)
        ax.set_title(ttl, pad=4)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend([Patch(fc=c, alpha=0.8, ec="#333"), Line2D([], [], color="#6b6b66", lw=1.3)],
               [f"{lab} (surface, dots = measured cells)", "synchronous, compiled kernels (grey wireframe)"],
               loc="upper center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"{title} over the grid: x = decode VRAM budget, y = request rate, z = the measured quantity",
                 y=0.985, fontsize=14)
    fig.savefig(REPO / "figures" / fname, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"-> figures/{fname}")


def plot_arm(rows, methods, fname, title):
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), sharex=True, gridspec_kw={"hspace": 0.3, "wspace": 0.25})
    for j, b in enumerate(BUDGETS):
        for m in methods:
            rs = [r for r in rows if r["method"] == m and r["budget"] == b]
            if not rs:
                continue
            lab, c, mk, ls = ARMS[m]
            x = [r["qps"] for r in rs]
            axes[0, j].plot(x, [r["wait_ms"] / 1e3 for r in rs], color=c, marker=mk, ls="-", label=lab)
            if m != "cops":     # the part of that queue time spent waiting for a slot
                axes[0, j].plot(x, [max(r["slot_wait_ms"], 0) / 1e3 for r in rs], color=c, marker=mk, ls=":",
                                mfc="white", label=f"{lab}: wait for a slot only")
            axes[1, j].plot(x, [r["tok_s"] for r in rs], color=c, marker=mk, ls="-", label=lab)
            axes[2, j].plot(x, [r["tpot_ms"] for r in rs], color=c, marker=mk, ls="-", label=lab)
            axes[3, j].plot(x, [r["resid_s"] for r in rs], color=c, marker=mk, ls="-", label=lab)
        axes[0, j].set_yscale("symlog", linthresh=0.01)
        axes[0, j].axhline(0.05, color="#b3261e", lw=0.9, ls=":")
        axes[2, j].set_yscale("log")
        axes[2, j].set_yticks([15, 20, 30, 50, 100, 200])
        axes[2, j].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axes[2, j].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axes[0, j].set_title(f"{b} GiB", fontsize=13)
        axes[3, j].set_xlabel("request rate (req/s)")
        for i in range(4):
            axes[i, j].grid(color="#ececea", lw=0.6)
            axes[i, j].spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("(a) mean decode queue wait (s)")
    axes[1, 0].set_ylabel("(b) output tokens / s")
    axes[2, 0].set_ylabel("(c) TPOT p50, end-user (ms, log)")
    axes[3, 0].set_ylabel("(d) residence (s)")
    axes[0, 0].text(1.0, 0.06, "50 ms capacity line", color="#b3261e", fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(h), frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(f"{title} vs the synchronous baseline: the metrics of §4.1 plus residence", y=0.98, fontsize=14)
    fig.savefig(REPO / "figures" / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> figures/{fname}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["cops", "cops_async", "mp18"])
