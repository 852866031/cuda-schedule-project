#!/usr/bin/env python3
"""KV-slot study, D10/D11: the asynchronous load options against the synchronous compiled
baseline, per budget and rate. Per cell from the joined timelines:

  slot wait  = vLLM queue wait − the asynchronous prefetch (lookup asked → answered);
               the synchronous arms have no prefetch, so slot wait = queue wait
  residence  = admitted → last token
  throughput = output tokens / s (summary)

    .venv/bin/python scripts/plots/plot_kv_async.py [method ...]
        -> figures/kv_async.png, output/kv_vram/async_summary.csv
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
TL = REPO / "output" / "kv_vram" / "timeline"
QPS = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
BUDGETS = [24, 26, 28]
ARMS = {"cops": ("synchronous (compiled kernels)", "#9a9a95", "o", "-"),
        "cops_async": ("enable_async_loading", "#2a78d6", "s", "-"),
        "mp18": ("MP connector (vLLM 0.18.1)", "#eb6834", "^", "--")}
SUFFIX = {"cops": ["_tr_r2", "_tr"], "cops_async": ["_tr_g"], "mp18": ["_tr_g", "_tr"]}


def cell(method, b, q):
    for suf in SUFFIX[method]:
        p = TL / f"kv_{method}_b{b}_q{q:g}{suf}_req.csv"
        if p.exists():
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

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, gridspec_kw={"hspace": 0.28, "wspace": 0.25})
    for j, b in enumerate(BUDGETS):
        for m in methods:
            rs = [r for r in rows if r["method"] == m and r["budget"] == b]
            if not rs:
                continue
            lab, c, mk, ls = ARMS[m]
            x = [r["qps"] for r in rs]
            axes[0, j].plot(x, [max(r["slot_wait_ms"], 0) / 1e3 for r in rs], color=c, marker=mk, ls=ls, label=lab)
            axes[1, j].plot(x, [r["tok_s"] for r in rs], color=c, marker=mk, ls=ls, label=lab)
            axes[2, j].plot(x, [r["resid_s"] for r in rs], color=c, marker=mk, ls=ls, label=lab)
        axes[0, j].set_yscale("symlog", linthresh=0.01)
        axes[0, j].axhline(0.05, color="#b3261e", lw=0.9, ls=":")
        axes[0, j].set_title(f"{b} GiB", fontsize=13)
        axes[2, j].set_xlabel("request rate (req/s)")
        for i in range(3):
            axes[i, j].grid(color="#ececea", lw=0.6)
            axes[i, j].spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("mean wait for a slot (s)")
    axes[1, 0].set_ylabel("output tokens / s")
    axes[2, 0].set_ylabel("residence (s)")
    axes[0, 0].text(1.0, 0.06, "50 ms capacity line", color="#b3261e", fontsize=9)
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(h), frameon=False, bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("Asynchronous load options vs the synchronous baseline (slot wait excludes the async prefetch)",
                 y=1.0, fontsize=14)
    fig.savefig(REPO / "figures" / "kv_async.png", dpi=150, bbox_inches="tight")
    print("-> figures/kv_async.png")


if __name__ == "__main__":
    main(sys.argv[1:] or ["cops", "cops_async"])
