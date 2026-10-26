#!/usr/bin/env python3
"""KV-slot study, baseline grid figures (from output/summary_kv.csv, output/kv_vram/mgc.csv
and the per-request timeline CSVs):

  kv_grid_3d.png      3D surface: decode VRAM x request rate -> decode queue wait
  kv_grid_heat.png    heatmaps with contours: queue wait, throughput, decode-only TPOT
  kv_mgc.png          measured vs queueing-model wait per budget (FCFS replay with the
                      effective slot count; Lee-Longton M/G/c approximation)
  kv_wait_vs_free.png scatter: free whole-request slots when the request reached the decode
                      scheduler vs its decode queue wait

Cell choice: the traced cells (`_tr`); where a cell was rerun, the clean run (`ok`).

    .venv/bin/python scripts/plots/plot_kv_grid.py
"""
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm

REPO = Path(__file__).resolve().parent.parent.parent
FIGS = REPO / "figures"
TL = REPO / "output" / "kv_vram" / "timeline"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "legend.fontsize": 10})
BUDGETS = [24, 26, 28]
METHOD = os.environ.get("KV_METHOD", "base")      # base = Python-fallback LMCache, cops = compiled
WHO = {"base": "LMCache Python fallback", "cops": "LMCache compiled kernels"}.get(METHOD, METHOD)
C_B = {24: "#2a78d6", 26: "#eb6834", 28: "#1baf7a"}
SLOTS = {24: "10 slots*", 26: "12 slots", 28: "15 slots"}


def load_rows():
    rows = [r for r in csv.DictReader(open(REPO / "output" / "summary_kv.csv"))
            if r["method"] == METHOD and r["trace"] == "True" and r["ok"] == "True"
            and not r.get("trace_invalid")]
    best = {}
    for r in rows:
        key = (float(r["budget_gib"]), float(r["qps"]))
        if key not in best or (r["name"].endswith("_r2") and not best[key]["name"].endswith("_tr")):
            best.setdefault(key, r)
    # prefer the plain `_tr` run when both exist (E1 showed them within noise)
    out = {}
    for r in rows:
        key = (float(r["budget_gib"]), float(r["qps"]))
        if r["name"].endswith("_tr") or key not in out:
            out[key] = r
    return out


ROWS = load_rows()
QPS_ALL = sorted({k[1] for k in ROWS})
QPS_COMMON = [q for q in QPS_ALL if all((b, q) in ROWS for b in BUDGETS)]


def val(b, q, col):
    r = ROWS.get((b, q))
    return float(r[col]) if r and r.get(col) not in (None, "", "None") else np.nan


# ------------------------------------------------------------------ 3D
def fig_3d():
    panels = [("decode_queue_ms", 1e-3, "mean wait for a slot (s)", "(a) decode queue wait", cm.Blues),
              ("tok_per_s", 1, "output tokens / s", "(b) throughput", cm.Greens),
              ("tpot_p50_ms", 1, "ms per token (token 1 → last, log)", "(c) TPOT p50, end-user", cm.Oranges)]
    fig = plt.figure(figsize=(19, 6.4))
    B, Q = np.meshgrid(BUDGETS, QPS_COMMON, indexing="ij")
    for k, (col, sc, zl, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        Z = np.array([[val(b, q, col) * sc for q in QPS_COMMON] for b in BUDGETS])
        if col == "tpot_p50_ms":
            Z = np.log10(Z)
            ax.set_zticks(np.log10([20, 50, 100, 200]))
            ax.set_zticklabels(["20", "50", "100", "200"])
        ax.plot_surface(B, Q, Z, cmap=cmap, edgecolor="#333", lw=0.6, alpha=0.45 if k == 0 else 0.9,
                        rstride=1, cstride=1)
        if k == 0:
            # iso-wait lines drawn ON the surface: contour paths from a 2D contour of the
            # same grid, lifted 0.4 s above the (bilinearly interpolated) surface so the
            # renderer does not hide them under the facets
            def f2(pts):   # bilinear on the (BUDGETS x QPS_COMMON) grid
                out = []
                for x, y in pts:
                    i = min(max(np.searchsorted(BUDGETS, x) - 1, 0), len(BUDGETS) - 2)
                    j = min(max(np.searchsorted(QPS_COMMON, y) - 1, 0), len(QPS_COMMON) - 2)
                    u = (x - BUDGETS[i]) / (BUDGETS[i + 1] - BUDGETS[i])
                    v = (y - QPS_COMMON[j]) / (QPS_COMMON[j + 1] - QPS_COMMON[j])
                    out.append((1 - u) * (1 - v) * Z[i, j] + u * (1 - v) * Z[i + 1, j]
                               + (1 - u) * v * Z[i, j + 1] + u * v * Z[i + 1, j + 1])
                return np.array(out)
            tmp = plt.figure()
            cs2 = tmp.gca().contour(B, Q, Z, levels=[0.05, 0.5, 2, 5, 10])
            for lev, segs in zip(cs2.levels, cs2.allsegs):
                for seg in segs:
                    if len(seg) < 2:
                        continue
                    zz = f2(seg) + 0.8
                    ax.plot(seg[:, 0], seg[:, 1], zz, color="#b3261e", lw=3.2, zorder=10)
                    ax.text(seg[-1, 0], seg[-1, 1], zz[-1] + 0.3, f"{lev:g} s", color="#b3261e",
                            fontsize=8.5, fontweight="bold", ha="center")
            plt.close(tmp)
            ax.text2D(0.02, 0.9, "red: iso-wait lines on the surface at 0.05 (capacity), 0.5, 2, 5, 10 s",
                      transform=ax.transAxes, fontsize=9.5, color="#b3261e")
        for b in BUDGETS:
            for q in QPS_ALL:
                z = val(b, q, col) * sc
                if col == "tpot_p50_ms":
                    z = np.log10(z)
                if not np.isnan(z):
                    ax.scatter([b], [q], [z], color=C_B[b], s=22, depthshade=False)
        ax.set_xlabel("VRAM budget (GiB)", labelpad=8)
        ax.set_ylabel("request rate (req/s)", labelpad=8)
        ax.set_zlabel(zl, labelpad=10)
        ax.set_xticks(BUDGETS)
        ax.view_init(elev=24, azim=-55)
        ax.set_box_aspect(None, zoom=0.82)
        ax.set_title(title, pad=4)
    fig.suptitle(f"The grid as surfaces ({WHO}): x = decode VRAM budget, y = request rate, "
                 "z = the measured quantity (dots = measured cells)", y=0.98)
    fig.savefig(FIGS / "kv_grid_3d.png", dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


# ------------------------------------------------------------------ heatmaps
def fig_heat(contours=False):
    panels = [("decode_queue_ms", 1e-3, "decode queue wait, mean (s)", "(a) queue wait before decode",
               True, [0.05, 0.5, 2, 5, 10]),
              ("tok_per_s", 1, "output tokens / s", "(b) throughput", False, [200, 250, 300, 340]),
              ("tpot_p50_ms", 1, "ms per token (token 1 → last)", "(c) TPOT p50, end-user", True,
               [20, 50, 100])]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    fig.subplots_adjust(wspace=0.5)
    for ax, (col, sc, cb, title, logc, levels) in zip(axes, panels):
        Z = np.array([[val(b, q, col) * sc for q in QPS_COMMON] for b in BUDGETS])
        norm = LogNorm(vmin=max(0.005, np.nanmin(Z[Z > 0])), vmax=np.nanmax(Z)) if logc else None
        if col == "tpot_p50_ms":
            norm = LogNorm(vmin=15, vmax=np.nanmax(Z))
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="Blues", norm=norm,
                       extent=[-0.5, len(QPS_COMMON) - 0.5, -0.5, len(BUDGETS) - 0.5])
        for i, b in enumerate(BUDGETS):
            for j, q in enumerate(QPS_COMMON):
                z = Z[i, j]
                ax.text(j, i, f"{z:.2f}" if col == "decode_queue_ms" and z < 1 else f"{z:.0f}" if z >= 10 else f"{z:.1f}",
                        ha="center", va="center", fontsize=9.5,
                        color="white" if (norm and z > np.sqrt(norm.vmin * norm.vmax))
                        or (not norm and z > np.nanmean(Z)) else "#111")
        Xc, Yc = np.meshgrid(range(len(QPS_COMMON)), range(len(BUDGETS)))
        if contours and col == "decode_queue_ms":
            cs = ax.contour(Xc, Yc, Z, levels=levels, colors="#b3261e", linewidths=2.2)
            ax.clabel(cs, fmt=lambda v: f"{v:g} s", fontsize=10, inline=True, inline_spacing=6,
                      colors="#b3261e")
            for txt in cs.labelTexts:
                txt.set_bbox(dict(fc="white", ec="none", pad=1.5))
                txt.set_fontweight("bold")
        ax.set_xticks(range(len(QPS_COMMON)))
        ax.set_xticklabels([f"{q:g}" for q in QPS_COMMON])
        ax.set_yticks(range(len(BUDGETS)))
        ax.set_yticklabels([f"{b} GiB\n{SLOTS[b]}" for b in BUDGETS], fontsize=9)
        ax.set_xlabel("request rate (req/s)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label=cb, fraction=0.046, pad=0.03)
    fig.suptitle(f"Grid ({WHO}): decode VRAM budget × request rate   "
                 "(* 24 GiB holds 9.95 requests; a 10th is admitted on its 392 prompt blocks)"
                 + ("   — red: iso-wait contours (s), interpolated between cells" if contours else ""), y=1.02)
    fig.savefig(FIGS / ("kv_grid_heat_contours.png" if contours else "kv_grid_heat.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ M/G/c
def fig_mgc():
    rows = list(csv.DictReader(open(REPO / "output" / "kv_vram" / "mgc.csv")))
    by = {}
    for r in rows:
        if r["method"] != METHOD or r["name"].endswith("_r2") and (r["name"][:-3] in {x["name"] for x in rows}):
            continue
        by.setdefault(float(r["budget_gib"]), []).append(r)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, b in zip(axes, BUDGETS):
        rs = sorted(by.get(b, []), key=lambda r: float(r["qps"]))
        q = [float(r["qps"]) for r in rs]
        c = int(rs[0]["c"])
        meas = [float(r["wait_meas_ms"]) / 1000 for r in rs]
        rep = [float(r["wait_replay_ms"]) / 1000 for r in rs]
        rep1 = [float(r["wait_replay_c1_ms"]) / 1000 for r in rs]
        ll = [float(r["wait_ll_ms"]) / 1000 if r["wait_ll_ms"] not in ("", "None") else np.nan for r in rs]
        ceff = c + 1 if b == 24 else c
        ax.plot(q, meas, "o-", color="#111", lw=3.2, ms=8, label="measured", zorder=4)
        ax.plot(q, rep1 if b == 24 else rep, "s--", color=C_B[b], lw=1.4, ms=4.5, zorder=6,
                label="FCFS replay with the slots the engine ran with (10 / 12 / 15) — coincides")
        coff = c if b == 24 else c + 1
        ax.plot(q, rep if b == 24 else rep1, "^:", color="#9a9a95", lw=1.3, ms=5,
                label="FCFS replay with one slot off (9 / 13 / 16)")
        ax.text(0.97, 0.06, f"grey: c = {coff}", transform=ax.transAxes, ha="right", fontsize=10,
                color="#6b6b66")
        ax.plot(q, ll, "d-.", color="#8a6d00", lw=1.2, ms=5, label="Lee–Longton M/G/c")
        ax.set_yscale("symlog", linthresh=0.05)
        ax.set_xlabel("request rate (req/s)")
        ax.set_title(f"{b} GiB: grant {c if b != 24 else 9.95:g} slots → c = {ceff}")
        ax.axhline(0.05, color="#b3261e", lw=0.8, ls=":")
        ax.grid(color="#ececea", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("decode queue wait, mean (s, symlog)")
    axes[0].text(1.02, 0.055, "50 ms capacity line", color="#b3261e", fontsize=8.5, va="bottom")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.suptitle("Measured decode queue wait vs queueing models (service time = each request's "
                 "measured residence, Poisson-like arrivals)", y=1.2)
    fig.savefig(FIGS / "kv_mgc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ scatter
def fig_scatter():
    """x = k = running requests that must finish before this one is admitted: requests
    queued ahead of it (the trace's waiting count includes the request itself) minus the
    free admissions (free blocks // 392), floored at 0. Prediction: k x (residence / c)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    rng = np.random.default_rng(0)
    C = {24: 10, 26: 12, 28: 15}
    for ax, b in zip(axes, BUDGETS):
        rates = (2.0, 2.5) if METHOD == "base" else (2.5, 3.0)   # where waiting starts
        for q, mk, al, ls in ((rates[0], "o", 0.85, "--"), (rates[1], "^", 0.55, ":")):
            p = TL / f"kv_{METHOD}_b{b:g}_q{q:g}_tr_r2_req.csv"
            if not p.exists():
                p = TL / f"kv_{METHOD}_b{b:g}_q{q:g}_tr_req.csv"
            if not p.exists():
                continue
            rq = [r for r in csv.DictReader(open(p)) if r.get("wait") and r.get("resident")]
            k = np.array([max(0, int(float(r["wait_at_arr"])) - int(r["free_at_arr"]) // 392) for r in rq])
            w = np.array([float(r["wait"]) for r in rq])
            gap = np.mean([float(r["resident"]) for r in rq]) / C[b]
            print(f"  {p.name}: k=0 {int((k == 0).sum())} (max wait {w[k == 0].max() * 1e3:.1f} ms), "
                  f"k>=1 {int((k >= 1).sum())} (min wait {w[k >= 1].min() * 1e3 if (k >= 1).any() else 0:.0f} ms)")
            ax.scatter(k + rng.uniform(-0.18, 0.18, len(k)), w, s=16, marker=mk, color=C_B[b], alpha=al, lw=0,
                       label=f"{q:g} req/s")
            kk = np.arange(0, max(k.max(), 1) + 1)
            ax.plot(kk, kk * gap, ls, color="#444", lw=1.2,
                    label=f"k × residence ÷ c ({gap:.2f} s)")
        ax.set_xlabel("k = requests that must finish first")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_title(f"{b} GiB ({SLOTS[b]})")
        ax.grid(color="#ececea", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper left", fontsize=9)
    axes[0].set_ylabel("this request's decode queue wait (s)")
    fig.suptitle("A request's wait = the slots that must free before its turn × the time between frees "
                 "(each dot = one request)", y=1.02)
    fig.savefig(FIGS / "kv_wait_vs_free.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_3d()
    fig_heat()
    fig_mgc()
    fig_scatter()
    print("-> kv_grid_3d.png kv_grid_heat.png kv_mgc.png kv_wait_vs_free.png")
