#!/usr/bin/env python
"""Per-second bins, per-phase tables and figures for one evaluation arm (design §3.7).
  analyze_coloc.py --arm adjust --dir results --schedule "0:1,90:2,..."
Inputs: <dir>/<arm>_{load.jsonl,sampler.csv,controller.csv,ft.csv,xfer_inf.csv,xfer_ft.csv,t0.txt}
Outputs: <dir>/<arm>_phases.csv, <arm>_phases.md, <arm>_bins.csv, <arm>_timeline.png, <arm>_xfer.png
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True); ap.add_argument("--dir", default="results")
ap.add_argument("--schedule", required=True); ap.add_argument("--skip", type=float, default=10.0)
ap.add_argument("--engine-fixed-gib", type=float, default=17.7); ap.add_argument("--total-gib", type=float, default=31.3536)
args = ap.parse_args()
GiB = 1024 ** 3; P = lambda s: os.path.join(args.dir, f"{args.arm}_{s}")
t0 = float(open(P("t0.txt")).read())
sched = sorted((float(a), float(b)) for a, b in (x.split(":") for x in args.schedule.split(",")))
reqs = [json.loads(l) for l in open(P("load.jsonl")) if not json.loads(l)["rid"].startswith("w")]
samp = pd.read_csv(P("sampler.csv")); samp["rel"] = samp.t - t0
ctl = pd.read_csv(P("controller.csv")) if os.path.exists(P("controller.csv")) else None
if ctl is not None: ctl["rel"] = ctl.t - t0
ft = pd.read_csv(P("ft.csv")) if os.path.exists(P("ft.csv")) and os.path.getsize(P("ft.csv")) > 50 else None
if ft is not None: ft["rel"] = ft.ts - t0
xi = pd.read_csv(P("xfer_inf.csv")) if os.path.exists(P("xfer_inf.csv")) else None
xf = pd.read_csv(P("xfer_ft.csv")) if os.path.exists(P("xfer_ft.csv")) else None
num = lambda df, c: pd.to_numeric(df[c], errors="coerce") if df is not None and c in df else None
end = max(r["t_done"] for r in reqs) - t0
def rps_at(rel):
    r = sched[0][1]
    for t, v in sched:
        if rel >= t: r = v
    return r

# ---- per-request rows
rows = []
for r in reqs:
    ok = bool(r["status"] == 200 and not r["error"] and r["tok_t"])
    rows.append(dict(t_send=r["t_send"] - t0, t_done=r["t_done"] - t0, ok=ok,
                     ttft=(r["tok_t"][0] - r["t_send"]) if ok else np.nan,
                     tpot=((r["t_done"] - r["tok_t"][0]) / max(1, r["n_out"] - 1)) if ok else np.nan, n_out=r["n_out"]))
R = pd.DataFrame(rows)

# ---- phases
phases = []
bounds = [t for t, _ in sched] + [end]
for (ta, rps), tb in zip(sched, bounds[1:]):
    a, b = ta + args.skip, tb
    w = R[(R.t_send >= a) & (R.t_send < b)]; okw = w[w.ok]
    done = R[(R.t_done >= a) & (R.t_done < b) & R.ok]
    sw = samp[(samp.rel >= a) & (samp.rel < b)]
    def delta(c):
        v = num(sw, c); v = v.dropna() if v is not None else v
        return (v.iloc[-1] - v.iloc[0]) if v is not None and len(v) > 1 else np.nan
    qs, qc = delta("queue_sum"), delta("queue_count")
    ph = dict(t_from=ta, t_to=tb, rps=rps, n=len(w), ok=int(w.ok.sum()), failed=int((~w.ok).sum()),
              ttft50=okw.ttft.median() * 1e3, ttft95=okw.ttft.quantile(0.95) * 1e3, tpot50=okw.tpot.median() * 1e3,
              tpot95=okw.tpot.quantile(0.95) * 1e3, tok_s_done=done.n_out.sum() / max(1e-9, b - a),
              offered_tok_s=rps * 128, queue_ms=(qs / qc * 1e3) if qc and qc == qc and qc > 0 else np.nan,
              preempt=delta("preemptions"), nvml_gib=num(sw, "nvml_used").median() / GiB,
              kvc_used_gib=num(sw, "kvc_used").median() / GiB if num(sw, "kvc_used") is not None else np.nan,
              running=num(sw, "running").mean(), waiting_max=num(sw, "waiting").max())
    if ctl is not None:
        cw = ctl[(ctl.rel >= a) & (ctl.rel < b)]
        ph.update(limit_gib=num(cw, "limit_gib").median(), need_gib=num(cw, "need_gib").median(), ft_budget_gib=num(cw, "ft_budget_gib").median())
    if ft is not None:
        fw = ft[(ft.rel >= a) & (ft.rel < b)]; fok = fw[num(fw, "tok_s").notna()]
        ph.update(ft_steps=len(fok), ft_tok_s=(num(fok, "tok_s") * num(fok, "dt_ms") / 1e3).sum() / max(1e-9, b - a),
                  ft_tier_mode=fw.tier.mode().iloc[0] if len(fw) else np.nan, ft_cfg_mode=fw.cfg.mode().iloc[0] if len(fw) else np.nan,
                  ft_reserved_gib=num(fw, "reserved_gib").median(), ft_gated_frac=(num(fok, "gated_ms").sum() / max(1e-9, num(fok, "dt_ms").sum())))
    phases.append(ph)
ph = pd.DataFrame(phases); ph.to_csv(P("phases.csv"), index=False)
cols = ["rps", "n", "failed", "ttft50", "ttft95", "tpot50", "tpot95", "queue_ms", "tok_s_done", "offered_tok_s", "preempt", "kvc_used_gib"] + \
       ([c for c in ("limit_gib", "ft_budget_gib") if c in ph] ) + ([c for c in ("ft_tier_mode", "ft_cfg_mode", "ft_reserved_gib", "ft_tok_s", "ft_gated_frac") if c in ph])
md = [f"## {args.arm}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
for _, r in ph.iterrows():
    md.append("| " + " | ".join(("" if pd.isna(r[c]) else (f"{r[c]:.0f}" if c in ("n","failed","ttft50","ttft95","queue_ms","tok_s_done","offered_tok_s","preempt","ft_tok_s") else f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]))) for c in cols) + " |")
open(P("phases.md"), "w").write("\n".join(md) + "\n"); print("\n".join(md))

# ---- timeline figure (the view as a timeline)
fig, ax = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
tt = np.arange(0, end, 1.0)
ax[0].step(tt, [rps_at(t) for t in tt], where="post", color="k", label="offered RPS")
if ctl is not None: ax[0].plot(ctl.rel, num(ctl, "lambda_hat"), color="C1", alpha=0.7, label="λ̂ (5-s)")
ax[0].set_ylabel("req/s"); ax[0].legend(fontsize=8, loc="upper left")
# memory map
fixed = args.engine_fixed_gib
kv_used = num(samp, "kvc_used") / GiB
ax[1].fill_between(samp.rel, 0, fixed, color="#cfe3cf", label="engine fixed (weights+runtime)")
ax[1].fill_between(samp.rel, fixed, fixed + kv_used, color="#2a9d8f", label="in-flight KV (kvcached used)")
if ctl is not None:
    lim = np.interp(samp.rel, ctl.rel, num(ctl, "limit_gib"))
    ax[1].plot(samp.rel, fixed + lim, "r--", lw=1, label="KV limit")
    bud = np.interp(samp.rel, ctl.rel, num(ctl, "ft_budget_gib"))
if ft is not None:
    res = np.interp(samp.rel, ft.rel, num(ft, "reserved_gib").ffill().fillna(0))
    top = fixed + (lim if ctl is not None else kv_used)
    ax[1].fill_between(samp.rel, args.total_gib - res, args.total_gib, color="#e9c46a", label="tenant (reserved)")
ax[1].plot(samp.rel, num(samp, "nvml_used") / GiB, color="k", lw=0.8, label="NVML used (whole GPU)")
ax[1].set_ylim(0, args.total_gib + 0.5); ax[1].set_ylabel("GiB"); ax[1].legend(fontsize=7, loc="center left")
# decode metrics
ax[2].plot(R.t_send, R.tpot * 1e3, ".", ms=3, label="TPOT per request"); ax[2].set_ylabel("TPOT ms"); ax[2].set_yscale("log"); ax[2].legend(fontsize=8)
q = num(samp, "queue_sum"); qc = num(samp, "queue_count")
ax[3].plot(samp.rel, ((q.diff() / qc.diff()) * 1e3).rolling(20, min_periods=1).mean(), label="scheduler queue wait (ms, rolling)")
ax[3].set_yscale("log"); ax[3].set_ylabel("queue wait ms"); ax[3].legend(fontsize=8, loc="upper left")
a3 = ax[3].twinx(); a3.step(samp.rel, num(samp, "waiting"), where="post", color="C1", lw=0.8, label="# waiting"); a3.set_ylabel("# waiting"); a3.legend(fontsize=8, loc="upper right")
if ft is not None:
    ax[4].plot(ft.rel, num(ft, "tok_s"), ".", ms=3, label="FT tok/s per step")
    a2 = ax[4].twinx(); a2.step(ft.rel, num(ft, "cfg"), where="post", color="C3", lw=1, label="FT config (0=full … 7=pause)"); a2.set_ylim(-0.5, 7.5); a2.set_ylabel("cfg")
    ax[4].set_ylabel("FT tok/s"); ax[4].legend(fontsize=8, loc="upper left"); a2.legend(fontsize=8, loc="upper right")
else:
    ax[4].text(0.5, 0.5, "no tenant in this arm", transform=ax[4].transAxes, ha="center")
for a_ in ax:
    for t, _ in sched[1:]: a_.axvline(t, color="k", ls=":", lw=0.8)
ax[4].set_xlabel("time since end of warm-up (s)"); ax[4].set_xlim(-45, end + 5); fig.suptitle(f"{args.arm}"); fig.tight_layout(); fig.savefig(P("timeline.png"), dpi=120)

# ---- transfer figure
fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
def bars(ax_, df, color, label):
    if df is None or not len(df): return
    d = df.copy(); d["a"] = d.ts_start - t0; d["b"] = d.ts_end - t0
    d = d[(d.b > 0) & (d.a < end)]
    ax_.barh([label] * len(d), d.b - d.a, left=d.a, height=0.6, color=color, alpha=0.6)
    return d
bars(ax[0], xi, "C0", "inf offload (h2d/d2h)"); bars(ax[0], xf, "C3", "ft copies"); ax[0].set_ylabel("transfers")
for df, c, lab in ((xi, "C0", "inf bytes/s"), (xf, "C3", "ft bytes/s")):
    if df is None or not len(df): continue
    d = df.copy(); d["b"] = d.ts_end - t0
    binned = d.groupby(np.floor(d.b)).bytes.sum() / 2**30
    ax[1].bar(binned.index, binned.values, width=1.0, color=c, alpha=0.6, label=lab)
ax[1].set_ylabel("GiB/s (by completion second)"); ax[1].legend(fontsize=8)
if "pcie_tx_kbs" in samp:
    ax[2].plot(samp.rel, num(samp, "pcie_rx_kbs") / 2**20, lw=0.6, label="NVML PCIe RX (host→GPU) GiB/s")
    ax[2].plot(samp.rel, num(samp, "pcie_tx_kbs") / 2**20, lw=0.6, label="NVML PCIe TX (GPU→host) GiB/s"); ax[2].legend(fontsize=8)
ax[2].set_xlabel("time since end of warm-up (s)"); ax[2].set_xlim(-45, end + 5); fig.suptitle(f"{args.arm}: host<->GPU transfers"); fig.tight_layout(); fig.savefig(P("xfer.png"), dpi=120)
# contention: inf load duration vs concurrent ft bytes
if xi is not None and len(xi):
    d = xi.copy(); d["gpu_ms"] = pd.to_numeric(d.gpu_ms, errors="coerce")
    loads = d[d.dir == "h2d"]
    if xf is not None and len(xf):
        conc = []
        for _, r in loads.iterrows():
            o = xf[(xf.ts_start < r.ts_end) & (xf.ts_end > r.ts_start)]
            conc.append(o.bytes.sum() / 2**20)
        loads = loads.assign(ft_concurrent_mib=conc)
        lo, hi = loads[loads.ft_concurrent_mib == 0], loads[loads.ft_concurrent_mib > 0]
        print(f"inf loads: {len(loads)}; median gpu_ms without concurrent ft copies {lo.gpu_ms.median():.1f} ms (n={len(lo)}), "
              f"with {hi.gpu_ms.median():.1f} ms (n={len(hi)}); median load {loads.bytes.median()/2**20:.0f} MiB")
    else:
        print(f"inf loads: {len(loads)}; median gpu_ms {loads.gpu_ms.median():.1f}, median {loads.bytes.median()/2**20:.0f} MiB")
print("figures:", P("timeline.png"), P("xfer.png"))
