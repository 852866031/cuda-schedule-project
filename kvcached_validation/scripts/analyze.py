#!/usr/bin/env python
"""Join a sampler CSV, a loadgen JSONL and (optionally) a resize JSONL for one run.
Prints a Markdown summary and writes a figure (memory trace + inter-token latency + running
requests over time, with the resize instants marked).

  analyze.py --name expA --sampler results/expA_sampler.csv --load results/expA_load.jsonl \
             --resize results/expA_resize.jsonl --figdir results

Memory release/acquire latency: for each resize command, the time from t_cmd until the NVML
used-memory series first comes within 2% (of the step) of its post-step plateau, where the
plateau is the median over the window [t_cmd+25 s, t_cmd+35 s] (or until the next command).
Inter-token latency (ITL): per request, differences of consecutive token timestamps; the time
series takes the median ITL of all tokens emitted in each 1-s bin.
"""
import argparse, json, math, statistics as st
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--sampler", required=True)
ap.add_argument("--load", required=True)
ap.add_argument("--resize", default=None)
ap.add_argument("--figdir", default="results")
ap.add_argument("--plateau-from", type=float, default=25.0)
ap.add_argument("--plateau-to", type=float, default=35.0)
args = ap.parse_args()

s = pd.read_csv(args.sampler)
reqs = [json.loads(l) for l in open(args.load)]
rs = [json.loads(l) for l in open(args.resize)] if args.resize else []
t0 = min(s.t.min(), min(r["t_send"] for r in reqs))
GiB = 1024**3

# --- request-level stats
ok = [r for r in reqs if r["status"] == 200 and not r["error"]]
bad = [r for r in reqs if r not in ok]
n_short = sum(1 for r in ok if r.get("finish_reason") == "length" and r["n_out"] < 1) 
ttft = [r["tok_t"][0] - r["t_send"] for r in ok if r["tok_t"]]
e2e = [r["t_done"] - r["t_send"] for r in ok]
itl_pts = []   # (t, itl)
for r in ok:
    tt = r["tok_t"]; tn = r.get("tok_n") or [1] * len(tt)
    for i, (a, b) in enumerate(zip(tt, tt[1:])):
        n = tn[i + 1]            # tokens delivered by the later chunk
        itl_pts.extend([(b - t0, (b - a) / n)] * n)   # one point per token, per-step time
itl = pd.DataFrame(itl_pts, columns=["t", "itl"])
itl["bin"] = np.floor(itl.t)
itl_ts = itl.groupby("bin").itl.median()

def pct(x, p): return float(np.percentile(x, p)) if len(x) else float("nan")
lines = [f"## {args.name}", "",
         f"requests: {len(reqs)} total, {len(ok)} ok, {len(bad)} failed/errored; "
         f"output tokens: {sum(r['n_out'] for r in ok)}; SSE chunks: {sum(r.get('n_chunks', r['n_out']) for r in ok)} "
         f"(ITL below is per token = chunk gap / tokens in chunk)",
         f"TTFT s: p50 {pct(ttft,50):.3f}  p90 {pct(ttft,90):.3f}  p99 {pct(ttft,99):.3f}",
         f"E2E  s: p50 {pct(e2e,50):.3f}  p90 {pct(e2e,90):.3f}  max {max(e2e) if e2e else float('nan'):.3f}",
         f"ITL  s: p50 {pct(itl.itl,50):.4f}  p90 {pct(itl.itl,90):.4f}  p99 {pct(itl.itl,99):.4f}  max {itl.itl.max() if len(itl) else float('nan'):.3f}",
        ]
stalls = [(b - t0, b - a, r["rid"]) for r in ok for a, b in zip(r["tok_t"], r["tok_t"][1:]) if b - a > 1.0]
lines.append(f"token-stream stalls > 1 s: {len(stalls)} in {len(set(x[2] for x in stalls))} requests"
             + (f"; longest {max(x[1] for x in stalls):.2f} s at t={[x[0] for x in stalls if x[1]==max(y[1] for y in stalls)][0]:.1f} s" if stalls else ""))
if bad:
    lines.append("failed examples: " + "; ".join(f"{r['rid']}: {r['status']} {str(r['error'])[:120]}" for r in bad[:3]))

# --- preemptions from vLLM metrics
pre = pd.to_numeric(s.preemptions, errors="coerce").dropna()
if len(pre): lines.append(f"vLLM preemptions counter: {pre.iloc[0]:.0f} → {pre.iloc[-1]:.0f} (Δ {pre.iloc[-1]-pre.iloc[0]:.0f})")

# --- resize latencies
s["rel"] = s.t - t0
kl_num = pd.to_numeric(s.kvc_limit, errors="coerce")
for i, r in enumerate(rs):
    # Reference instant: when the new limit became visible in kvcached's shared segment
    # (first sampler row after the command whose kvc_limit differs), else kvctl's return time.
    tc = r["t_done"] - t0
    seg = s[(s.t >= r["t_cmd"]) & (s.t <= r["t_done"] + 1.0)]
    base = kl_num[s.t < r["t_cmd"]].dropna()
    if len(seg) and len(base):
        chg = seg[kl_num.loc[seg.index] != base.iloc[-1]]
        if len(chg): tc = chg.rel.iloc[0] if "rel" in chg else chg.t.iloc[0] - t0
    r["t_ref"] = tc + t0
    nxt = (rs[i+1]["t_cmd"] - t0) if i + 1 < len(rs) else s.rel.max()
    before = s[(s.rel > tc - 2) & (s.rel < tc)].nvml_used.median()
    pf, pt = min(tc + args.plateau_from, nxt - 1), min(tc + args.plateau_to, nxt)
    plateau = s[(s.rel >= pf) & (s.rel <= pt)].nvml_used.median()
    step = plateau - before
    lat = float("nan"); t_first = float("nan")
    if abs(step) > 64 * 1024**2:
        tol = 0.02 * abs(step)
        after = s[(s.rel >= tc) & (s.rel <= nxt)]
        hit = after[(after.nvml_used - plateau).abs() <= tol]
        if len(hit): lat = hit.rel.iloc[0] - tc
        moved = after[(after.nvml_used - before).abs() > 0.05 * abs(step)]
        if len(moved): t_first = moved.rel.iloc[0] - tc
    # time until kvcached's mapped bytes first fit under the new limit (shrink only)
    kvu = pd.to_numeric(s.kvc_used, errors="coerce"); kvl = pd.to_numeric(s.kvc_limit, errors="coerce")
    fit = s[(s.rel >= tc) & (s.rel <= nxt) & (kvu <= kvl)]
    t_fit = (fit.rel.iloc[0] - tc) if len(fit) else float("nan")
    kvc_before = s[(s.rel > tc - 2) & (s.rel < tc)].kvc_used.median()
    kvc_after = s[(s.rel >= pf) & (s.rel <= pt)].kvc_used.median()
    run_at = pd.to_numeric(s[(s.rel > tc - 1) & (s.rel < tc)].running, errors="coerce").median()
    lines.append(f"resize t={tc:.1f}s → {r['size']} (kvctl rc={r['rc']}, kvctl took {r['t_done']-r['t_cmd']:.2f}s, limit visible {tc-(r['t_cmd']-t0):.2f}s after invocation): "
                 f"NVML used {before/GiB:.2f} → {plateau/GiB:.2f} GiB; first movement after {t_first:.2f}s, "
                 f"within 2% of plateau after {lat:.2f}s; kvcached used {kvc_before/GiB if kvc_before==kvc_before else float('nan'):.2f} → "
                 f"{kvc_after/GiB if kvc_after==kvc_after else float('nan'):.2f} GiB (fits under the limit after {t_fit:.2f}s); running reqs at cmd: {run_at}")
    # ITL before/after
    b = itl[(itl.t > tc - 15) & (itl.t < tc)].itl; a = itl[(itl.t > tc + 2) & (itl.t < tc + 17)].itl
    lines.append(f"    ITL median 15 s before: {b.median() if len(b) else float('nan'):.4f} s; 2–17 s after: {a.median() if len(a) else float('nan'):.4f} s; "
                 f"during first 2 s after: p50 {pct(itl[(itl.t>=tc)&(itl.t<tc+2)].itl,50):.4f} max {itl[(itl.t>=tc)&(itl.t<tc+2)].itl.max() if len(itl[(itl.t>=tc)&(itl.t<tc+2)]) else float('nan'):.3f}")

print("\n".join(lines))
with open(f"{args.figdir}/{args.name}_summary.md", "w") as f: f.write("\n".join(lines) + "\n")

# --- figure
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
ax[0].plot(s.rel, s.nvml_used / GiB, label="GPU memory in use (NVML)", color="C0")
kv = pd.to_numeric(s.kvc_used, errors="coerce"); kl = pd.to_numeric(s.kvc_limit, errors="coerce")
if kv.notna().any():
    ax[0].plot(s.rel, kv / GiB, label="kvcached used", color="C1")
    ax[0].plot(s.rel, kl / GiB, label="kvcached limit", color="C3", ls="--")
ax[0].set_ylabel("GiB"); ax[0].legend(loc="upper right", fontsize=8)
ax[1].plot(itl_ts.index, itl_ts.values * 1000, ".-", ms=3, label="median ITL per 1-s bin")
ax[1].set_ylabel("ms"); ax[1].legend(loc="upper right", fontsize=8)
ax[2].plot(s.rel, pd.to_numeric(s.running, errors="coerce"), label="running", color="C2")
ax[2].plot(s.rel, pd.to_numeric(s.waiting, errors="coerce"), label="waiting", color="C4")
ax[2].set_ylabel("requests"); ax[2].set_xlabel("time since start (s)"); ax[2].legend(loc="upper right", fontsize=8)
for r in rs:
    for a in ax: a.axvline(r["t_ref"] - t0, color="k", ls=":", lw=1)
    ax[0].text(r["t_ref"] - t0, ax[0].get_ylim()[1] * 0.95, f"limit {r['size']}", fontsize=8, rotation=90, va="top")
fig.suptitle(args.name); fig.tight_layout()
fig.savefig(f"{args.figdir}/{args.name}.png", dpi=120)
print("figure:", f"{args.figdir}/{args.name}.png")
