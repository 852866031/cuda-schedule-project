#!/usr/bin/env python
"""Per-limit-window metrics for a live shrink sweep. Window k = from the moment limit k became
visible in kvcached's segment (resize.jsonl / sampler) to the next change; window 0 = from t0
(end of warm-up) to the first change. Reports TTFT, per-token ITL, throughput, preemptions,
GPU prefix-cache hit rate and external (DRAM offload) hit rate, memory, stalls."""
import argparse, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--sampler", required=True)
ap.add_argument("--load", required=True); ap.add_argument("--resize", required=True)
ap.add_argument("--t0", type=float, required=True); ap.add_argument("--step", type=float, required=True)
ap.add_argument("--figdir", default="results"); ap.add_argument("--settle", type=float, default=5.0,
    help="seconds after a limit change excluded from the window's request statistics")
args = ap.parse_args()
GiB = 1024**3
s = pd.read_csv(args.sampler); s["rel"] = s.t - args.t0
reqs = [json.loads(l) for l in open(args.load)]
reqs = [r for r in reqs if not r["rid"].startswith("w")]
rs = [json.loads(l) for l in open(args.resize)]
num = lambda c: pd.to_numeric(s[c], errors="coerce")
kl = num("kvc_limit")
# window boundaries: baseline from 0; then each limit's first visible instant
bounds = [("start", 0.0)]
for r in rs:
    seg = s[(s.t >= r["t_cmd"]) & (s.t <= r["t_done"] + 1.5)]
    base = kl[s.t < r["t_cmd"]].dropna()
    tv = r["t_done"] - args.t0
    if len(seg) and len(base):
        chg = seg[kl.loc[seg.index] != base.iloc[-1]]
        if len(chg): tv = chg.rel.iloc[0]
    bounds.append((r["size"], tv))
bounds.append(("end", bounds[-1][1] + args.step))
rows = []
for k in range(len(bounds) - 1):
    label, a = bounds[k]; b = bounds[k + 1][1]
    win = [r for r in reqs if a + args.settle <= r["t_send"] - args.t0 < b]
    ok = [r for r in win if r["status"] == 200 and not r["error"] and r["tok_t"]]
    ttft = np.array([r["tok_t"][0] - r["t_send"] for r in ok])
    itl = []
    for r in ok:
        tt, tn = r["tok_t"], r.get("tok_n") or [1] * len(r["tok_t"])
        for i, (x, y) in enumerate(zip(tt, tt[1:])): itl.extend([(y - x) / tn[i + 1]] * tn[i + 1])
    itl = np.array(itl)
    toks = sum(r["n_out"] for r in ok); span = max(1e-9, b - a - args.settle)
    sw = s[(s.rel >= a) & (s.rel < b)]
    def delta(c):
        v = num(c).loc[sw.index].dropna(); return (v.iloc[-1] - v.iloc[0]) if len(v) > 1 else float("nan")
    pcq, pch, exq, exh = delta("pc_queries"), delta("pc_hits"), delta("ext_queries"), delta("ext_hits")
    stalls = sum(1 for r in ok for x, y in zip(r["tok_t"], r["tok_t"][1:]) if y - x > 1.0)
    rows.append(dict(limit=label, t_from=a, t_to=b, n=len(win), ok=len(ok), failed=len(win) - len(ok),
        ttft50=np.median(ttft) * 1e3 if len(ttft) else np.nan, ttft95=np.percentile(ttft, 95) * 1e3 if len(ttft) else np.nan,
        itl50=np.median(itl) * 1e3 if len(itl) else np.nan, itl95=np.percentile(itl, 95) * 1e3 if len(itl) else np.nan,
        tok_s=toks / span, preempt=delta("preemptions"),
        gpu_hit=pch / pcq if pcq else np.nan, dram_hit=exh / exq if exq else np.nan,
        kvc_used=num("kvc_used").loc[sw.index].median() / GiB, nvml=num("nvml_used").loc[sw.index].median() / GiB,
        running=num("running").loc[sw.index].median(), waiting=num("waiting").loc[sw.index].max(), stalls=stalls))
df = pd.DataFrame(rows)
hdr = ["limit", "window s", "req ok/fail", "TTFT p50 ms", "TTFT p95 ms", "ITL p50 ms", "ITL p95 ms", "tok/s", "preempt Δ",
       "GPU prefix hit", "DRAM hit", "kvcached used GiB", "NVML GiB", "running med", "waiting max", "stalls>1s"]
lines = [f"## {args.name}", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
for r in rows:
    lines.append(f"| {r['limit']} | {r['t_from']:.0f}–{r['t_to']:.0f} | {r['ok']}/{r['failed']} | {r['ttft50']:.0f} | {r['ttft95']:.0f} | "
                 f"{r['itl50']:.1f} | {r['itl95']:.1f} | {r['tok_s']:.0f} | {r['preempt']:.0f} | "
                 f"{r['gpu_hit']*100 if r['gpu_hit']==r['gpu_hit'] else float('nan'):.0f}% | {r['dram_hit']*100 if r['dram_hit']==r['dram_hit'] else float('nan'):.0f}% | "
                 f"{r['kvc_used']:.2f} | {r['nvml']:.2f} | {r['running']:.0f} | {r['waiting']:.0f} | {r['stalls']} |")
print("\n".join(lines)); open(f"{args.figdir}/{args.name}_summary.md", "w").write("\n".join(lines) + "\n")
df.to_csv(f"{args.figdir}/{args.name}_windows.csv", index=False)
# figure
fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
ax[0].plot(s.rel, num("nvml_used") / GiB, label="GPU memory in use (NVML)")
ax[0].plot(s.rel, num("kvc_used") / GiB, label="kvcached used")
# limit schedule: kvcached main publishes limits through its control segment, so the legacy
# segment field the sampler reads may stay at the start-up value; draw the schedule from the
# kvctl records instead (staircase), starting from the sampler's initial value.
def _bytes(sz):
    u = sz[-1].upper(); n = float(sz[:-1]) if u in "GMK" else float(sz)
    return n * {"G": GiB, "M": 1024**2, "K": 1024}.get(u, 1)
lim_x, lim_y = [s.rel.min()], [kl.dropna().iloc[0] if kl.notna().any() else np.nan]
for lab, x in bounds[1:-1]:
    lim_x += [x, x]; lim_y += [lim_y[-1], _bytes(lab)]
lim_x.append(s.rel.max()); lim_y.append(lim_y[-1])
ax[0].plot(lim_x, np.array(lim_y) / GiB, "--", label="kvcached limit (kvctl schedule)")
ax[0].set_ylabel("GiB"); ax[0].legend(fontsize=8, loc="upper right")
tt = [(r["t_send"] - args.t0, (r["tok_t"][0] - r["t_send"]) * 1e3) for r in reqs if r["tok_t"]]
ax[1].plot([x for x, _ in tt], [y for _, y in tt], ".", ms=3); ax[1].set_ylabel("TTFT ms"); ax[1].set_yscale("log")
ax[2].plot(s.rel, num("running"), label="running"); ax[2].plot(s.rel, num("waiting"), label="waiting"); ax[2].legend(fontsize=8); ax[2].set_ylabel("requests")
q, h, eq, eh = num("pc_queries"), num("pc_hits"), num("ext_queries"), num("ext_hits")
ax[3].plot(s.rel, (h.diff() / q.diff()).rolling(50, min_periods=5).mean(), label="GPU prefix hit (rolling)")
ax[3].plot(s.rel, (eh.diff() / eq.diff()).rolling(50, min_periods=5).mean(), label="DRAM (offload) hit (rolling)")
ax[3].set_ylim(0, 1.05); ax[3].legend(fontsize=8); ax[3].set_ylabel("hit rate"); ax[3].set_xlabel("time since end of warm-up (s)")
for lab, x in bounds[1:-1]:
    for a_ in ax: a_.axvline(x, color="k", ls=":", lw=1)
    ax[0].text(x, ax[0].get_ylim()[1] * 0.95, lab, rotation=90, va="top", fontsize=8)
fig.suptitle(args.name); fig.tight_layout(); fig.savefig(f"{args.figdir}/{args.name}.png", dpi=120)
print("figure:", f"{args.figdir}/{args.name}.png")
