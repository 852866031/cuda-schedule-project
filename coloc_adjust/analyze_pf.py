#!/usr/bin/env python
"""P3 prefill-tenant metrics: per phase, the tenant's prefill throughput (prompt tokens/s), its
latency (time to the single output token) and failures; adapter state over time. Merges
ft_tok_s (= prompt tok/s) and pf_* columns into <arm>_phases.csv so the cross-scheme figures treat
the prefill tenant like the training tenant, and draws <arm>_pf.png."""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_names import name as arm_name

ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); ap.add_argument("--dir", default="results")
ap.add_argument("--schedule", required=True); ap.add_argument("--skip", type=float, default=10.0)
args = ap.parse_args(); P = lambda s: os.path.join(args.dir, f"{args.arm}_{s}")
t0 = float(open(P("t0.txt")).read())
sched = sorted((float(a), float(b)) for a, b in (x.split(":") for x in args.schedule.split(",")))
reqs = [json.loads(l) for l in open(P("pfload.jsonl"))]
R = pd.DataFrame([dict(t=r["t_send"] - t0, ok=bool(r["status"] == 200 and not r["error"] and r["tok_t"]), ptok=r["prompt_tokens"],
                       lat=((r["tok_t"][0] - r["t_send"]) if r["tok_t"] else np.nan), t_done=r["t_done"] - t0) for r in reqs])
end = R.t_done.max()
pf = pd.read_csv(P("pf.csv")); pf["rel"] = pf.ts - t0
ph = pd.read_csv(P("phases.csv"))
bounds = [t for t, _ in sched] + [end]
for i, ((ta, rps), tb) in enumerate(zip(sched, bounds[1:])):
    a, b = ta + args.skip, tb
    w = R[(R.t >= a) & (R.t < b)]; ok = w[w.ok]
    done = R[(R.t_done >= a) & (R.t_done < b) & R.ok]
    pw = pf[(pf.rel >= a) & (pf.rel < b)]
    if i < len(ph):
        ph.loc[i, "ft_tok_s"] = done.ptok.sum() / max(1e-9, b - a)          # prefill tokens/s delivered
        ph.loc[i, "pf_lat50"] = ok.lat.median() * 1e3; ph.loc[i, "pf_lat95"] = ok.lat.quantile(0.95) * 1e3
        ph.loc[i, "pf_n"] = len(w); ph.loc[i, "pf_failed"] = int((~w.ok).sum())
        ph.loc[i, "pf_limit_gib"] = pw.limit_gib.median() if len(pw) else np.nan
        ph.loc[i, "pf_paused_frac"] = pw.paused.mean() if len(pw) else np.nan; ph.loc[i, "pf_asleep_frac"] = pw.asleep.mean() if len(pw) else np.nan
        ph.loc[i, "ft_reserved_gib"] = (pw.kvc_used_gib.median() if len(pw) else np.nan)
ph.to_csv(P("phases.csv"), index=False)
cols = ["rps", "tpot50", "queue_ms", "ft_tok_s", "pf_lat50", "pf_lat95", "pf_n", "pf_failed", "pf_limit_gib", "pf_paused_frac", "pf_asleep_frac"]
print(ph[[c for c in cols if c in ph]].round(2).to_string(index=False))
# figure: tenant throughput, latency, adapter state
fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1.0, 0.8], hspace=0.1))
tt = np.arange(0, end, 5.0); thr = [R[(R.t_done >= t) & (R.t_done < t + 5) & R.ok].ptok.sum() / 5 / 1e3 for t in tt]
ax[0].bar(tt, thr, width=5, align="edge", color="#e9c46a"); ax[0].set_ylabel("prefill k tok/s (5-s bins)")
ax[1].plot(R[R.ok].t, R[R.ok].lat * 1e3, ".", ms=2.5, color="#ff7f0e", alpha=0.5); ax[1].set_yscale("log"); ax[1].set_ylabel("prefill latency (ms)")
ax[2].step(pf.rel, pf.limit_gib, where="post", color="#c1121f", label="tenant KV limit (GiB)")
ax[2].fill_between(pf.rel, 0, pf.paused * pf.limit_gib.max(), step="post", color="#999999", alpha=0.3, label="paused")
ax[2].fill_between(pf.rel, 0, pf.asleep * pf.limit_gib.max(), step="post", color="#444444", alpha=0.4, label="asleep")
ax[2].set_ylabel("GiB"); ax[2].legend(fontsize=8, loc="upper left", ncol=3, frameon=False); ax[2].set_xlabel("time since end of warm-up (s)"); ax[2].set_xlim(-35, end + 5)
for a_ in ax:
    a_.grid(True, axis="y", alpha=0.25)
    for t, _ in sched[1:]: a_.axvline(t, color="k", ls=":", lw=0.6, alpha=0.6)
fig.suptitle(f"{arm_name(args.arm)}: the prefill tenant", fontsize=12); fig.savefig(P("pf.png"), dpi=130, bbox_inches="tight"); print("figure:", P("pf.png"))
