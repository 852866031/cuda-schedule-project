#!/usr/bin/env python
"""The workload trace driven through the router in every scheme.
  plot_workload.py --load results/adjust_notenant_load.jsonl --schedule "0:1,90:2,..." --out report/figures/workload.png
(a) offered arrival rate (the schedule) and the arrivals actually generated, 10-s bins;
(b) how often each of the 32 sessions was asked for (Zipf-1.1 popularity).
"""
import argparse, json, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--load", default="results/adjust_notenant_load.jsonl")
ap.add_argument("--schedule", default="0:1,90:2,180:3,270:1,360:4,420:0.5,480:2,510:3,540:2,570:1,600:1")
ap.add_argument("--out", default="report/figures/workload.png")
args = ap.parse_args()
t0 = float(open(args.load + ".t0").read())
reqs = [json.loads(l) for l in open(args.load)]
main = [r for r in reqs if r["rid"].startswith("r")]
warm = [r for r in reqs if r["rid"].startswith("w")]
t_send = np.array([r["t_send"] - t0 for r in main]); end = t_send.max()
sid = np.array([int(re.search(r"_s(\d+)", r["rid"]).group(1)) for r in main])
sched = sorted((float(a), float(b)) for a, b in (x.split(":") for x in args.schedule.split(",")))
bounds = [t for t, _ in sched] + [end]

fig, ax = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw=dict(width_ratios=[2.2, 1]))
tt = np.arange(0, end + 1, 1.0)
offered = np.array([next(r for t, r in reversed(sched) if ts >= t) for ts in tt])
ax[0].step(tt, offered, where="post", color="k", lw=1.8, label="offered rate (schedule)")
bins = np.arange(0, end + 10, 10); counts, _ = np.histogram(t_send, bins)
ax[0].bar(bins[:-1], counts / 10, width=10, align="edge", color="#1f77b4", alpha=0.45, label="arrivals generated (10-s bins)")
for (ta, r), tb in zip(sched, bounds[1:]):
    if tb - ta >= 25:
        ax[0].text((ta + tb) / 2, r + 0.15, f"{r:g}", ha="center", va="bottom", fontsize=9)
    ax[0].axvline(ta, color="k", ls=":", lw=0.6, alpha=0.5)
ax[0].set_xlim(-5, end + 5); ax[0].set_ylim(0, 6.3)
ax[0].set_xlabel("time since end of warm-up (s)"); ax[0].set_ylabel("requests / s")
ax[0].set_title(f"(a) arrival rate over the {end/60:.0f}-minute run ({len(main)} requests after a {len(warm)}-request warm-up)")
ax[0].legend(loc="upper left", fontsize=9, frameon=False)
ax[0].text(0.99, 0.97, "every request: a 6144-token session prefix + 128 new tokens in, 128 tokens out\n"
           "Poisson arrivals; session picked by Zipf(1.1) over 32 sessions; prefix caching off,\n"
           "prefixes fetched from the host-RAM KV store (≈0.77 GiB of KV per request in flight)",
           transform=ax[0].transAxes, ha="right", va="top", fontsize=8.5,
           bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc"))
ax[0].grid(True, axis="y", alpha=0.3)

n = np.bincount(sid, minlength=32)
ax[1].bar(np.arange(32), n, color="#ff7f0e")
ax[1].set_xlabel("session (rank by popularity)"); ax[1].set_ylabel("requests in the run")
ax[1].set_title("(b) session popularity, Zipf(1.1)")
ax[1].text(0.97, 0.95, f"top session: {n.max()} of {len(main)} ({100*n.max()/len(main):.0f} %)\ntop 8 sessions: {100*np.sort(n)[::-1][:8].sum()/len(main):.0f} %",
           transform=ax[1].transAxes, ha="right", va="top", fontsize=9)
ax[1].grid(True, axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(args.out, dpi=150); print("figure:", args.out)
