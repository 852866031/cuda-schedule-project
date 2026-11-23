#!/usr/bin/env python3
"""Storyline figure: decode on vLLM when its decode steps may use only N of the 170 SMs.
Reads coloc_adjust/results_sm/eager_gpu<g>_<N>.json, written by `EAGER=1 coloc_adjust/run_sm_quick.sh`: a plain vLLM engine
(no kvcached, no host-RAM KV tier, no tenant) started with --enforce-eager, one launch per N, its decode steps on a green
context of N SMs; a closed-loop load keeps 1, 4 or 8 requests in flight (64-token prompt, 200 output tokens), 15 s per level.
The time per output token runs from the first to the last token of a request (median over the requests of a level).
Speed = that time with all SMs / with N SMs, each GPU against its own run with all SMs.
Eager mode is needed: with CUDA graphs (the engine's default) the green context does not confine the engine's main kernels
(q_gpu*_*.json, printed below for the record: 8 SMs still gives 73-78 % of the full speed).
  python3 storyline/make_sm_vllm_fig.py   ->  storyline/figures/decode_vs_sms_vllm.png"""
import glob, json, os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, "..", "coloc_adjust", "results_sm")
def load(prefix):
    runs = {}   # gpu -> {sms: {C: median ms or None}}
    for p in sorted(glob.glob(os.path.join(D, prefix + "_gpu*_*.json"))):
        g, n = re.match(prefix + r"_gpu(\d)_(\w+)\.json", os.path.basename(p)).groups()
        runs.setdefault(int(g), {})[170 if n == "full" else int(n)] = {int(c): v["median_ms"] for c, v in json.load(open(p)).items()}
    pts = {}    # sms -> {C: speed %}
    for g, d in sorted(runs.items()):
        for n in sorted(d):
            sp = {c: (d[170][c] / d[n][c] * 100 if d[n][c] else None) for c in d[n]}
            print(f"{prefix:5s} gpu {g} {n:4d} SMs   " + "   ".join(f"{c} in flight: " + (f"{d[n][c]:6.2f} ms {sp[c]:4.0f} %" if d[n][c] else "   no request finished") for c in sorted(d[n])))
            if n != 170 or 170 not in pts: pts[n] = sp
    return pts
graphs = load("q"); pts = load("eager")
sms = sorted(pts); LEVELS = sorted(pts[170])
COL = dict(zip(LEVELS, ["#9ecae1", "#4292c6", "#08306b"]))
fig, ax = plt.subplots(figsize=(9.6, 4.6)); fig.subplots_adjust(top=0.90, bottom=0.14, left=0.085, right=0.73)
for c in LEVELS:
    xs = [n for n in sms if pts[n].get(c)]
    ax.plot(xs, [pts[n][c] for n in xs], "-o", color=COL[c], lw=2, ms=5, markeredgecolor="white", markeredgewidth=0.8,
            label=f"{c} request{'s' if c > 1 else ''} in the batch")
ax.axhline(100, color="#5f6368", lw=0.8, ls="--"); ax.axvline(96, color="#5f6368", lw=1.0, ls="--")
ax.set_xlim(0, 176); ax.set_ylim(0, 120); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_xticks([8, 32, 56, 80, 96, 128, 170])
ax.set_xlabel("SMs decode may use (the card has 170)"); ax.set_ylabel("decode speed, % of the speed with all 170 SMs")
ax.set_title("Decode on vLLM: speed against the number of SMs it may use", loc="left", fontsize=11, fontweight="bold")
ax.grid(True, axis="y", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9.5, title="vLLM engine (eager mode),\ndecode alone on the GPU", title_fontsize=9, alignment="left")
out = os.path.join(H, "figures", "decode_vs_sms_vllm.png"); fig.savefig(out, dpi=140); print(out)
