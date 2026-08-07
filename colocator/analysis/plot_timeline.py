#!/usr/bin/env python
"""Render a run: per-client GPU Gantt from the colocated trace + latency CDFs.

    python colocator/analysis/plot_timeline.py runs/full

Outputs into <run_dir>/:
    timeline_colocated.png — GPU lanes (kernels + memcpys per client), full
        span and a zoom on a busy window
    latency_cdf.png — per-client iteration-latency CDFs across modes
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KERNEL_COLORS = {"sgemm": "#d62728", "elementwise": "#1f77b4",
                 "reduce": "#2ca02c", "sleep": "#17becf",
                 "streamcopy": "#9467bd", "blockcopy": "#bcbd22",
                 "fma32": "#e377c2", "fma64": "#8c564b", "other": "#7f7f7f"}
MEMCPY_COLOR = "#ff7f0e"


def load_trace(run_dir):
    p = os.path.join(run_dir, "colocated", "trace.json")
    if not os.path.exists(p):
        return None
    t = pd.read_json(p)
    return t[t.t_gpu_start_ns.notna()]


def plot_gantt(ax, trace, t0, xlim=None):
    lanes = []
    labels = []
    for client in sorted(trace.client.unique()):
        for what, sel in [("kernels", trace.kind == "cudaLaunchKernel"),
                          ("memcpys", trace.kind.isin(["cudaMemcpy", "cudaMemcpyAsync"]))]:
            g = trace[(trace.client == client) & sel]
            lanes.append(g)
            labels.append(f"client {client} {what}")

    # Minimum drawn width so sub-microsecond ops (tiny D2H result copies,
    # 1 us elementwise kernels) stay visible instead of aliasing to gray.
    span_ms = (trace.t_gpu_end_ns.max() - t0) / 1e6 if xlim is None else xlim[1] - xlim[0]
    min_w = span_ms / 2000
    for y, g in enumerate(lanes):
        if g.empty:
            continue
        spans = [((s - t0) / 1e6, max((e - s) / 1e6, min_w))
                 for s, e in zip(g.t_gpu_start_ns, g.t_gpu_end_ns)]
        colors = [MEMCPY_COLOR if k != "cudaLaunchKernel" else
                  KERNEL_COLORS.get(n, "#7f7f7f")
                  for k, n in zip(g.kind, g["name"])]
        ax.broken_barh(spans, (y + 0.1, 0.8), facecolors=colors, edgecolor="none")

    ax.set_yticks([y + 0.5 for y in range(len(labels))])
    ax.set_yticklabels(labels)
    ax.set_xlabel("time (ms)")
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(axis="x", alpha=0.3)


def make_timeline(run_dir):
    trace = load_trace(run_dir)
    if trace is None:
        print("[plot] no colocated trace; skipping timeline")
        return
    t0 = trace.t_gpu_start_ns.min()
    span_ms = (trace.t_gpu_end_ns.max() - t0) / 1e6

    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    plot_gantt(axes[0], trace, t0)
    axes[0].set_title(f"colocated GPU timeline — full span ({span_ms:.0f} ms)")

    # zoom: 80 ms window starting at the median GPU timestamp (busy region)
    mid = (trace.t_gpu_start_ns.median() - t0) / 1e6
    plot_gantt(axes[1], trace, t0, xlim=(mid, mid + 80))
    axes[1].set_title("zoom (80 ms)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in KERNEL_COLORS.values()]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=MEMCPY_COLOR))
    fig.legend(handles, list(KERNEL_COLORS) + ["memcpy"],
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = os.path.join(run_dir, "timeline_colocated.png")
    fig.savefig(out, dpi=130)
    print(f"[plot] wrote {out}")


def make_cdfs(run_dir):
    modes = [m for m in ("seq", "streams", "colocated")
             if os.path.exists(os.path.join(run_dir, m, "results.json"))]
    if not modes:
        return
    clients = {}
    for mode in modes:
        with open(os.path.join(run_dir, mode, "results.json")) as f:
            res = json.load(f)
        for c in res["clients"]:
            clients.setdefault(c["name"], {})[mode] = np.array(c["lat_s"]) * 1e3

    fig, axes = plt.subplots(1, len(clients), figsize=(6 * len(clients), 4))
    if len(clients) == 1:
        axes = [axes]
    for ax, (name, per_mode) in zip(axes, clients.items()):
        for mode, lat in per_mode.items():
            x = np.sort(lat)
            y = np.arange(1, len(x) + 1) / len(x)
            ax.plot(x, y, label=mode, drawstyle="steps-post")
        ax.set_xscale("log")
        ax.set_xlabel("iteration latency (ms)")
        ax.set_ylabel("CDF")
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out = os.path.join(run_dir, "latency_cdf.png")
    fig.savefig(out, dpi=130)
    print(f"[plot] wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    args = ap.parse_args()
    make_timeline(args.run_dir)
    make_cdfs(args.run_dir)


if __name__ == "__main__":
    main()
