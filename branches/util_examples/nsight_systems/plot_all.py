#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "outputs"

NVML_CSV = OUT_DIR / "nvml_metrics.csv"
DCGM_CSV = OUT_DIR / "dcgm_metrics.csv"
PLOT_FILE = OUT_DIR / "combined_nvml_dcgm.png"


def load_nvml():
    data = defaultdict(list)
    with NVML_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gid = int(row["gpu_id"])
            data[gid].append(
                {
                    "time_s": float(row["time_s"]),
                    "gpu_util": float(row["gpu_util"]),
                    "mem_util": float(row["mem_util"]),
                }
            )
    return data


def load_dcgm():
    data = defaultdict(list)
    with DCGM_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gid = int(row["gpu_id"])
            data[gid].append(
                {
                    "time_s": float(row["time_s"]),
                    "sm_active": float(row["sm_active"]),
                    "sm_occupancy": float(row["sm_occupancy"]),
                }
            )
    return data


def dedup_legend(handles, labels):
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    return list(seen.values()), list(seen.keys())


def main():
    nvml = load_nvml()
    dcgm = load_dcgm()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)

    # ---------------- First row: NVML ----------------
    # NVML GPU util
    for gid in sorted(nvml.keys()):
        xs = [r["time_s"] for r in nvml[gid]]
        ys = [r["gpu_util"] for r in nvml[gid]]
        axes[0, 0].plot(xs, ys, linewidth=1.8, label=f"GPU {gid}")
    axes[0, 0].set_title("NVML GPU Utilization")
    axes[0, 0].set_ylabel("GPU util (%)")
    axes[0, 0].grid(True, alpha=0.3)

    # NVML memory util
    for gid in sorted(nvml.keys()):
        xs = [r["time_s"] for r in nvml[gid]]
        ys = [r["mem_util"] for r in nvml[gid]]
        axes[0, 1].plot(xs, ys, linewidth=1.8, label=f"GPU {gid}")
    axes[0, 1].set_title("NVML Memory Utilization")
    axes[0, 1].set_ylabel("Memory util (%)")
    axes[0, 1].grid(True, alpha=0.3)

    # ---------------- Second row: DCGM ----------------
    # DCGM SM_ACTIVE
    for gid in sorted(dcgm.keys()):
        xs = [r["time_s"] for r in dcgm[gid]]
        ys = [r["sm_active"] for r in dcgm[gid]]
        axes[1, 0].plot(xs, ys, linewidth=1.8, label=f"GPU {gid}")
    axes[1, 0].set_title("DCGM SM_ACTIVE")
    axes[1, 0].set_ylabel("SM_ACTIVE")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylim(0, 1.2)
    axes[1, 0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[1, 0].grid(True, alpha=0.3)

    # DCGM SM_OCCUPANCY
    for gid in sorted(dcgm.keys()):
        xs = [r["time_s"] for r in dcgm[gid]]
        ys = [r["sm_occupancy"] for r in dcgm[gid]]
        axes[1, 1].plot(xs, ys, linewidth=1.8, label=f"GPU {gid}")
    axes[1, 1].set_title("DCGM SM_OCCUPANCY")
    axes[1, 1].set_ylabel("SM_OCCUPANCY")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylim(0, 1.2)
    axes[1, 1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[1, 1].grid(True, alpha=0.3)

    # One global legend
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    handles, labels = dedup_legend(handles, labels)

    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(PLOT_FILE, dpi=160)
    plt.close(fig)

    print(f"Saved combined plot to {PLOT_FILE}")


if __name__ == "__main__":
    main()