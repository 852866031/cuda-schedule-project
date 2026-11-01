#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_csv(path: Path):
    data = defaultdict(list)
    with path.open("r", newline="") as f:
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


def plot_metric(ax, data, key, title, ylabel):
    legend_handles = []
    legend_labels = []

    for gid in sorted(data.keys()):
        rows = data[gid]
        xs = [r["time_s"] for r in rows]
        ys = [r[key] for r in rows]

        if not xs:
            continue

        line, = ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"GPU {gid}")
        legend_handles.append(line)
        legend_labels.append(f"GPU {gid}")

        avg_val = sum(ys) / len(ys)
        ax.axhline(avg_val, linestyle="--", linewidth=1.1, alpha=0.7)

        text_y = 0.95 if gid == 0 else 0.87
        ax.text(
            0.03,
            text_y,
            f"GPU {gid} avg = {avg_val:.4f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.2", alpha=0.15),
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, alpha=0.3)

    return legend_handles, legend_labels


def main():
    this_dir = Path(__file__).resolve().parent
    out_dir = this_dir / "outputs"
    csv_path = out_dir / "dcgm_metrics.csv"
    plot_path = out_dir / "dcgm_metrics_subplots.png"

    data = load_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharex=True)

    handles, labels = plot_metric(
        axes[0], data,
        key="sm_active",
        title="DCGM SM_ACTIVE Over Time",
        ylabel="SM_ACTIVE",
    )

    plot_metric(
        axes[1], data,
        key="sm_occupancy",
        title="DCGM SM_OCCUPANCY Over Time",
        ylabel="SM_OCCUPANCY",
    )

    axes[1].set_xlabel("Time (s)")

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.98),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(f"[OK] Saved plot to {plot_path}")


if __name__ == "__main__":
    main()