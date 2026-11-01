#!/usr/bin/env python3
import csv
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


GPU_IDS = [0, 1]
FIELD_IDS = ["1002", "1003", "1004", "1005"]
FIELD_NAMES = {
    "1002": "SM_ACTIVE",
    "1003": "SM_OCCUPANCY",
    "1004": "TENSOR_ACTIVE",
    "1005": "DRAM_ACTIVE",
}

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
NVML_DIR = PROJECT_ROOT / "nvml"
OUT_DIR = THIS_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RAW_LOG = OUT_DIR / "dcgm_dmon_raw.log"
CSV_LOG = OUT_DIR / "dcgm_metrics.csv"
PLOT_FILE = OUT_DIR / "dcgm_metrics_subplots.png"


def require_cmd(cmd: str) -> None:
    if shutil.which(cmd) is None:
        raise RuntimeError(f"Required command not found in PATH: {cmd}")


def check_dcgm_host_running() -> None:
    proc = subprocess.run(
        ["systemctl", "is-active", "nvidia-dcgm"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0 and proc.stdout.strip() == "active":
        print("[OK] nvidia-dcgm is active.")
        return

    proc = subprocess.run(
        ["pgrep", "-f", "nv-hostengine"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        print("[OK] nv-hostengine is running.")
        return

    raise RuntimeError(
        "DCGM host is not running.\n"
        "Try: sudo systemctl --now enable nvidia-dcgm"
    )


def compile_workload() -> None:
    print("[INFO] Compiling two_cases ...")
    subprocess.run(
        ["make", "-C", str(NVML_DIR), "compile"],
        check=True,
    )
    print("[OK] Compile finished.")


def start_dmon() -> subprocess.Popen:
    cmd = [
        "dcgmi", "dmon",
        "-e", ",".join(FIELD_IDS),
        "-i", ",".join(map(str, GPU_IDS)),
    ]
    print("[INFO] Starting:", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def start_workload() -> subprocess.Popen:
    cmd = ["make", "-C", str(NVML_DIR), "run"]
    print("[INFO] Starting workload:", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        stdout=None,   # inherit terminal
        stderr=None,
        text=True,
    )


def terminate_process(proc: subprocess.Popen, name: str) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        print(f"[WARN] {name} did not terminate, killing it.")
        proc.kill()


def parse_dmon_stream(dmon_proc: subprocess.Popen, workload_proc: subprocess.Popen):
    """
    Parse dcgmi dmon output while workload is running.
    After the workload exits, keep collecting for a short tail window,
    then stop.
    """
    start_time = time.time()
    data = defaultdict(list)

    workload_done_time = None
    tail_seconds = 1.0

    with RAW_LOG.open("w") as rawf:
        while True:
            # Check whether workload has finished
            if workload_done_time is None and workload_proc.poll() is not None:
                workload_done_time = time.time()

            # After workload finishes, only keep a short tail
            if workload_done_time is not None and (time.time() - workload_done_time) > tail_seconds:
                break

            line = dmon_proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue

            rawf.write(line)
            rawf.flush()
            print(f"[DCGM] {line.rstrip()}")

            parts = line.strip().split()
            if len(parts) < 2 or parts[0] != "GPU":
                continue

            try:
                gpu_id = int(parts[1])
            except ValueError:
                continue

            if gpu_id not in GPU_IDS:
                continue

            numeric_tokens = []
            for token in parts[2:]:
                try:
                    numeric_tokens.append(float(token))
                except ValueError:
                    pass

            if len(numeric_tokens) < len(FIELD_IDS):
                continue

            t = time.time() - start_time
            row = {"time_s": t}
            for i, fid in enumerate(FIELD_IDS):
                row[fid] = numeric_tokens[i]
            data[gpu_id].append(row)

    return data


def drain_workload_output(proc: subprocess.Popen) -> None:
    if proc.stdout is None:
        return
    for line in proc.stdout:
        print(f"[WORKLOAD] {line.rstrip()}")


def save_csv(data) -> None:
    with CSV_LOG.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gpu_id", "time_s"] + FIELD_IDS)
        for gid in sorted(data.keys()):
            for row in data[gid]:
                writer.writerow(
                    [gid, f"{row['time_s']:.6f}"] + [row[fid] for fid in FIELD_IDS]
                )
    print(f"[OK] Saved CSV to {CSV_LOG}")


def plot_subplots(data) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), sharex=True)
    axes = list(axes)

    # Keep one global legend instead of one legend per subplot
    legend_handles = []
    legend_labels = []

    for ax_idx, (ax, fid) in enumerate(zip(axes, FIELD_IDS)):
        for gid in GPU_IDS:
            rows = data.get(gid, [])
            xs = [r["time_s"] for r in rows]
            ys = [r[fid] for r in rows]

            if not xs:
                continue

            line, = ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"GPU {gid}")

            # Record handles only once for global legend
            if ax_idx == 0:
                legend_handles.append(line)
                legend_labels.append(f"GPU {gid}")

            avg_val = sum(ys) / len(ys)

            # Draw average line
            ax.axhline(avg_val, linestyle="--", linewidth=1.2, alpha=0.8)

            # Put average text inside each subplot
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

        ax.set_title(FIELD_NAMES[fid])
        ax.set_xlabel("Time (s)")
        if ax_idx == 0:
            ax.set_ylabel("Metric value")

        # Give extra headroom for avg text, but do not show 1.2 as a tick
        ax.set_ylim(0, 1.3)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        ax.grid(True, alpha=0.3)

    # Global legend at the top center
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=len(GPU_IDS),
        frameon=True,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(PLOT_FILE, dpi=160)
    plt.close(fig)
    print(f"[OK] Saved plot to {PLOT_FILE}")


def summarize(data) -> None:
    print("\n=== Summary ===")
    for gid in GPU_IDS:
        rows = data.get(gid, [])
        if not rows:
            print(f"GPU {gid}: no samples")
            continue
        msg = [f"GPU {gid}: samples={len(rows)}"]
        for fid in FIELD_IDS:
            avg = sum(r[fid] for r in rows) / len(rows)
            msg.append(f"{FIELD_NAMES[fid]} avg={avg:.4f}")
        print(", ".join(msg))


def main() -> None:
    require_cmd("dcgmi")
    require_cmd("make")

    check_dcgm_host_running()
    compile_workload()

    dmon_proc = start_dmon()
    time.sleep(1.0)
    workload_proc = start_workload()

    try:
        data = parse_dmon_stream(dmon_proc, workload_proc)
    finally:
        terminate_process(workload_proc, "workload")
        terminate_process(dmon_proc, "dcgmi dmon")

    save_csv(data)
    plot_subplots(data)
    summarize(data)

    print("\nArtifacts:")
    print(f"  Raw log : {RAW_LOG}")
    print(f"  CSV     : {CSV_LOG}")
    print(f"  Plot    : {PLOT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)