#!/usr/bin/env python3
import csv
import itertools
import math
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "tuning_outputs"
OUT_DIR.mkdir(exist_ok=True)

RESULTS_CSV = OUT_DIR / "tuning_results.csv"

GPU_IDS = [0, 1]
FIELD_IDS = ["1002", "1003"]  # 1002 = SM_ACTIVE, 1003 = SM_OCCUPANCY


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
        return

    proc = subprocess.run(
        ["pgrep", "-f", "nv-hostengine"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return

    raise RuntimeError(
        "DCGM host is not running.\n"
        "Try: sudo systemctl --now enable nvidia-dcgm"
    )


def terminate_process(proc: subprocess.Popen, name: str) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()


def start_dmon(interval_ms: int = 100) -> subprocess.Popen:
    cmd = [
        "dcgmi", "dmon",
        "-d", str(interval_ms),
        "-e", ",".join(FIELD_IDS),
        "-i", "0,1",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def start_workload(arg_dict):
    cmd = [sys.executable, str(THIS_DIR / "workload_two_gpu.py")]
    for k, v in arg_dict.items():
        cmd.extend([f"--{k}", str(v)])
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def parse_dmon_stream(dmon_proc: subprocess.Popen, workload_proc: subprocess.Popen):
    data = defaultdict(list)

    workload_done_time = None
    tail_seconds = 1.0

    while True:
        if workload_done_time is None and workload_proc.poll() is not None:
            workload_done_time = time.time()

        if workload_done_time is not None and (time.time() - workload_done_time) > tail_seconds:
            break

        line = dmon_proc.stdout.readline()
        if not line:
            time.sleep(0.05)
            continue

        parts = line.strip().split()
        if len(parts) < 3 or parts[0] != "GPU":
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

        if len(numeric_tokens) < 2:
            continue

        data[gpu_id].append(
            {
                "sm_active": numeric_tokens[0],
                "sm_occupancy": numeric_tokens[1],
            }
        )

    return data


def nonzero_mean(values, eps=1e-12):
    """Average after dropping zeros (and near-zeros)."""
    kept = [v for v in values if abs(v) > eps]
    if not kept:
        return None, 0, len(values)
    return sum(kept) / len(kept), len(kept), len(values)


def summarize_run(data):
    """
    Drop zero values before averaging.
    Return both averages and how many nonzero samples were kept.
    """
    summary = {}
    for gid in GPU_IDS:
        rows = data.get(gid, [])

        active_vals = [r["sm_active"] for r in rows]
        occ_vals = [r["sm_occupancy"] for r in rows]

        active_avg, active_kept, active_total = nonzero_mean(active_vals)
        occ_avg, occ_kept, occ_total = nonzero_mean(occ_vals)

        summary[gid] = {
            "sm_active_avg": active_avg,
            "sm_occupancy_avg": occ_avg,
            "sm_active_kept": active_kept,
            "sm_active_total": active_total,
            "sm_occupancy_kept": occ_kept,
            "sm_occupancy_total": occ_total,
        }
    return summary


def score_summary(summary):
    """
    Lower is better.

    We want:
    - GPU0/GPU1 SM_ACTIVE similar
    - GPU0/GPU1 SM_OCCUPANCY similar
    - but avoid trivial solutions where both are low
    - also avoid solutions where too many samples are zero / dropped
    """
    g0 = summary[0]
    g1 = summary[1]

    if (
        g0["sm_active_avg"] is None or g1["sm_active_avg"] is None or
        g0["sm_occupancy_avg"] is None or g1["sm_occupancy_avg"] is None
    ):
        return 1e9

    active_diff = abs(g0["sm_active_avg"] - g1["sm_active_avg"])
    occ_diff = abs(g0["sm_occupancy_avg"] - g1["sm_occupancy_avg"])

    active_mean = 0.5 * (g0["sm_active_avg"] + g1["sm_active_avg"])
    occ_mean = 0.5 * (g0["sm_occupancy_avg"] + g1["sm_occupancy_avg"])

    # Reject "both low" solutions by penalizing low mean activity/occupancy
    # Tune these thresholds to your preference.
    active_floor = 0.20
    occ_floor = 0.05

    active_penalty = max(0.0, active_floor - active_mean) * 8.0
    occ_penalty = max(0.0, occ_floor - occ_mean) * 8.0

    # Penalize runs where too many samples were zero and got dropped
    g0_active_keep_ratio = g0["sm_active_kept"] / max(1, g0["sm_active_total"])
    g1_active_keep_ratio = g1["sm_active_kept"] / max(1, g1["sm_active_total"])
    g0_occ_keep_ratio = g0["sm_occupancy_kept"] / max(1, g0["sm_occupancy_total"])
    g1_occ_keep_ratio = g1["sm_occupancy_kept"] / max(1, g1["sm_occupancy_total"])

    keep_ratio = min(
        g0_active_keep_ratio,
        g1_active_keep_ratio,
        g0_occ_keep_ratio,
        g1_occ_keep_ratio,
    )
    keep_penalty = max(0.0, 0.50 - keep_ratio) * 5.0

    return active_diff + occ_diff + active_penalty + occ_penalty + keep_penalty


def run_one_config(arg_dict):
    dmon_proc = start_dmon(interval_ms=100)
    time.sleep(1.0)
    workload_proc = start_workload(arg_dict)

    try:
        data = parse_dmon_stream(dmon_proc, workload_proc)
    finally:
        terminate_process(workload_proc, "workload")
        terminate_process(dmon_proc, "dcgmi dmon")

    summary = summarize_run(data)
    score = score_summary(summary)
    return summary, score


def main():
    require_cmd("dcgmi")
    check_dcgm_host_running()

    # Keep the search space modest first.
    search_space = {
        "duration-s": [1],
        "n": [1 << 22],
        "cpu-sleep-ms": [20.0],

        "g0-threads": [256],
        "g0-blocks-mult": [4.0],
        "g0-inner-iters": [64, 96, 128],
        "g0-launches-per-iter": [64, 128],
        "g0-shmem-kb": [1],

        "g1-threads": [128, 256],
        "g1-blocks-mult": [4.0],
        "g1-inner-iters": [16, 32],
        "g1-launches-per-iter": [4, 6, 8],
        "g1-pointwise-repeats": [1],
        "g1-shmem-kb": [1],
    }

    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    total = math.prod(len(v) for v in values)

    rows_out = []

    for idx, combo in enumerate(itertools.product(*values), start=1):
        arg_dict = dict(zip(keys, combo))
        print(f"[{idx}/{total}] Running config: {arg_dict}")

        summary, score = run_one_config(arg_dict)

        row = dict(arg_dict)
        row.update(
            {
                "gpu0_sm_active_avg": summary[0]["sm_active_avg"],
                "gpu1_sm_active_avg": summary[1]["sm_active_avg"],
                "gpu0_sm_occupancy_avg": summary[0]["sm_occupancy_avg"],
                "gpu1_sm_occupancy_avg": summary[1]["sm_occupancy_avg"],

                "gpu0_sm_active_kept": summary[0]["sm_active_kept"],
                "gpu0_sm_active_total": summary[0]["sm_active_total"],
                "gpu1_sm_active_kept": summary[1]["sm_active_kept"],
                "gpu1_sm_active_total": summary[1]["sm_active_total"],

                "gpu0_sm_occupancy_kept": summary[0]["sm_occupancy_kept"],
                "gpu0_sm_occupancy_total": summary[0]["sm_occupancy_total"],
                "gpu1_sm_occupancy_kept": summary[1]["sm_occupancy_kept"],
                "gpu1_sm_occupancy_total": summary[1]["sm_occupancy_total"],

                "score": score,
            }
        )
        rows_out.append(row)

        def fmt(x):
            return "None" if x is None else f"{x:.4f}"

        print(
            f"    score={score:.4f} | "
            f"g0_active={fmt(summary[0]['sm_active_avg'])} "
            f"({summary[0]['sm_active_kept']}/{summary[0]['sm_active_total']} kept), "
            f"g1_active={fmt(summary[1]['sm_active_avg'])} "
            f"({summary[1]['sm_active_kept']}/{summary[1]['sm_active_total']} kept), "
            f"g0_occ={fmt(summary[0]['sm_occupancy_avg'])} "
            f"({summary[0]['sm_occupancy_kept']}/{summary[0]['sm_occupancy_total']} kept), "
            f"g1_occ={fmt(summary[1]['sm_occupancy_avg'])} "
            f"({summary[1]['sm_occupancy_kept']}/{summary[1]['sm_occupancy_total']} kept)"
        )

    rows_out.sort(key=lambda x: x["score"])

    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nSaved results to {RESULTS_CSV}")
    print("\nTop 10 configs:")
    for row in rows_out[:10]:
        print(row)


if __name__ == "__main__":
    main()