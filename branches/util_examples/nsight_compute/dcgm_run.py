#!/usr/bin/env python3
import csv
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


GPU_IDS = [0, 1]
FIELD_IDS = ["1002", "1003"]  # 1002 = SM_ACTIVE, 1003 = SM_OCCUPANCY

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RAW_LOG = OUT_DIR / "dcgm_raw.log"
CSV_LOG = OUT_DIR / "dcgm_metrics.csv"


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


def terminate_process(proc: subprocess.Popen, name: str) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        print(f"[WARN] {name} did not terminate, killing it.")
        proc.kill()


def start_dmon(interval_ms: int = 100) -> subprocess.Popen:
    cmd = [
        "dcgmi", "dmon",
        "-d", str(interval_ms),
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
    cmd = [sys.executable, str(THIS_DIR / "workload_two_gpu.py")]
    print("[INFO] Starting workload:", " ".join(cmd))
    return subprocess.Popen(cmd, stdout=None, stderr=None, text=True)


def parse_dmon_stream(dmon_proc: subprocess.Popen, workload_proc: subprocess.Popen):
    start_time = time.time()
    data = defaultdict(list)

    workload_done_time = None
    tail_seconds = 1.0

    with RAW_LOG.open("w") as rawf:
        while True:
            if workload_done_time is None and workload_proc.poll() is not None:
                workload_done_time = time.time()

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

            if len(numeric_tokens) < len(FIELD_IDS):
                continue

            t = time.time() - start_time
            row = {
                "time_s": t,
                "sm_active": numeric_tokens[0],
                "sm_occupancy": numeric_tokens[1],
            }
            data[gpu_id].append(row)

    return data


def save_csv(data) -> None:
    with CSV_LOG.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gpu_id", "time_s", "sm_active", "sm_occupancy"])
        for gid in sorted(data.keys()):
            for row in data[gid]:
                writer.writerow([
                    gid,
                    f"{row['time_s']:.6f}",
                    row["sm_active"],
                    row["sm_occupancy"],
                ])

    print(f"[OK] Saved CSV to {CSV_LOG}")


def summarize(data) -> None:
    print("\n=== Summary ===")
    for gid in GPU_IDS:
        rows = data.get(gid, [])
        if not rows:
            print(f"GPU {gid}: no samples")
            continue
        avg_active = sum(r["sm_active"] for r in rows) / len(rows)
        avg_occ = sum(r["sm_occupancy"] for r in rows) / len(rows)
        print(
            f"GPU {gid}: samples={len(rows)}, "
            f"avg SM_ACTIVE={avg_active:.4f}, "
            f"avg SM_OCCUPANCY={avg_occ:.4f}"
        )


def main():
    require_cmd("dcgmi")
    check_dcgm_host_running()

    dmon_proc = start_dmon(interval_ms=100)
    time.sleep(1.0)
    workload_proc = start_workload()

    try:
        data = parse_dmon_stream(dmon_proc, workload_proc)
    finally:
        terminate_process(workload_proc, "workload")
        terminate_process(dmon_proc, "dcgmi dmon")

    save_csv(data)
    summarize(data)

    print("\nArtifacts:")
    print(f"  Raw log : {RAW_LOG}")
    print(f"  CSV     : {CSV_LOG}")


if __name__ == "__main__":
    main()