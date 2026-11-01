#!/usr/bin/env python3
import csv
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

CSV_LOG = OUT_DIR / "dcgm_metrics.csv"
RAW_LOG = OUT_DIR / "dcgm_raw.log"

GPU_IDS = [0, 1]
FIELD_IDS = ["1002", "1003"]


def require_cmd(cmd: str):
    if shutil.which(cmd) is None:
        raise RuntimeError(f"{cmd} not found")


def start_dmon(interval_ms=100):
    cmd = [
        "dcgmi", "dmon",
        "-d", str(interval_ms),
        "-e", ",".join(FIELD_IDS),
        "-i", ",".join(map(str, GPU_IDS)),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)


def parse_dmon_stream(dmon_proc, workload_proc):
    data = defaultdict(list)
    done_time = None
    tail_s = 1.0
    start = time.time()

    with RAW_LOG.open("w") as rawf:
        while True:
            if done_time is None and workload_proc.poll() is not None:
                done_time = time.time()

            if done_time is not None and time.time() - done_time > tail_s:
                break

            line = dmon_proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue

            rawf.write(line)
            parts = line.strip().split()
            if len(parts) < 3 or parts[0] != "GPU":
                continue

            try:
                gid = int(parts[1])
            except ValueError:
                continue

            nums = []
            for token in parts[2:]:
                try:
                    nums.append(float(token))
                except ValueError:
                    pass

            if len(nums) < 2:
                continue

            data[gid].append(
                {
                    "time_s": time.time() - start,
                    "sm_active": nums[0],
                    "sm_occupancy": nums[1],
                }
            )
    return data


def save_csv(data):
    with CSV_LOG.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gpu_id", "time_s", "sm_active", "sm_occupancy"])
        for gid in sorted(data.keys()):
            for row in data[gid]:
                writer.writerow([gid, f"{row['time_s']:.6f}", row["sm_active"], row["sm_occupancy"]])


def terminate(proc):
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()


def main():
    require_cmd("dcgmi")
    dmon_proc = start_dmon(interval_ms=100)
    time.sleep(1.0)
    workload_proc = subprocess.Popen([sys.executable, str(THIS_DIR / "inf_sys.py")])

    try:
        data = parse_dmon_stream(dmon_proc, workload_proc)
    finally:
        terminate(workload_proc)
        terminate(dmon_proc)

    save_csv(data)
    print(f"Saved DCGM CSV to {CSV_LOG}")


if __name__ == "__main__":
    main()