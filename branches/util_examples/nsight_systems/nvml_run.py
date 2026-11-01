#!/usr/bin/env python3
import csv
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

from pynvml import *

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

CSV_LOG = OUT_DIR / "nvml_metrics.csv"

GPU_IDS = [0, 1]


def monitor(proc, interval_s=0.1):
    nvmlInit()
    handles = {gid: nvmlDeviceGetHandleByIndex(gid) for gid in GPU_IDS}
    data = defaultdict(list)

    start = time.time()
    done_time = None
    tail_s = 1.0

    while True:
        if done_time is None and proc.poll() is not None:
            done_time = time.time()

        if done_time is not None and time.time() - done_time > tail_s:
            break

        t = time.time() - start
        for gid in GPU_IDS:
            util = nvmlDeviceGetUtilizationRates(handles[gid])
            mem = nvmlDeviceGetMemoryInfo(handles[gid])
            data[gid].append(
                {
                    "time_s": t,
                    "gpu_util": util.gpu,
                    "mem_util": util.memory,
                    "mem_used_mb": mem.used / (1024**2),
                }
            )
        time.sleep(interval_s)

    nvmlShutdown()
    return data


def save_csv(data):
    with CSV_LOG.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gpu_id", "time_s", "gpu_util", "mem_util", "mem_used_mb"])
        for gid in sorted(data.keys()):
            for row in data[gid]:
                writer.writerow(
                    [gid, f"{row['time_s']:.6f}", row["gpu_util"], row["mem_util"], f"{row['mem_used_mb']:.3f}"]
                )


def main():
    proc = subprocess.Popen([sys.executable, str(THIS_DIR / "inf_sys.py")])
    data = monitor(proc, interval_s=0.1)
    proc.wait()
    save_csv(data)
    print(f"Saved NVML CSV to {CSV_LOG}")


if __name__ == "__main__":
    main()