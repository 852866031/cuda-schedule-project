#!/usr/bin/env python3
"""Per-GPU DCGM sampler for the priority study: scripts/common/gpu_monitor.py plus the
tensor-pipe field, the honest "compute utilization" for prefill vs decode (SM-active
reads high for decode too). One row per GPU per second:

    ts,gpu,sm_active,sm_occupancy,tensor_active,dram_active,fb_used_mib

    gpu_monitor.py out.csv     # runs until killed
"""

import subprocess
import sys
import time

FIELDS = "1002,1003,1004,1005,252"   # SMACT, SMOCC, PIPE_TENSOR_ACTIVE, DRAMA, FB_USED


def main():
    proc = subprocess.Popen(["dcgmi", "dmon", "-e", FIELDS, "-d", "1000"],
                            stdout=subprocess.PIPE, text=True, bufsize=1)
    with open(sys.argv[1], "w", buffering=1) as out:
        out.write("ts,gpu,sm_active,sm_occupancy,tensor_active,dram_active,fb_used_mib\n")
        for line in proc.stdout:
            p = line.split()
            if len(p) != 7 or p[0] != "GPU":
                continue
            try:
                gpu, v = int(p[1]), [float(x) for x in p[2:]]
            except ValueError:
                continue
            out.write(f"{time.time():.1f},{gpu},{v[0]:.4f},{v[1]:.4f},{v[2]:.4f},"
                      f"{v[3]:.4f},{v[4]:.0f}\n")


if __name__ == "__main__":
    main()
