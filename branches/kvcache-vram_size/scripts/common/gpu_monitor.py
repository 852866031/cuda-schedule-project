#!/usr/bin/env python3
"""Per-GPU utilization sampler built on DCGM profiling metrics.

Samples, once a second per GPU, into a CSV:

    ts,gpu,sm_active,sm_occupancy,dram_active,fb_used_mib

  sm_active     fraction of time at least one warp was resident on an SM (SMACT, 1002)
  sm_occupancy  resident warps / maximum warps (SMOCC, 1003)
  dram_active   fraction of cycles the DRAM interface was busy (DRAMA, 1005) -- for this
                project the most important line: decode is memory-bandwidth-bound
  fb_used_mib   framebuffer memory in use (FB_USED, 252)

Verified working on this box's RTX 5090s (GeForce; DCGM 4.x). Runs until killed:

    scripts/common/gpu_monitor.py output/gpumon/run.csv
"""

import subprocess
import sys
import time

FIELDS = "1002,1003,1005,252"


def main():
    out_path = sys.argv[1]
    proc = subprocess.Popen(
        ["dcgmi", "dmon", "-e", FIELDS, "-d", "1000"],
        stdout=subprocess.PIPE, text=True, bufsize=1)
    with open(out_path, "w", buffering=1) as out:
        out.write("ts,gpu,sm_active,sm_occupancy,dram_active,fb_used_mib\n")
        for line in proc.stdout:
            parts = line.split()
            # data rows look like: GPU 0 0.123 0.045 0.678 24567.000
            if len(parts) != 6 or parts[0] != "GPU":
                continue
            try:
                gpu = int(parts[1])
                vals = [float(x) for x in parts[2:]]
            except ValueError:
                continue   # the first sample per field is N/A
            out.write(f"{time.time():.1f},{gpu},{vals[0]:.4f},{vals[1]:.4f},"
                      f"{vals[2]:.4f},{vals[3]:.0f}\n")


if __name__ == "__main__":
    main()
