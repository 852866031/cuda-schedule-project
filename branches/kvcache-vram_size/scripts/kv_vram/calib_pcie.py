#!/usr/bin/env python3
"""E0.2: pinned / pageable H2D and D2H bandwidth on one GPU, reusing
scripts/common/calibrate_pcie.py's timing (CUDA events, median of N copies).
Adds the transfer sizes this study moves: 4 MiB-ish per-layer-per-block-run pieces,
25 MiB = one request's KV for ONE layer (6400 tokens x 4 KiB), 32 MiB = one LMCache
256-token chunk (all 32 layers), 800 MiB = one whole request.

    .venv/bin/python scripts/kv_vram/calib_pcie.py --gpu 1   # -> output/kv_vram/pcie_gpu1.json
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, required=True)
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import torch                    # noqa: E402
import calibrate_pcie as c      # noqa: E402

c.SIZES_MIB = [1, 4, 16, 25, 32, 64, 256, 800, 1024]


def link():
    q = subprocess.run(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=pcie.link.gen.current,"
                        "pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return [int(x) for x in q.stdout.strip().split(",")]


print(f"GPU {args.gpu}: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
idle = link()
pinned = c.bandwidth_sweep(pinned=True)
loaded = link()
pageable = c.bandwidth_sweep(pinned=False)
out = {"gpu_index": args.gpu, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "link_idle_gen_maxgen_width_maxwidth": idle, "link_after_load": loaded,
       "pinned": pinned, "pageable": pageable,
       "peak_pinned_h2d_gbps": max(r["h2d_gbps"] for r in pinned),
       "peak_pinned_d2h_gbps": max(r["d2h_gbps"] for r in pinned)}
dst = HERE.parents[1] / "output" / "kv_vram" / f"pcie_gpu{args.gpu}.json"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, indent=1))
print(f"link idle {idle} after load {loaded}; peak pinned H2D {out['peak_pinned_h2d_gbps']} GB/s -> {dst}")
