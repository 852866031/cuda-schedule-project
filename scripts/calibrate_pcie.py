#!/usr/bin/env python3
"""Phase 0 calibration: the physical constants every later number is read against.

Measures three things on GPU 0:

  1. Pinned-memory H2D / D2H bandwidth across transfer sizes. The KV offload path moves
     blocks over PCIe, so this bandwidth is the ceiling on what offloading can cost.
  2. Whether a 24 GiB pinned host allocation actually succeeds. `ulimit -l` on this box is
     ~7.5 GiB, far under the planned CPU KV pool. CUDA's cudaHostAlloc normally pins on the
     process's behalf and escapes RLIMIT_MEMLOCK, but the sweep depends on it, so prove it.
  3. Pageable (non-pinned) bandwidth, for contrast -- it is the number you get if the
     offload pool ever falls back to unpinned memory.

Writes output/calibration_pcie.json.

Usage:
    python scripts/calibrate_pcie.py                 # full run
    python scripts/calibrate_pcie.py --skip-big-pin  # skip the 24 GiB allocation test
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"

# KV cache geometry for Llama-3-8B fp16: 32 layers * 8 kv heads * 128 dim * 2 (K,V) * 2 bytes
KV_BYTES_PER_TOKEN = 32 * 8 * 128 * 2 * 2  # 131072 B = 0.125 MiB

SIZES_MIB = [1, 4, 16, 64, 256, 1024]
MIB = 1 << 20
GIB = 1 << 30


def pcie_link_state() -> dict:
    """Current PCIe generation/width. Gen reads as 1 at idle and ramps under load."""
    try:
        q = subprocess.run(
            ["nvidia-smi", "-i", "0", "--query-gpu=pcie.link.gen.current,"
             "pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        gen_cur, gen_max, w_cur, w_max = [x.strip() for x in q.stdout.strip().split(",")]
        return {"gen_current": int(gen_cur), "gen_max": int(gen_max),
                "width_current": int(w_cur), "width_max": int(w_max)}
    except Exception as e:  # nvidia-smi shape varies across driver versions
        return {"error": str(e)}


def time_copy(dst, src, iters: int, warmup: int = 3) -> float:
    """Median seconds per copy, timed with CUDA events on the current stream."""
    for _ in range(warmup):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        dst.copy_(src, non_blocking=True)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def bandwidth_sweep(pinned: bool) -> list:
    """H2D and D2H bandwidth at each transfer size."""
    rows = []
    for size_mib in SIZES_MIB:
        n = size_mib * MIB
        host = torch.empty(n, dtype=torch.uint8, pin_memory=pinned)
        dev = torch.empty(n, dtype=torch.uint8, device="cuda")
        # Small transfers need many iterations to time reliably; large ones few.
        iters = max(5, min(50, int(2048 / size_mib)))

        h2d = time_copy(dev, host, iters)
        d2h = time_copy(host, dev, iters)

        rows.append({
            "size_mib": size_mib,
            "iters": iters,
            "h2d_ms": round(h2d * 1e3, 4),
            "d2h_ms": round(d2h * 1e3, 4),
            "h2d_gbps": round(n / h2d / 1e9, 3),   # decimal GB/s, as PCIe is spec'd
            "d2h_gbps": round(n / d2h / 1e9, 3),
        })
        print(f"  {'pinned' if pinned else 'pageable':9s} {size_mib:5d} MiB  "
              f"H2D {rows[-1]['h2d_gbps']:6.2f} GB/s  D2H {rows[-1]['d2h_gbps']:6.2f} GB/s")
        del host, dev
        torch.cuda.empty_cache()
    return rows


def big_pin_test(target_gib: int) -> dict:
    """Can we pin `target_gib` of host memory? Allocate in 2 GiB chunks and hold them all."""
    chunk = 2 * GIB
    chunks, pinned_gib = [], 0
    t0 = time.time()
    try:
        for _ in range(target_gib // 2):
            chunks.append(torch.empty(chunk, dtype=torch.uint8, pin_memory=True))
            pinned_gib += 2
        ok, err = True, None
    except Exception as e:
        ok, err = False, f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0

    # With the full pool held, confirm transfers still work at speed.
    h2d_gbps = None
    if chunks:
        dev = torch.empty(256 * MIB, dtype=torch.uint8, device="cuda")
        src = chunks[-1][: 256 * MIB]
        h2d_gbps = round((256 * MIB) / time_copy(dev, src, 10) / 1e9, 3)
        del dev

    del chunks
    torch.cuda.empty_cache()
    return {
        "target_gib": target_gib, "pinned_gib": pinned_gib, "success": ok,
        "error": err, "alloc_seconds": round(elapsed, 2),
        "h2d_gbps_while_held": h2d_gbps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin-gib", type=int, default=24, help="pinned pool size to prove (GiB)")
    ap.add_argument("--skip-big-pin", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  {props.total_memory / GIB:.2f} GiB  torch {torch.__version__}")

    link_idle = pcie_link_state()
    print(f"PCIe idle: gen {link_idle.get('gen_current')}/{link_idle.get('gen_max')} "
          f"x{link_idle.get('width_current')}/{link_idle.get('width_max')}")

    print("\nPinned bandwidth:")
    pinned = bandwidth_sweep(pinned=True)
    link_load = pcie_link_state()  # sampled right after sustained traffic
    print(f"PCIe after load: gen {link_load.get('gen_current')} x{link_load.get('width_current')}")

    print("\nPageable bandwidth (contrast):")
    pageable = bandwidth_sweep(pinned=False)

    big_pin = None
    if not args.skip_big_pin:
        print(f"\nPinning {args.pin_gib} GiB of host memory (ulimit -l is ~7.5 GiB)...")
        big_pin = big_pin_test(args.pin_gib)
        print(f"  success={big_pin['success']} pinned={big_pin['pinned_gib']} GiB "
              f"in {big_pin['alloc_seconds']}s  err={big_pin['error']}")

    # Peak sustained bandwidth, taken at the largest transfer size.
    peak_h2d = max(r["h2d_gbps"] for r in pinned)
    peak_d2h = max(r["d2h_gbps"] for r in pinned)

    # The constant the whole Case A curve is read against: cost of pulling one 8192-token
    # prefix (= 1 GiB of KV for this model) back from DRAM.
    kv_gib_per_8k = 8192 * KV_BYTES_PER_TOKEN / GIB
    t_load_8k_ms = round((8192 * KV_BYTES_PER_TOKEN) / (peak_h2d * 1e9) * 1e3, 2)

    result = {
        "gpu": props.name,
        "gpu_total_gib": round(props.total_memory / GIB, 3),
        "torch": torch.__version__,
        "pcie_idle": link_idle,
        "pcie_under_load": link_load,
        "pinned": pinned,
        "pageable": pageable,
        "big_pin_test": big_pin,
        "derived": {
            "peak_h2d_gbps": peak_h2d,
            "peak_d2h_gbps": peak_d2h,
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
            "kv_gib_per_8k_prefix": round(kv_gib_per_8k, 3),
            "t_load_8k_prefix_ms": t_load_8k_ms,
        },
    }
    path = OUT / "calibration_pcie.json"
    path.write_text(json.dumps(result, indent=2))

    print(f"\npeak pinned H2D {peak_h2d:.2f} GB/s | D2H {peak_d2h:.2f} GB/s")
    print(f"t_load for an 8192-token prefix ({kv_gib_per_8k:.2f} GiB of KV): {t_load_8k_ms} ms")
    print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
