#!/usr/bin/env python
"""Apply a timed sequence of kvcached memory limits with `kvctl limit`, logging the wall-clock
instant each command was issued and finished (JSONL), so the sampler trace can be aligned.
  resize.py --ipc vllm_a --out results/x_resize.jsonl --step 40:4G --step 90:12G
Steps are 'seconds_after_start:size'. The script sleeps until each offset."""
import argparse, json, subprocess, time
ap = argparse.ArgumentParser()
ap.add_argument("--ipc", default="vllm_a")
ap.add_argument("--out", required=True)
ap.add_argument("--step", action="append", required=True)
ap.add_argument("--t0", type=float, default=None, help="epoch of t=0 (default: now)")
args = ap.parse_args()
t0 = args.t0 or time.time()
with open(args.out, "a") as f:
    for s in args.step:
        off, size = s.split(":")
        time.sleep(max(0.0, t0 + float(off) - time.time()))
        t_cmd = time.time()
        r = subprocess.run(["kvctl", "limit", args.ipc, size], capture_output=True, text=True)
        t_done = time.time()
        rec = {"t_cmd": t_cmd, "t_done": t_done, "offset": float(off), "size": size,
               "rc": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()[-500:]}
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(json.dumps(rec))
