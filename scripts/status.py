#!/usr/bin/env python3
"""Live status of whatever sweep is running. Safe to run any time -- reads only.

    python scripts/status.py           # snapshot
    watch -n 20 python scripts/status.py   # refresh every 20s
"""

import csv
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT, LOGS = REPO / "output", REPO / "output" / "logs"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def running():
    """(driver command, server config name) for whatever is in flight."""
    try:
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    except Exception:
        return None, None
    driver = server = None
    for line in ps.splitlines():
        if "run_sweep.py" in line and "ps -eo" not in line:
            driver = line.split("run_sweep.py")[1].strip()[:70]
        if "vllm.entrypoints.cli.main" in line:
            m = re.search(r"--gpu-memory-utilization\s+([\d.]+)", line)
            off = "--kv-offloading-size" in line
            if m:
                server = f"util={m.group(1)} {'offload' if off else 'no-offload'}"
    return driver, server


def all_rows():
    rows = []
    for p in sorted(OUT.glob("summary_*.csv")):
        if any(t in p.name for t in ("smoke", "pilot")):
            continue
        for r in csv.DictReader(open(p)):
            r["_src"] = p.name
            rows.append(r)
    return rows


def latest_log():
    logs = [p for p in LOGS.glob("*.log") if p.stat().st_size > 0]
    return max(logs, key=lambda p: p.stat().st_mtime) if logs else None


def main():
    driver, server = running()
    print("=" * 78)
    if driver:
        print(f"RUNNING  run_sweep.py {driver}")
        print(f"         server: {server or '(starting/stopped)'}")
    else:
        print("RUNNING  nothing -- no sweep in flight")

    log = latest_log()
    if log:
        age = time.time() - log.stat().st_mtime
        text = ANSI.sub("", log.read_text(errors="replace"))
        eng = [l for l in text.splitlines() if "Engine 000:" in l]
        print(f"\nCURRENT CONFIG  {log.stem}   (log touched {age:.0f}s ago)")
        if eng:
            last = eng[-1].split("Engine 000:")[1].strip()
            print(f"  {last[:150]}")
        if age > 120 and driver:
            print(f"  !! log idle {age:.0f}s -- watchdog fires at 180s")

    rows = [r for r in all_rows() if r.get("gpu_kv_gib")]
    print(f"\nCOMPLETED  {len(rows)} configs")
    print(f"{'config':24s} {'KV GiB':>7s} {'ttft50':>8s} {'ttft95':>9s} {'itl':>6s} "
          f"{'dram%':>6s} {'preempt':>8s} {'tok/s':>7s} {'hung':>5s}")
    for r in sorted(rows, key=lambda r: (r["skew"], r["arm"], -float(r["budget_gib"]))):
        dram = r.get("dram_hit_rate") or ""
        dram = f"{float(dram) * 100:.0f}" if dram else "-"
        print(f"{r['name']:24s} {r['gpu_kv_gib']:>7s} {r['ttft_p50_ms']:>8s} "
              f"{r['ttft_p95_ms']:>9s} {r['itl_p50_ms']:>6s} {dram:>6s} "
              f"{r['preemptions']:>8s} {r['output_tok_per_s']:>7s} "
              f"{'YES' if r.get('hung') == 'True' else '':>5s}")

    # Each config costs ~3.5 min: ~25s startup + ~20s warmup + 151s measured + teardown.
    if driver and "uniform" in driver:
        done = len([r for r in rows if r["skew"] == "uniform"])
        left = 8 - done + (1 if "nooffload" not in driver else 0)
        print(f"\nETA  ~{left} configs left x ~3.5 min = ~{left * 3.5:.0f} min")
    print("=" * 78)


if __name__ == "__main__":
    main()
