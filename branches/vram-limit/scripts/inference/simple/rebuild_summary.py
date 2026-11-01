#!/usr/bin/env python3
"""Regenerate a summary CSV from output/raw/*.json using the current derive_row().

Derived columns change as we learn what the engine actually reports (e.g. vLLM 0.15.1 drops
the KV connector byte counters that the dev build had). Re-deriving from the saved raw records
means a derivation fix never costs a re-run of the experiment.

    python scripts/rebuild_summary.py --tag main
"""

import argparse
import csv
import json
from pathlib import Path

from run_sweep import derive_row
from workload import build_workload

REPO = Path(__file__).resolve().parent.parent.parent.parent
OUT = REPO / "output"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--sessions", type=int, default=32)
    ap.add_argument("--prefix-len", type=int, default=6144)
    args = ap.parse_args()

    wl = build_workload(args.sessions, args.prefix_len)  # only geometry is used by derive_row
    rows = []
    for p in sorted((OUT / "raw").glob("*.json")):
        rec = json.loads(p.read_text())
        if "config" not in rec:
            continue
        if args.tag == "main" and rec["config"].get("kv_offload_gib") not in (None, 24.0):
            continue  # skip smoke/probe records with other pool sizes
        rows.append(derive_row(rec, wl))

    if not rows:
        print("no records found")
        return
    path = OUT / f"summary_{args.tag}_rebuilt.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"rebuilt {len(rows)} rows -> {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
