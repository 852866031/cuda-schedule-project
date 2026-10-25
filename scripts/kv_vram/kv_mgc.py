#!/usr/bin/env python3
"""Queueing-theory check of the decode engine as c integer KV slots.

For every traced cell (timeline CSVs from kv_timeline.py) compare the measured decode
queue wait with three predictions that use only c (from the probe) and the measured
residence S (t_fin - t_sched per request):

  mmc        Erlang C with mean S (exponential service)          -- M/M/c
  ll         Lee-Longton: Wq(M/G/c) ~ (1 + Cs^2) / 2 * Wq(M/M/c)   -- M/G/c approx.
  replay     discrete-event replay: the cell's own decode arrival times and each
             request's own S, c FCFS servers. Exact for "c slots, FCFS, this S", so a
             gap to the measurement is what integer slots do NOT explain (fractional
             slack + preemption, batch-dependent S).
  replay_c1  the same with c + 1 servers (what one more slot would have bought)

Arrival rate for the formulas = measured decode arrivals / window. rho = lambda S / c.
Output: output/kv_vram/mgc.csv

    .venv/bin/python scripts/kv_vram/kv_mgc.py
"""
import csv
import heapq
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TL = REPO / "output" / "kv_vram" / "timeline"
RAW = REPO / "output" / "raw"


def erlang_c_wait(lam, s_mean, c):
    a = lam * s_mean
    rho = a / c
    if rho >= 1:
        return math.inf
    terms = sum(a ** k / math.factorial(k) for k in range(c))
    top = a ** c / math.factorial(c) / (1 - rho)
    pw = top / (terms + top)
    return pw * s_mean / (c * (1 - rho))


def replay(arr, svc, c):
    """FCFS, c servers; arr sorted. Returns per-request waits."""
    free = [0.0] * c
    heapq.heapify(free)
    waits = []
    for a, s in zip(arr, svc):
        t = heapq.heappop(free)
        start = max(a, t)
        waits.append(start - a)
        heapq.heappush(free, start + s)
    return np.array(waits)


def main():
    rows = []
    for p in sorted(TL.glob("kv_*_req.csv")):
        name = p.name[:-len("_req.csv")]
        raw = RAW / f"{name}.json"
        if not raw.exists():
            continue
        rec = json.loads(raw.read_text())
        c = rec["launch"]["decode"]["slots"]
        rq = [r for r in csv.DictReader(open(p)) if r.get("t_sched") and r.get("t_fin")]
        rq.sort(key=lambda r: float(r["t_arr"]))
        arr = np.array([float(r["t_arr"]) for r in rq])
        res = np.array([float(r["t_fin"]) - float(r["t_sched"]) for r in rq])
        wait = np.array([float(r["wait"]) for r in rq])
        lam = (len(arr) - 1) / (arr[-1] - arr[0])
        s_mean, cs2 = res.mean(), res.var() / res.mean() ** 2
        mmc = erlang_c_wait(lam, s_mean, c)
        rp, rp1 = replay(arr, res, c), replay(arr, res, c + 1)
        rows.append({
            "name": name, "method": rec["method"], "budget_gib": rec["budget_gib"], "qps": rec["qps"],
            "c": c, "lambda": round(lam, 3), "S_mean_s": round(s_mean, 3), "S_cv2": round(cs2, 3),
            "rho": round(lam * s_mean / c, 3),
            "wait_meas_ms": round(1000 * wait.mean(), 1),
            "wait_meas_p95_ms": round(1000 * np.percentile(wait, 95), 1),
            "wait_mmc_ms": round(1000 * mmc, 1) if math.isfinite(mmc) else None,
            "wait_ll_ms": round(1000 * mmc * (1 + cs2) / 2, 1) if math.isfinite(mmc) else None,
            "wait_replay_ms": round(1000 * rp.mean(), 1),
            "wait_replay_p95_ms": round(1000 * np.percentile(rp, 95), 1),
            "wait_replay_c1_ms": round(1000 * rp1.mean(), 1),
            "preempted": sum(1 for r in rq if r.get("preempted")),
        })
    out = REPO / "output" / "kv_vram" / "mgc.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print({k: r[k] for k in ("name", "c", "rho", "S_mean_s", "wait_meas_ms", "wait_mmc_ms",
                                 "wait_ll_ms", "wait_replay_ms", "wait_replay_c1_ms")})


if __name__ == "__main__":
    main()
