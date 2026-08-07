#!/usr/bin/env python
"""Build a unified per-op trace from a run and verify/summarize it.

Input: a run directory produced by `run_demo.py --observer on` containing one
or more mode subdirectories (seq/, streams/, colocated/), each with
results.json + obs_*.csv, and (colocated only) issue_log.csv.

For the colocated mode it joins three sources into ONE row per operation:

    issue_log.csv        t_intercept, t_issue      (scheduler, MONOTONIC_RAW)
      |  n-th launch by the scheduler thread
      v
    obs_runtime.csv      correlation id            (CUPTI, scheduler's TID)
      |  same correlation id
      v
    obs_kernels/memcpys  t_gpu_start, t_gpu_end    (CUPTI GPU timestamps)

The join key is ORDER: the scheduler issues ops strictly one at a time, so
its k-th cudaLaunchKernel runtime record (by start time) is the k-th
KernelLaunch row of the issue log; same for cudaMemcpyAsync. CUPTI
timestamps are mapped to CLOCK_MONOTONIC_RAW with the offset from
obs_calib.csv (drift across the run is checked and reported).

Outputs (into <run>/<mode>/):
    trace.json — one record per op with all four timestamps (mono ns)

`--self-check` (default on) asserts, for the colocated mode:
    * every issue-log kernel/memcpy row matched a GPU activity record
    * per op: t_intercept <= t_issue <= t_gpu_start < t_gpu_end (small
      tolerance on the issue->gpu_start edge for clock-mapping error)
    * per client: GPU start times are monotonic in issue order (stream FIFO)
    * zero GPU kernels launched directly by client threads (leak check)

Also prints per-mode metrics: queue delay, launch->gpu-start delay, kernel
durations by kernel class, per-client GPU busy time, A/B concurrency.

Usage:
    python colocator/analysis/analyze.py runs/full
    python colocator/analysis/analyze.py runs/full --self-check
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

CBID_LAUNCH = 211       # CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000
CBID_MEMCPY_ASYNC = 41  # CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020

# issue-log `kind` values that produce a GPU memcpy via cudaMemcpyAsync
MEMCPY_KINDS = ("cudaMemcpy", "cudaMemcpyAsync")
# issue-log `kind` values that produce GPU kernels: a plain launch produces
# exactly one; a replayed cuBLAS call launches 1+ kernels INTERNALLY (from
# the scheduler thread, via cuBLAS's own statically-linked runtime — CUPTI
# still records them, with the scheduler's TID).
KERNEL_KINDS = ("cudaLaunchKernel", "cublasGemmEx", "cublasSgemm")


def kernel_class(mangled):
    """Human label for a mangled kernel name. Order matters (k_copy_tb before
    k_copy). Covers the SGEMM extension, ATen native kernels, the probe
    kernels ported from the gpu-interfere profiler (ext/csrc/probes.cu), and
    the cuBLAS / SDPA kernels an LLM decode step runs (llmdecode)."""
    for tag, label in [("sgemm_tiled", "sgemm"), ("reduce", "reduce"),
                       ("elementwise", "elementwise"), ("Functor", "elementwise"),
                       ("k_sleep", "sleep"), ("k_copy_tb", "blockcopy"),
                       ("k_copy", "streamcopy"), ("k_fma32", "fma32"),
                       ("k_fma64", "fma64")]:
        if tag in mangled:
            return label
    low = mangled.lower()
    for tag, label in [("flash", "attention"), ("fmha", "attention"),
                       ("attention", "attention"),
                       ("nvjet", "gemm"), ("cutlass", "gemm"), ("gemv", "gemm"),
                       ("gemm", "gemm"), ("splitksum", "gemm"),
                       ("softmax", "reduce"), ("norm", "elementwise"),
                       ("catarray", "elementwise"), ("index", "elementwise"),
                       ("gather", "elementwise")]:
        if tag in low:
            return label
    return "other"


class ModeData:
    def __init__(self, mode_dir):
        self.dir = mode_dir
        with open(os.path.join(mode_dir, "results.json")) as f:
            self.results = json.load(f)
        self.mode = self.results["mode"]
        self.window = self.results.get("window")
        self.kernels = self._csv("obs_kernels.csv")
        self.memcpys = self._csv("obs_memcpys.csv")
        self.runtime = self._csv("obs_runtime.csv")
        self.calib = self._csv("obs_calib.csv")
        p = os.path.join(mode_dir, "issue_log.csv")
        self.issue = pd.read_csv(p) if os.path.exists(p) else None

    def _csv(self, name):
        p = os.path.join(self.dir, name)
        return pd.read_csv(p) if os.path.exists(p) else None

    @property
    def cupti_offset(self):
        """mono_ns - cupti_ns offset (mean of calib samples); also drift."""
        c = self.calib
        offs = c.mono_ns - c.cupti_ns
        return offs.mean(), offs.max() - offs.min()

    def to_mono(self, cupti_ts):
        return cupti_ts + self.cupti_offset[0]


# Auxiliary kernels cuBLAS launches as part of the SAME call as the main
# GEMM kernel that precedes them. Name-based and unambiguous: torch/aten
# kernel names never match, so a cuBLAS row can never steal an aten kernel.
CUBLAS_PARTNER_TAGS = ("splitkreduce", "splitk_reduce", "splitksum")


def assign_kernels(ops, kern, rt_start, off0):
    """Map each kernel-producing issue row to the GPU kernel record(s) it
    caused, walking the kernel records in CORRELATION order — correlation
    ids increase with API-call order on the scheduler thread regardless of
    which API launched (runtime, static-runtime inside cuBLAS, or driver),
    and the scheduler is serial, so kernel correlation order == issue order.

    The assignment is DETERMINISTIC: every row consumes exactly one kernel,
    and a cuBLAS row additionally consumes any immediately-following
    partner kernels (CUBLAS_PARTNER_TAGS, e.g. cublasLt::splitKreduce after
    a split-K GEMM). Verified to reconcile exactly (e.g. 96 extra kernels in
    a llmdecode run == 96 splitKreduce records). Host-launch anchors, where
    RUNTIME records exist, are used only as a sanity check plus a rolling
    clock-offset estimate (projection onto each plain row's hard
    [t_issue, t_returned] window — launches can sit anywhere inside it, so
    the offset is never nudged toward a guessed position).
    Returns a list of kernel-index lists, one per row."""
    out = []
    ki, n = 0, len(kern)
    off = off0
    low_names = kern["name"].str.lower().tolist()
    for op in ops.itertuples():
        if ki >= n:
            raise AssertionError(
                f"ran out of GPU kernel records at {op.kind} op_id {op.op_id}")
        if any(t in low_names[ki] for t in CUBLAS_PARTNER_TAGS):
            raise AssertionError(
                f"cuBLAS partner kernel at correlation {kern.correlation.iloc[ki]} "
                f"not preceded by a cuBLAS row (row {op.kind} op_id {op.op_id})")
        take = [ki]
        hs = rt_start.get(kern.correlation.iloc[ki])
        if hs is not None:
            if op.kind == "cudaLaunchKernel":
                mapped = hs + off
                if mapped < op.t_issue_ns:
                    off += op.t_issue_ns - mapped
                elif mapped > op.t_returned_ns:
                    off -= mapped - op.t_returned_ns
            elif not (op.t_issue_ns - 1e6 <= hs + off <= op.t_returned_ns + 1e6):
                raise AssertionError(
                    f"{op.kind} op_id {op.op_id} (client {op.client}): first "
                    f"kernel's host launch outside its issue window — join drifted")
        ki += 1
        if op.kind != "cudaLaunchKernel":  # consume the call's partner kernels
            while ki < n and any(t in low_names[ki] for t in CUBLAS_PARTNER_TAGS):
                take.append(ki)
                ki += 1
        out.append(take)
    if ki != n:
        raise AssertionError(f"{n - ki} GPU kernel records unmatched after join")
    return out


def build_colocated_trace(md):
    """Join issue log + runtime records + GPU activity into one table."""
    sched_tid = md.results["tids"]["sched"]
    off, drift = md.cupti_offset

    issue = md.issue.reset_index(names="row")
    rt = md.runtime[md.runtime.thread_id == sched_tid]
    rows = []

    # -- kernels (incl. cuBLAS-internal ones): correlation-order join --------
    # Kernels caused by the scheduler = all GPU kernel records except those
    # whose launch RUNTIME record carries another thread's TID (e.g. the main
    # thread's context-init kernel). Host launch anchors where records exist.
    rt_all = md.runtime[md.runtime.cbid == CBID_LAUNCH] \
               .drop_duplicates("correlation").set_index("correlation")
    kern = md.kernels.sort_values("correlation").reset_index(drop=True)
    foreign = kern.correlation.map(
        lambda c: c in rt_all.index and int(rt_all.thread_id.loc[c]) != sched_tid)
    kern = kern[~foreign].reset_index(drop=True)
    rt_start = rt_all[rt_all.thread_id == sched_tid].start_ns.to_dict()

    ops = issue[issue.kind.isin(KERNEL_KINDS)].sort_values("t_issue_ns") \
               .reset_index(drop=True)
    idx_lists = assign_kernels(ops, kern, rt_start, off)
    for i in range(len(ops)):
        op = ops.iloc[i]
        gs = [kern.iloc[j] for j in idx_lists[i]]
        g = max(gs, key=lambda r: r.end_ns - r.start_ns)  # longest → class
        rows.append({
            "client": int(op.client),
            "op_id": int(op.op_id),
            "kind": op.kind,
            "name": kernel_class(g["name"]),
            "raw_name": g["name"],
            "n_kernels": len(gs),
            "bytes": int(op.bytes),
            "grid": [int(op.grid_x), int(op.grid_y), int(op.grid_z)],
            "block": [int(op.block_x), int(op.block_y), int(op.block_z)],
            "correlation": int(g.correlation),
            "t_intercept_ns": int(op.t_intercept_ns),
            "t_issue_ns": int(op.t_issue_ns),
            "t_gpu_start_ns": int(min(r.start_ns for r in gs) + off),
            "t_gpu_end_ns": int(max(r.end_ns for r in gs) + off),
            "gpu_stream": int(g.stream),
        })

    # -- memcpys: strict 1:1 order join (unchanged) --------------------------
    api = rt[rt.cbid == CBID_MEMCPY_ASYNC].sort_values("start_ns").reset_index(drop=True)
    ops = issue[issue.kind.isin(MEMCPY_KINDS)].reset_index(drop=True)
    if len(api) != len(ops):
        raise AssertionError(
            f"order-join failed for memcpys: {len(api)} runtime records "
            f"vs {len(ops)} issue-log rows")
    gpu = md.memcpys.set_index("correlation") if md.memcpys is not None else None
    for i in range(len(ops)):
        op = ops.iloc[i]
        corr = api.iloc[i].correlation
        g = gpu.loc[corr] if gpu is not None and corr in gpu.index else None
        rows.append({
            "client": int(op.client),
            "op_id": int(op.op_id),
            "kind": op.kind,
            "name": op.kind,
            "raw_name": "",
            "n_kernels": 0,
            "bytes": int(op.bytes),
            "grid": [int(op.grid_x), int(op.grid_y), int(op.grid_z)],
            "block": [int(op.block_x), int(op.block_y), int(op.block_z)],
            "correlation": int(corr),
            "t_intercept_ns": int(op.t_intercept_ns),
            "t_issue_ns": int(op.t_issue_ns),
            "t_gpu_start_ns": int(g.start_ns + off) if g is not None else None,
            "t_gpu_end_ns": int(g.end_ns + off) if g is not None else None,
            "gpu_stream": int(g.stream) if g is not None else None,
        })

    trace = pd.DataFrame(rows).sort_values("t_issue_ns").reset_index(drop=True)
    return trace, {"cupti_offset_ns": off, "cupti_drift_ns": drift, "sched_tid": sched_tid}


def build_seq_trace(md):
    """Observer-only trace for a SINGLE-CLIENT seq run (no colocator, so no
    issue log): every GPU kernel and memcpy CUPTI saw inside the mode window,
    with the host-side launch time joined by correlation id where a RUNTIME
    record exists. Kernels with no RUNTIME record were launched through the
    CUDA *driver* API. NOTE this count UNDERSTATES what bypasses the
    interceptor: cuBLAS statically links the CUDA runtime, so its launches
    emit RUNTIME records yet never pass through the LD_PRELOAD-interposed
    libcudart PLT (measured on llmdecode: 225 kernels lack RUNTIME records
    but 2130/11819 launches are unseen by the interposer). The ground truth
    for interceptability is the interposer's own passthrough counter."""
    off, drift = md.cupti_offset
    lo, hi = md.window
    rt = (md.runtime.drop_duplicates("correlation").set_index("correlation")
          if md.runtime is not None else None)

    rows, no_rt_kernels = [], 0
    for df, kind in [(md.kernels, "cudaLaunchKernel"), (md.memcpys, "cudaMemcpyAsync")]:
        if df is None:
            continue
        for r in df.itertuples():
            s = r.start_ns + off
            if s < lo or s > hi:
                continue
            launch = None
            if rt is not None and r.correlation in rt.index:
                launch = int(rt.loc[r.correlation].start_ns + off)
            is_k = kind == "cudaLaunchKernel"
            if is_k and launch is None:
                no_rt_kernels += 1
            rows.append({
                "client": 0, "op_id": len(rows), "kind": kind,
                "name": kernel_class(r.name) if is_k else kind,
                "raw_name": r.name if is_k else "",
                "bytes": 0 if is_k else int(r.bytes),
                "grid": [r.grid_x, r.grid_y, r.grid_z] if is_k else [0, 0, 0],
                "block": [r.block_x, r.block_y, r.block_z] if is_k else [0, 0, 0],
                "correlation": int(r.correlation),
                "t_intercept_ns": None,
                "t_issue_ns": launch,          # host launch (RUNTIME record), if any
                "t_gpu_start_ns": int(s),
                "t_gpu_end_ns": int(r.end_ns + off),
                "gpu_stream": int(r.stream),
            })
    trace = pd.DataFrame(rows).sort_values("t_gpu_start_ns").reset_index(drop=True)
    trace["op_id"] = trace.index
    meta = {"cupti_offset_ns": off, "cupti_drift_ns": drift,
            "driver_api_kernels": no_rt_kernels}
    return trace, meta


def self_check(md, trace, meta):
    """Assertions from PROPOSAL.md §7 Phase 4. Raises on failure."""
    problems = []
    drift_ms = meta["cupti_drift_ns"] / 1e6
    tol_ns = max(2e6, 4 * abs(meta["cupti_drift_ns"]))  # >= 2 ms slack for clock mapping

    matched = trace[trace.t_gpu_start_ns.notna()]
    if len(matched) != len(trace):
        problems.append(f"{len(trace) - len(matched)} issue-log ops have no GPU activity record")

    bad_order = trace[(trace.t_intercept_ns > trace.t_issue_ns)]
    if len(bad_order):
        problems.append(f"{len(bad_order)} ops with t_intercept > t_issue")
    m = matched
    bad_gpu = m[m.t_issue_ns - tol_ns > m.t_gpu_start_ns]
    if len(bad_gpu):
        problems.append(f"{len(bad_gpu)} ops with t_issue > t_gpu_start (beyond {tol_ns / 1e6:.1f} ms tolerance)")
    bad_dur = m[m.t_gpu_start_ns >= m.t_gpu_end_ns]
    if len(bad_dur):
        problems.append(f"{len(bad_dur)} ops with t_gpu_start >= t_gpu_end")

    kern = m[m.kind.isin(KERNEL_KINDS)]
    for cl, g in kern.groupby("client"):
        starts = g.sort_values("op_id").t_gpu_start_ns.to_numpy()
        if (np.diff(starts) < 0).any():
            problems.append(f"client {cl}: GPU kernel starts not monotonic in issue order")

    # Leak check: kernels launched by client threads directly = interception bypass.
    # Restrict to this mode's time window (obs dumps are cumulative across modes).
    client_tids = set(md.results["tids"]["clients"])
    rt_by_corr = md.runtime[md.runtime.cbid == CBID_LAUNCH].set_index("correlation")
    kernels = md.kernels
    if md.window:
        off = meta["cupti_offset_ns"]
        kernels = kernels[(kernels.start_ns + off >= md.window[0])
                          & (kernels.start_ns + off <= md.window[1])]
    leaked = 0
    external = 0
    for corr in kernels.correlation:
        tid = int(rt_by_corr.loc[corr].thread_id) if corr in rt_by_corr.index else -1
        if tid in client_tids:
            leaked += 1
        elif tid != meta["sched_tid"]:
            external += 1  # e.g. main-thread context-init kernels: expected
    if leaked:
        problems.append(f"{leaked} GPU kernels launched directly by client threads (leak!)")

    print(f"[self-check] {len(trace)} ops joined, clock drift over run: {drift_ms:.3f} ms, "
          f"external (non-client, non-sched) kernels: {external}")
    if problems:
        for p in problems:
            print(f"[self-check] FAIL: {p}")
        raise AssertionError("self-check failed")
    print("[self-check] PASS: all ops matched, timestamps ordered, per-client GPU "
          "order monotonic, zero leaked client kernels")


def busy_intervals(df):
    """Merge [start,end) GPU intervals -> total busy ns and merged list."""
    iv = sorted(zip(df.t_gpu_start_ns, df.t_gpu_end_ns))
    merged = []
    for s, e in iv:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s for s, e in merged), merged


def overlap_ns(a, b):
    """Intersection time of two merged interval lists."""
    total, i, j = 0, 0, 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if s < e:
            total += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def metrics_colocated(trace):
    m = trace[trace.t_gpu_start_ns.notna()].copy()
    m["queue_delay_us"] = (m.t_issue_ns - m.t_intercept_ns) / 1e3
    m["launch_to_start_us"] = (m.t_gpu_start_ns - m.t_issue_ns) / 1e3
    m["dur_us"] = (m.t_gpu_end_ns - m.t_gpu_start_ns) / 1e3

    print("\n== colocated: per-op stage delays (us) ==")
    for cl, g in m.groupby("client"):
        print(f" client {cl}: queue-delay p50={g.queue_delay_us.median():8.1f} "
              f"p95={g.queue_delay_us.quantile(0.95):9.1f} | "
              f"launch->gpu-start p50={g.launch_to_start_us.median():8.1f} "
              f"p95={g.launch_to_start_us.quantile(0.95):9.1f}")

    print("\n== colocated: kernel durations by class (us) ==")
    kern = m[m.kind.isin(KERNEL_KINDS)]
    for (cl, name), g in kern.groupby(["client", "name"]):
        print(f" client {cl} {name:>12}: n={len(g):4d} p50={g.dur_us.median():9.1f} "
              f"p95={g.dur_us.quantile(0.95):9.1f}")

    busy = {}
    for cl, g in kern.groupby("client"):
        busy[cl], _ = busy_intervals(g)
    if len(busy) == 2:
        _, iv0 = busy_intervals(kern[kern.client == 0])
        _, iv1 = busy_intervals(kern[kern.client == 1])
        ov = overlap_ns(iv0, iv1)
        span = (kern.t_gpu_end_ns.max() - kern.t_gpu_start_ns.min())
        union = busy[0] + busy[1] - ov
        print(f"\n== colocated: GPU concurrency (kernels only) ==")
        print(f" client 0 busy {busy[0] / 1e6:9.1f} ms | client 1 busy {busy[1] / 1e6:9.1f} ms "
              f"| span {span / 1e6:9.1f} ms")
        print(f" overlap {ov / 1e6:9.1f} ms  ({100 * ov / max(union, 1):.1f}% of busy-union, "
              f"{100 * ov / max(min(busy.values()), 1):.1f}% of smaller client's busy time)")
    return m


def kernel_durations_by_mode(md):
    """Solo-vs-shared duration comparison input: durations per kernel class."""
    if md.kernels is None:
        return None
    k = md.kernels.copy()
    if md.window:
        off, _ = md.cupti_offset
        lo, hi = md.window
        k = k[(k.start_ns + off >= lo) & (k.start_ns + off <= hi)]
    k["name_class"] = k["name"].map(kernel_class)
    k["dur_us"] = (k.end_ns - k.start_ns) / 1e3
    return k.groupby("name_class").dur_us.agg(["count", "median",
                                               lambda s: s.quantile(0.95)]) \
            .rename(columns={"<lambda_0>": "p95"})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--self-check", action="store_true", default=True)
    ap.add_argument("--no-self-check", dest="self_check", action="store_false")
    args = ap.parse_args()

    modes = [d for d in ("seq", "streams", "colocated")
             if os.path.isdir(os.path.join(args.run_dir, d))]
    if not modes:
        sys.exit(f"no mode directories under {args.run_dir}")

    per_mode_durs = {}
    for mode in modes:
        md = ModeData(os.path.join(args.run_dir, mode))
        if md.kernels is not None:
            per_mode_durs[mode] = kernel_durations_by_mode(md)
        if mode == "seq" and md.kernels is not None and len(md.results["clients"]) == 1:
            # Solo run with observer data: build an observer-only trace so the
            # visualizer can replay it (no issue log to join — GPU truth only).
            trace, meta = build_seq_trace(md)
            out = os.path.join(md.dir, "trace.json")
            trace.to_json(out, orient="records", indent=1)
            nk = int((trace.kind == "cudaLaunchKernel").sum())
            print(f"[analyze] wrote {out} ({len(trace)} ops: {nk} kernels, "
                  f"{len(trace) - nk} memcpys; {meta['driver_api_kernels']} kernels "
                  f"without RUNTIME records — pure driver-API launches; statically-"
                  f"linked-runtime launches (cuBLAS) also bypass the interceptor)")
        if mode != "colocated" or md.issue is None:
            continue

        trace, meta = build_colocated_trace(md)
        out = os.path.join(md.dir, "trace.json")
        trace.to_json(out, orient="records", indent=1)
        print(f"[analyze] wrote {out} ({len(trace)} ops)")
        if args.self_check:
            self_check(md, trace, meta)
        metrics_colocated(trace)

    if len(per_mode_durs) > 1:
        print("\n== kernel duration by mode (interference; us, GPU-side) ==")
        table = pd.concat(per_mode_durs, axis=1)
        print(table.round(1).to_string())


if __name__ == "__main__":
    main()
