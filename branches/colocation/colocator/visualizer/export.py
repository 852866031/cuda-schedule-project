#!/usr/bin/env python
"""Export a colocated run as a self-contained replay animation (one HTML file).

    python colocator/visualizer/export.py runs/prio -o runs/prio/replay.html

Reads <run_dir>/colocated/{trace.json,results.json} (produced by
run_demo.py --observer on + analyze.py) and injects a compact per-op dataset
into template.html. The output is fully self-contained — no server, no
external requests — so it can be scp'd to a Mac (or anywhere) and opened
directly in a browser, or published as a web page.

    --fragment   also write <out>.fragment.html: the same page without the
                 <!doctype>/<html>/<body> wrapper, for embedding/publishing
                 through tools that supply their own document skeleton.
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def build_data(mode_dir, label, baseline_path=None, nav=None):
    """baseline_path: a seq-mode results.json supplying per-client SOLO
    iteration latencies for the overhead panel. Defaults to the run's own
    seq/ sibling when present.
    nav: {"workloads": [names...], "pairs": {"a|b": "file.html"}} for the
    replay's compare-selectors (export_all builds it across all runs).

    Also accepts a seq-mode dir holding an OBSERVER-ONLY trace (built by
    analyze.py for single-client solo runs): t_intercept is null and t_issue
    (host launch) may be — display times fall back to the GPU start, the page
    gets mode="seq" and hides the colocator pipeline panels."""
    with open(os.path.join(mode_dir, "trace.json")) as f:
        trace = json.load(f)
    with open(os.path.join(mode_dir, "results.json")) as f:
        results = json.load(f)
    seq = results["mode"] == "seq"

    if baseline_path is None and not seq:
        cand = os.path.join(os.path.dirname(mode_dir), "seq", "results.json")
        baseline_path = cand if os.path.exists(cand) else None
    solo_by_name = {}
    if baseline_path and not seq and os.path.exists(baseline_path):
        with open(baseline_path) as f:
            for c in json.load(f)["clients"]:
                solo_by_name[c["name"]] = [round(x * 1e3, 4) for x in c["lat_s"]]

    def host_ts(op):  # earliest host-side timestamp, falling back to GPU start
        return op["t_intercept_ns"] or op["t_issue_ns"] or op["t_gpu_start_ns"]

    # Trim: drop everything before the last "load_done" mark (stamped by
    # workloads with a heavy setup, e.g. llmdecode's 15 GiB weight upload) —
    # otherwise the load dwarfs the interesting part of the timeline.
    loads = [c.get("marks", {}).get("load_done") for c in results["clients"]]
    loads = [x for x in loads if x]
    if loads:
        cut = max(loads)
        trimmed = [op for op in trace
                   if (op["t_gpu_start_ns"] or op["t_issue_ns"]
                       or op["t_intercept_ns"] or 0) >= cut]
        if trimmed:
            trace = trimmed
        else:  # trace predates the marks (stale trace.json) — re-run analyze
            print(f"[export] WARNING: {mode_dir}: load_done mark trims ALL ops "
                  f"(stale trace.json? re-run analyze) — keeping full trace")

    t0 = min(host_ts(op) for op in trace)
    # Round order = issue order (the scheduler serializes; analyze.py already
    # sorts the trace by t_issue — or by GPU start for observer-only traces).
    trace = sorted(trace, key=lambda o: o["t_issue_ns"] or o["t_gpu_start_ns"])

    ops = []
    t_end = 0
    for rnd, op in enumerate(trace):
        is_copy = op["kind"] in ("cudaMemcpy", "cudaMemcpyAsync")
        rec = {
            "i": f'{op["client"]}-{op["op_id"]}',
            "c": op["client"],
            "k": "M" if is_copy else "K",  # cuBLAS ops count as kernels
            "cls": "copy" if is_copy else op["name"],
            "ti": (host_ts(op) - t0) / 1e3,
            "tis": ((op["t_issue_ns"] or op["t_gpu_start_ns"]) - t0) / 1e3,
            "tgs": None if op["t_gpu_start_ns"] is None else (op["t_gpu_start_ns"] - t0) / 1e3,
            "tge": None if op["t_gpu_end_ns"] is None else (op["t_gpu_end_ns"] - t0) / 1e3,
            "st": op.get("gpu_stream"),
            "rnd": rnd,
        }
        if is_copy:
            rec["b"] = op["bytes"]
        else:
            g = op["grid"]  # for cuBLAS ops the slots hold the GEMM m,n,k
            rec["g"] = "x".join(str(v) for v in g if v) or "1"
        ops.append(rec)
        t_end = max(t_end, rec["tge"] or rec["tis"])

    prios = results.get("priorities", [0] * len(results["clients"]))
    clients = [{"name": c["name"], "prio": prios[i], "iters": c.get("iters"),
                "desc": c.get("desc", "")}
               for i, c in enumerate(results["clients"])]
    sm_limit = results.get("sm_limit", 0) or 0
    if sm_limit:
        # Per-client stream ids so the page can split its stream lanes:
        # cuBLAS ops ran on the capped green-context stream, the rest on main.
        for i, c in enumerate(clients):
            capped = {op["gpu_stream"] for op in trace if op["client"] == i
                      and op["kind"] in ("cublasGemmEx", "cublasSgemm")}
            main = {op["gpu_stream"] for op in trace if op["client"] == i
                    and op["kind"] not in ("cublasGemmEx", "cublasSgemm")}
            c["streams"] = {"capped": next(iter(capped), None),
                            "main": next(iter(main), None)}
    policy = "solo" if seq else results.get("policy", "fcfs")
    perf = [{"name": c["name"],
             "coloc_ms": [round(x * 1e3, 4) for x in c["lat_s"]],
             "solo_ms": solo_by_name.get(c["name"])}
            for c in results["clients"]]
    # The moment BOTH clients crossed the start barrier (max of the two,
    # recorded in the same CLOCK_MONOTONIC_RAW domain as the trace): left of
    # it is setup/check/warmup, right of it is the timed experiment.
    barrier_ns = [c.get("barrier_ns") for c in results["clients"]]
    barrier_us = (max(barrier_ns) - t0) / 1e3 if all(barrier_ns) else None
    title = f"{label} · solo (seq, observer only)" if seq else f"{label} · policy {policy}"
    if sm_limit:
        title += f" · SM limit {sm_limit}%"
    return {"title": title, "clients": clients, "mode": results["mode"],
            "policy": policy, "sm_limit": sm_limit, "perf": perf,
            "barrier_us": barrier_us, "nav": nav,
            "t_end_us": round(t_end + 1000), "ops": ops}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", help="run directory containing colocated/")
    ap.add_argument("-o", "--out", default=None,
                    help="output HTML path (default <run_dir>/replay.html)")
    ap.add_argument("--label", default=None, help="title label (default: run dir name)")
    ap.add_argument("--fragment", action="store_true",
                    help="also write <out>.fragment.html without the document wrapper")
    args = ap.parse_args()

    mode_dir = os.path.join(args.run_dir, "colocated")
    label = args.label or os.path.basename(os.path.normpath(args.run_dir))
    data = build_data(mode_dir, label)

    with open(os.path.join(HERE, "template.html")) as f:
        fragment = f.read()
    fragment = fragment.replace("__VIZ_DATA__", json.dumps(data, separators=(",", ":")))
    fragment = fragment.replace("__VIZ_TITLE__", label)

    out = args.out or os.path.join(args.run_dir, "replay.html")
    standalone = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                  f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
                  f"<title>Colocator replay — {label}</title>\n</head>\n<body>\n"
                  + fragment + "\n</body>\n</html>\n")
    with open(out, "w") as f:
        f.write(standalone)
    print(f"[export] {out} ({len(data['ops'])} ops, "
          f"{os.path.getsize(out) // 1024} KiB, self-contained)")

    if args.fragment:
        fout = out.replace(".html", "") + ".fragment.html"
        with open(fout, "w") as f:
            f.write(fragment)
        print(f"[export] {fout}")


if __name__ == "__main__":
    main()
