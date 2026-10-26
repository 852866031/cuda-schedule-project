#!/usr/bin/env python3
"""Join a cell's decode-engine trace (site/kv_trace.py) with its client records.

Per request (measured requests only; warm-up requests are dropped by key):
  index, session, t_submit (client), t_arr (reached the decode scheduler), free blocks
  and running/waiting at arrival, t_sched (admitted), ext_tokens, blocks, t_first (the
  decode engine's first token = regenerated token #1), t_fin, and the client's token
  times. Derived: wait = t_sched - t_arr (the decode queue), load = t_first - t_sched,
  decode = t_fin - t_first, resident = t_fin - t_sched (how long the request held KV).
Per scheduler call ("sch"): running, waiting, free blocks -> <name>_state.csv.
Per step (one "upd" line = one executed batch): t, batch size, tokens, admitted, and
  the wall time since the previous executed batch (dt; NaN across idle gaps > 0.5 s).

    .venv/bin/python scripts/kv_vram/kv_timeline.py output/raw/kv_base_b24_q2_tr.json
    -> output/kv_vram/timeline/<name>_req.csv, <name>_steps.csv, <name>_loads.csv
"""
import csv
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from workload import build_workload    # noqa: E402

OUTDIR = REPO / "output" / "kv_vram" / "timeline"
REQ_BLOCKS = 400


def load_cell(raw_path):
    rec = json.loads(Path(raw_path).read_text())
    ev = []
    with gzip.open(REPO / rec["trace_file"], "rt") as f:
        for line in f:
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rec, ev


def build(rec, ev):
    wl = build_workload(32, 6144, 128, rec["requests"], "zipf", 1.1, 0)
    by_key = {tuple(r.token_ids[6144:6148]): r for r in wl.requests}
    cli = {r["index"]: r for r in rec.get("records", [])}
    reqs, steps, loads, state = {}, [], [], []
    last_upd = None
    for e in ev:
        k = e.get("ev")
        if k == "arr":
            w = by_key.get(tuple(e["key"]))
            if w is None:
                continue                                  # warm-up request
            reqs[e["id"]] = {"id": e["id"], "index": w.index, "session": w.session_id,
                             "t_arr": e["t"], "free_at_arr": e["free"],
                             "run_at_arr": e["run"], "wait_at_arr": e["wait"]}
        elif k == "sch":
            state.append({"t": e["t"], "run": e["run"], "wait": e["wait"], "free": e["free"]})
            for rid, ext, nb in e.get("new", []):
                r = reqs.get(rid)
                if r is not None and "t_sched" not in r:
                    r.update(t_sched=e["t"], ext_tokens=ext, blocks=nb,
                             free_after_sched=e["free"], run_after_sched=e["run"])
            for rid in e.get("pre", []):
                if rid in reqs:
                    reqs[rid]["preempted"] = reqs[rid].get("preempted", 0) + 1
            for rid, ext in e.get("res", []):
                r = reqs.get(rid)
                if r is not None:
                    r["t_resumed"] = e["t"]
                    r["resumed_ext"] = ext
        elif k == "upd":
            dt = None
            if last_upd is not None and e["t"] - last_upd < 0.5:
                dt = e["t"] - last_upd
            last_upd = e["t"]
            steps.append({"t": e["t"], "n": e["n"], "ntok": e["ntok"], "nnew": e["nnew"],
                          "dt": dt})
            for rid in e.get("first", []):
                r = reqs.get(rid)
                if r is not None and "t_first" not in r:
                    r["t_first"] = e["t"]
        elif k == "fin":
            r = reqs.get(e["id"])
            if r is not None:
                r.update(t_fin=e["t"], nout=e["nout"], free_after_fin=e["free"])
        elif k == "load":
            loads.append({"t": e["t"], "dt": e["dt"], "n": e["n"], "ntok": e["ntok"]})
    out = []
    for r in reqs.values():
        c = cli.get(r["index"], {})
        r["t_submit"] = c.get("t_submit")
        r["client_ok"] = c.get("ok")
        if c.get("ttft") is not None:
            r["t_tok1"] = c["t_submit"] + c["ttft"]
            if c.get("itls"):
                r["t_tok2"] = r["t_tok1"] + c["itls"][0]
                r["t_last"] = r["t_tok1"] + sum(c["itls"])
        if "t_sched" in r:
            r["wait"] = r["t_sched"] - r["t_arr"]
            r["free_slots_at_arr"] = r["free_at_arr"] // REQ_BLOCKS
        if "t_first" in r and "t_sched" in r:
            r["load"] = r["t_first"] - r["t_sched"]
        if "t_fin" in r and "t_first" in r:
            r["decode"] = r["t_fin"] - r["t_first"]
        if "t_fin" in r and "t_sched" in r:
            r["resident"] = r["t_fin"] - r["t_sched"]
        out.append(r)
    out.sort(key=lambda r: r["index"])
    return out, steps, loads, state


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for p in sys.argv[1:]:
        rec, ev = load_cell(p)
        reqs, steps, loads, state = build(rec, ev)
        n = rec["name"]
        write_csv(OUTDIR / f"{n}_req.csv", reqs)
        write_csv(OUTDIR / f"{n}_steps.csv", steps)
        write_csv(OUTDIR / f"{n}_loads.csv", loads)
        write_csv(OUTDIR / f"{n}_state.csv", state)
        done = [r for r in reqs if "resident" in r]
        print(f"{n}: {len(reqs)} measured requests traced ({len(done)} complete), "
              f"{len(steps)} steps, {len(loads)} load calls, "
              f"{sum(1 for e in ev if e.get('ev') == 'skip')} skip lines")


if __name__ == "__main__":
    main()
