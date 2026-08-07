#!/usr/bin/env python
"""Export replay HTMLs for ALL runs into one ready-to-share directory.

    python colocator/visualizer/export_all.py                # runs/ -> replays/
    python colocator/visualizer/export_all.py --runs runs --out replays

Scans <runs>/*/colocated/ for traces, exports each as a self-contained
<out>/<run-name>.html (via export.py), and writes <out>/index.html linking
them all. If a run has observer CSVs but no trace.json yet, analyze.py is
invoked automatically to build it. Runs recorded without the observer
(no obs_*.csv) are skipped — there is nothing GPU-side to animate.

The output directory is fully portable: copy it anywhere (e.g. scp to a Mac)
and open index.html.
"""

import argparse
import html
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from export import build_data  # noqa: E402

ANALYZE = os.path.join(HERE, "..", "analysis", "analyze.py")

INDEX_STYLE = """
body { background:#0d0d0d; color:#fff; font:14px/1.5 system-ui,sans-serif; padding:30px; }
h1 { font-size:18px; } .sub { color:#898781; font-size:12px; }
table { border-collapse:collapse; margin-top:14px; }
td, th { padding:6px 16px 6px 0; text-align:left; font-variant-numeric:tabular-nums; }
th { color:#c3c2b7; font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
a { color:#7fd1ff; } tr { border-bottom:1px solid #2c2c2a; }
"""


def ensure_trace(mode_dir, run_dir):
    """Return True if trace.json exists or was successfully built."""
    if os.path.exists(os.path.join(mode_dir, "trace.json")):
        return True
    if not os.path.exists(os.path.join(mode_dir, "obs_kernels.csv")):
        return False  # recorded without observer — nothing to animate
    print(f"[export_all] {run_dir}: no trace.json, running analyze.py ...")
    r = subprocess.run([sys.executable, ANALYZE, run_dir, "--no-self-check"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[export_all]   analyze failed: {r.stderr.strip().splitlines()[-1:]}")
        return False
    return os.path.exists(os.path.join(mode_dir, "trace.json"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default="runs", help="directory containing run dirs (default: runs)")
    ap.add_argument("--out", default="replays", help="output directory (default: replays)")
    ap.add_argument("--baseline", default=None,
                    help="seq-mode results.json for solo latencies in the overhead panel "
                         "(default: each run's own seq/, else <runs>/full/seq)")
    args = ap.parse_args()

    default_baseline = args.baseline
    if not default_baseline:
        for cand in (os.path.join(args.runs, "solo", "seq", "results.json"),
                     os.path.join(args.runs, "full", "seq", "results.json")):
            if os.path.exists(cand):
                default_baseline = cand
                break

    with open(os.path.join(HERE, "template.html")) as f:
        template = f.read()

    os.makedirs(args.out, exist_ok=True)

    # First pass: which runs exist, which pair of workloads each holds, and
    # under which throttle config — builds the compare-selector navigation
    # embedded in every page. Layouts scanned:
    #   <runs>/<pair>/colocated              (legacy, config "throttle_none")
    #   <runs>/<throttle_cfg>/<pair>/colocated   (run_pairs.py layout)
    #   <runs>/<name>/seq                    (single-client solo run with
    #       observer data → observer-only replay, config "solo"; the
    #       multi-workload runs/solo baseline has >1 client and is skipped)
    exportable = []
    skipped = []

    def scan_run(cfg, name, run_dir):
        mode_dir = os.path.join(run_dir, "colocated")
        if not os.path.isdir(mode_dir):
            return False
        if not ensure_trace(mode_dir, run_dir):
            skipped.append(f"{cfg}/{name}")
            return True
        with open(os.path.join(mode_dir, "results.json")) as f:
            cnames = [c["name"] for c in json.load(f)["clients"]]
        exportable.append((cfg, name, run_dir, mode_dir, cnames))
        return True

    def scan_solo(name, run_dir):
        mode_dir = os.path.join(run_dir, "seq")
        res_path = os.path.join(mode_dir, "results.json")
        if not os.path.exists(res_path):
            return
        with open(res_path) as f:
            clients = json.load(f)["clients"]
        if len(clients) != 1:
            return  # multi-workload seq baselines are not replayable
        if not ensure_trace(mode_dir, run_dir):
            return  # recorded without the observer
        exportable.append(("solo", name, run_dir, mode_dir, [clients[0]["name"]]))

    for entry in sorted(os.listdir(args.runs)):
        entry_dir = os.path.join(args.runs, entry)
        if not os.path.isdir(entry_dir):
            continue
        if scan_run("throttle_none", entry, entry_dir):
            continue  # legacy flat layout
        if entry.startswith(("throttle_", "smlimit_")):
            for name in sorted(os.listdir(entry_dir)):
                scan_run(entry, name, os.path.join(entry_dir, name))
        else:
            scan_solo(entry, entry_dir)

    def page_name(cfg, name):
        return f"{name}.{cfg}.html"

    workload_set = sorted({n for *_, cn in exportable for n in cn})
    # Two config axes: throttle_* (op-count cap) and smlimit_* (green-context
    # SM percentage for cuBLAS ops); a combined run's dir is
    # "throttle_<N>+smlimit_<P>". One dropdown per axis; the page recombines
    # the two selections into the full cfg key.
    def cfg_axes(cfg):
        parts = cfg.split("+")
        thr = next((p for p in parts if p.startswith("throttle_")), "throttle_none")
        sm = next((p for p in parts if p.startswith("smlimit_")), None)
        return thr, sm
    non_solo = [cfg for cfg, *_ in exportable if cfg != "solo"]
    cfgs = sorted({cfg_axes(c)[0] for c in non_solo},
                  key=lambda c: (c != "throttle_none", c))
    smcfgs = sorted({sm for c in non_solo if (sm := cfg_axes(c)[1])})
    # "a|b" -> {cfg: filename} for pairs; a lone workload name (no "|") keys
    # its SINGLE-CLIENT colocated runs — the page's "(alone)" selector option.
    pair_map = {}
    for cfg, name, _, _, cn in exportable:
        if len(cn) > 2 or cfg == "solo":  # seq solo pages don't navigate
            continue
        key = "|".join(sorted(cn)) if len(cn) == 2 else cn[0]
        canonical = "__".join(sorted(cn)) if len(cn) == 2 else cn[0]
        slot = pair_map.setdefault(key, {})
        if cfg not in slot or name == canonical:
            slot[cfg] = page_name(cfg, name)
    nav = {"workloads": workload_set, "pairs": pair_map, "cfgs": cfgs,
           "smcfgs": smcfgs}

    rows = []
    for cfg, name, run_dir, mode_dir, cnames in exportable:
        own_seq = os.path.join(run_dir, "seq", "results.json")
        data = build_data(mode_dir, name,
                          own_seq if os.path.exists(own_seq) else default_baseline,
                          nav={**nav, "cur_cfg": cfg})
        page = template.replace("__VIZ_DATA__", json.dumps(data, separators=(",", ":"))) \
                       .replace("__VIZ_TITLE__", name)
        out_path = os.path.join(args.out, page_name(cfg, name))
        with open(out_path, "w") as f:
            f.write("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
                    f"<title>Colocator replay — {name} ({cfg})</title>\n</head>\n<body>\n"
                    + page + "\n</body>\n</html>\n")

        prios = ",".join(str(c["prio"]) for c in data["clients"])
        clients = " + ".join(c["name"] for c in data["clients"])
        rows.append((page_name(cfg, name), f"{name} ({cfg})", clients, prios,
                     data.get("policy", "fcfs"), len(data["ops"]),
                     data["t_end_us"] / 1e3, os.path.getsize(out_path) // 1024))
        print(f"[export_all] {out_path}  ({len(data['ops'])} ops)")

    body = "".join(
        f"<tr><td><a href='{html.escape(fn)}'>{html.escape(label)}</a></td>"
        f"<td>{html.escape(c)}</td><td>{p}</td><td>{html.escape(pol)}</td><td>{o}</td>"
        f"<td>{s:.0f} ms</td><td>{kb} KiB</td></tr>"
        for fn, label, c, p, pol, o, s, kb in rows)
    with open(os.path.join(args.out, "index.html"), "w") as f:
        f.write("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                "<title>Colocator replays</title>\n"
                f"<style>{INDEX_STYLE}</style>\n</head>\n<body>\n"
                "<h1>Colocator replays</h1>\n"
                "<div class='sub'>self-contained animations of colocated runs — "
                "open any link; the whole directory is portable</div>\n"
                "<table><tr><th>run</th><th>clients</th><th>stream priorities</th>"
                f"<th>policy</th><th>ops</th><th>span</th><th>size</th></tr>{body}</table>\n"
                "</body>\n</html>\n")
    print(f"[export_all] {os.path.join(args.out, 'index.html')}  ({len(rows)} runs"
          + (f", skipped: {', '.join(skipped)} — no observer data" if skipped else "") + ")")


if __name__ == "__main__":
    main()
