#!/usr/bin/env python
"""How many bytes each segment of a tenant step moves to and from GPU memory.

A segment is the work between two checkpoints of ft_worker.py (block hooks, the checkpoints of head_loss.py).
The tenant runs alone with --nvtx (it synchronises and emits an NVTX mark at every checkpoint) under
Nsight Systems, which samples the device's memory read and write throughput 10,000 times a second
(`nsys profile --gpu-metrics-devices`, no sudo needed on this machine). The throughput is integrated between
consecutive marks and averaged over the steps. The counters are in % of a peak; a known workload (copies of a
1 GiB tensor) gives GB/s per 1 % (calibration, kept in <out>_calib.json).

  SCRIPT=segment_bytes.py GPU=1 ./run_knee_corun.sh [--variants base,split,vocab8192] [--sms 32] [--seconds 14]
  (MPS daemon and GPU selection as for the co-runs; the tenant sits on a green context of --sms SMs)

Writes results_prof/seg_<variant>.csv (one row per segment, in step order, with the running total used to cut
a step into chunks of equal bytes), seg_<variant>_trace.csv and _marks.csv (the samples and the checkpoints of
one step), results_prof/seg_summary.csv and .md. Raw captures are deleted (--keep)."""
import argparse, bisect, csv, json, os, re, sqlite3, statistics as st, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = {"base": [], "split": ["--head", "split"], "compile": ["--compile"], "compile_split": ["--compile", "--head", "split"],
            "cb": ["--compile-blocks", "--no-slow-shapes"], "cb_split": ["--compile-blocks", "--no-slow-shapes", "--head", "split"]}      # cb = each block compiled on its own
def variant_flags(v):
    if v in VARIANTS: return VARIANTS[v]
    m = re.fullmatch(r"(compile_|cb_)?vocab(\d+)(f?)", v)
    if m: return ({"compile_": ["--compile"], "cb_": ["--compile-blocks", "--no-slow-shapes"]}.get(m.group(1), [])) + ["--head", "vocab", "--vocab-block", m.group(2)] + (["--head-fused"] if m.group(3) else [])
    raise SystemExit(f"unknown variant {v}")
ap = argparse.ArgumentParser()
ap.add_argument("--variants", default="base,split"); ap.add_argument("--sms", type=int, default=32, help="tenant green context (0 = full device)")
ap.add_argument("--tier", type=int, default=2, help="tenant shape (TIERS index; 2 = 4 micro-batches, the usual one). Other shapes are written as <variant>_t<tier>");
ap.add_argument("--seconds", type=float, default=14.0); ap.add_argument("--skip-steps", type=int, default=8, help="steps left out at the start")
ap.add_argument("--out", default=os.path.join(HERE, "results_prof", "seg")); ap.add_argument("--nsys", default="/usr/local/cuda-12.8/bin/nsys")
ap.add_argument("--compile-wait", type=int, default=150, help="seconds before the capture starts for a compiled variant"); ap.add_argument("--freq", type=int, default=10000); ap.add_argument("--recalibrate", action="store_true"); ap.add_argument("--keep", action="store_true")
args = ap.parse_args()
os.makedirs(os.path.dirname(args.out), exist_ok=True)

from nsys_tools import Capture, drop
import nsys_tools
def capture(tag, cmd, env=None, delay=0): return nsys_tools.capture(args.out, tag, cmd, env=env, delay=delay, nsys=args.nsys, freq=args.freq)
def calibrate(): return nsys_tools.calibrate(args.out, recalibrate=args.recalibrate, keep=args.keep, nsys=args.nsys, freq=args.freq)

# ---------------------------------------------------------------- one variant
def measure(v0, K):
    v = v0 if args.tier == 2 else f"{v0}_t{args.tier}"          # the name the files and the summary row carry
    ftcsv = f"{args.out}_{v}_ft.csv"
    if os.path.exists(ftcsv): os.remove(ftcsv)
    cmd = [sys.executable, os.path.join(HERE, "ft_worker.py"), "--nvtx", "--out", ftcsv, "--fresh", "--fixed-budget-gib", "20", "--fixed-tier", str(args.tier), "--duration", str(args.seconds + (args.compile_wait + 30 if ("compile" in v0 or v0.startswith("cb") or v0.endswith("f")) else 0)),
           "--snapshot", "/dev/shm/tenant_bytes_seg_snapshot.pt", "--snapshot-every", "100000000", "--state-file", "/dev/shm/tenant_bytes_seg_state"] + (["--green-sms", str(args.sms)] if args.sms else []) + variant_flags(v0)
    os.makedirs("/dev/shm/segment_bytes_xfer", exist_ok=True)
    db = capture(v, cmd, env=dict(os.environ, COLOC_XFER_DIR="/dev/shm/segment_bytes_xfer"), delay=args.compile_wait if ("compile" in v0 or v0.startswith("cb") or v0.endswith("f")) else 0)   # the compiled tenant compiles every shape first (about 130 s)
    if not os.path.exists(ftcsv) or os.path.getsize(ftcsv) < 200: raise RuntimeError(f"the tenant did not run; see {args.out}_{v}.log")
    cap = Capture(db)
    steps = {}
    for s, m in cap.marks:
        g = re.fullmatch(r"s(\d+)(begin|end|u\d+:(.*))", m)
        if g: steps.setdefault(int(g.group(1)), []).append((s, "begin" if g.group(2) == "begin" else "end" if g.group(2) == "end" else (g.group(3) or "?")))
    ok = {k: ms for k, ms in steps.items() if ms[0][1] == "begin" and ms[-1][1] == "end"}
    if len(ok) <= args.skip_steps + 3: raise RuntimeError(f"{v}: only {len(ok)} complete steps; see {args.out}_{v}.log")
    use = [ms for k, ms in sorted(ok.items())][args.skip_steps:]
    n = st.mode(len(ms) for ms in use); use = [ms for ms in use if len(ms) == n]
    segs = []
    for j in range(n - 1):
        label = use[0][j][1] if use[0][j][1] != "begin" else "before_first_checkpoint"
        r = [cap.pct_s("read", ms[j][0], ms[j + 1][0]) * K["read"] for ms in use]; w = [cap.pct_s("write", ms[j][0], ms[j + 1][0]) * K["write"] for ms in use]
        t = [(ms[j + 1][0] - ms[j][0]) / 1e6 for ms in use]; tot = [a + b for a, b in zip(r, w)]
        segs.append(dict(i=j, label=label, read_gb=st.mean(r), write_gb=st.mean(w), total_gb=st.mean(tot), sd_gb=st.pstdev(tot), ms=st.mean(t)))
    cum = 0.0
    for s in segs: cum += s["total_gb"]; s["cum_gb"] = cum
    with open(f"{args.out}_{v}.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["i", "label", "read_gb", "write_gb", "total_gb", "sd_gb", "ms", "gb_per_s", "cum_gb"])
        for s in segs: wr.writerow([s["i"], s["label"], f"{s['read_gb']:.4f}", f"{s['write_gb']:.4f}", f"{s['total_gb']:.4f}", f"{s['sd_gb']:.4f}", f"{s['ms']:.3f}", f"{s['total_gb'] / s['ms'] * 1e3:.0f}" if s["ms"] > 0.05 else "", f"{s['cum_gb']:.3f}"])
    one = use[len(use) // 2]                                     # one step as a trace: the sampled throughput and where the checkpoints fall
    tr, vr = cap.samples("read", one[0][0], one[-1][0]); tw, vw = cap.samples("write", one[0][0], one[-1][0])
    with open(f"{args.out}_{v}_trace.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["t_ms", "read_gbps", "write_gbps"])
        for a, b, c2 in zip(tr, vr, vw): wr.writerow([f"{(a - one[0][0]) / 1e6:.2f}", f"{b * K['read']:.0f}", f"{c2 * K['write']:.0f}"])
    with open(f"{args.out}_{v}_marks.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["t_ms", "label"])
        for t, l in one: wr.writerow([f"{(t - one[0][0]) / 1e6:.3f}", l])
    ft = [r for r in csv.DictReader(open(ftcsv)) if r.get("tok_s") and r.get("dt_ms")][args.skip_steps:]
    tokens = st.mean(float(r["tok_s"]) * float(r["dt_ms"]) / 1e3 for r in ft)
    step_ms = st.mean((ms[-1][0] - ms[0][0]) / 1e6 for ms in use)
    group = lambda pred: sum(s["total_gb"] for s in segs if pred(s["label"]))
    # a compiled body has no checkpoints: the last head segment then runs on into the whole backward pass of the body
    comp = "compile" in v
    head = group(lambda l: l in ("head:pass1", "head:pass2")) if comp and "vocab" in v else None if comp else group(lambda l: l.startswith("head:")); big = max(segs, key=lambda s: s["total_gb"])
    row = dict(variant=v, sms=args.sms or 170, steps=len(use), segments=len(segs), step_ms=f"{step_ms:.1f}", tok_s=f"{tokens / step_ms * 1e3:.0f}", step_gb=f"{cum:.2f}", read_gb=f"{sum(s['read_gb'] for s in segs):.2f}", write_gb=f"{sum(s['write_gb'] for s in segs):.2f}",
               mb_per_token=f"{cum / tokens * 1e3:.2f}", gb_per_s=f"{cum / step_ms * 1e3:.0f}", blocks_fwd_gb=f"{group(lambda l: l.endswith(':fwd')):.2f}", blocks_bwd_gb=f"{group(lambda l: l.endswith(':bwd')):.2f}", head_gb="" if head is None else f"{head:.2f}",
               largest_segment=big["label"], largest_gb=f"{big['total_gb']:.3f}", largest_ms=f"{big['ms']:.2f}", largest_gb_per_s=f"{big['total_gb'] / big['ms'] * 1e3:.0f}",
               seg_sd_median_pct=f"{st.median(s['sd_gb'] / s['total_gb'] * 100 for s in segs if s['total_gb'] > 0.02):.1f}", loss=next((r["loss"] for r in csv.DictReader(open(ftcsv)) if r.get("step") == "20"), ""))   # the same batch in every variant
    if not args.keep: drop(db)
    return row

K = calibrate(); rows = []
if os.path.exists(args.out + "_summary.csv"):                  # keep the rows of variants not measured this time
    rows = [r for r in csv.DictReader(open(args.out + "_summary.csv")) if r["variant"] not in [x if args.tier == 2 else f"{x}_t{args.tier}" for x in args.variants.split(",")]]
    rows = [{k: v for k, v in r.items() if v not in ("", None)} for r in rows]
for v in ([] if args.variants == "none" else args.variants.split(",")):       # "none": only rewrite the summary from the rows on file
    try: row = measure(v, K)
    except Exception as e: row = dict(variant=v, status=str(e))
    rows.append(row); print(row, flush=True); time.sleep(2)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(args.out + "_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
good = [r for r in rows if r.get("step_gb")]
with open(args.out + "_summary.md", "w") as f:
    f.write(f"# Bytes per segment of a tenant step ({time.strftime('%Y-%m-%d')})\n\nTenant alone, green context of {args.sms or 170} SMs, Nsight Systems GPU metrics at {args.freq} Hz; 1 % = {K['read']:.2f} GB/s read, {K['write']:.2f} GB/s write.\n"
            "In 'blocks' the last block's forward segment also holds whatever follows it without a checkpoint (with the model's own loss: the whole output layer and loss).\n"
            "A compiled tenant has no checkpoints in its body: 'blocks' is 0, and 'head' counts only the two passes over the vocabulary (blank where the head cannot be told apart).\n\n"
            "| variant | step ms | tokens/s | GB per step | MB per token | blocks forward GB | blocks backward GB | head segments GB | largest segment | its GB | its ms | its GB/s | loss at step 20 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in good: f.write(f"| {r['variant']} | {r['step_ms']} | {r['tok_s']} | {r['step_gb']} | {r['mb_per_token']} | {r['blocks_fwd_gb']} | {r['blocks_bwd_gb']} | {r.get('head_gb', '')} | {r['largest_segment']} | {r['largest_gb']} | {r['largest_ms']} | {r['largest_gb_per_s']} | {r['loss']} |\n")
    for r in rows:
        if r.get("status"): f.write(f"\n{r['variant']}: {r['status']}\n")
print(open(args.out + "_summary.md").read())
