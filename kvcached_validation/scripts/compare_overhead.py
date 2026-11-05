#!/usr/bin/env python
"""Compare two load runs (e.g. plain vs kvcached at the same static budget) on per-token step
time. Steps that include a 6k-token prefill chunk take ~0.5 s and would dominate the tail, so the
decode-only step time is reported separately as the median/p90 of per-token ITLs below 100 ms.
  compare_overhead.py --a plain results/expOH_plain_load.jsonl --b kvcached results/expOH_kvcached_load.jsonl
"""
import argparse, json
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--a", nargs=2, metavar=("NAME", "JSONL"), required=True)
ap.add_argument("--b", nargs=2, metavar=("NAME", "JSONL"), required=True)
args = ap.parse_args()

def stats(path):
    reqs = [json.loads(l) for l in open(path)]
    ok = [r for r in reqs if r["status"] == 200 and not r["error"]]
    itl = []
    for r in ok:
        tt, tn = r["tok_t"], r.get("tok_n") or [1] * len(r["tok_t"])
        for i, (a, b) in enumerate(zip(tt, tt[1:])):
            itl.extend([(b - a) / tn[i + 1]] * tn[i + 1])
    itl = np.array(itl); dec = itl[itl < 0.1]
    ttft = np.array([r["tok_t"][0] - r["t_send"] for r in ok if r["tok_t"]])
    span = max(r["t_done"] for r in ok) - min(r["t_send"] for r in ok)
    return dict(n=len(ok), fail=len(reqs) - len(ok), tok=sum(r["n_out"] for r in ok), span=span,
                itl50=np.median(itl), itl90=np.percentile(itl, 90), itl99=np.percentile(itl, 99),
                dec50=np.median(dec), dec90=np.percentile(dec, 90), dec_frac=len(dec) / len(itl),
                ttft50=np.median(ttft), ttft90=np.percentile(ttft, 90))
A, B = stats(args.a[1]), stats(args.b[1])
rows = [("requests ok / failed", "n", "fail"), ("output tokens", "tok", None), ("wall span s", "span", None),
        ("ITL p50 ms (all steps)", "itl50", None), ("ITL p90 ms (all steps)", "itl90", None), ("ITL p99 ms (all steps)", "itl99", None),
        ("decode-only step p50 ms", "dec50", None), ("decode-only step p90 ms", "dec90", None),
        ("share of steps under 0.1 s", "dec_frac", None), ("TTFT p50 s", "ttft50", None), ("TTFT p90 s", "ttft90", None)]
print(f"| metric | {args.a[0]} | {args.b[0]} | ratio {args.b[0]}/{args.a[0]} |\n|---|---|---|---|")
for label, k, k2 in rows:
    a, b = A[k], B[k]
    if k2: print(f"| {label} | {a} / {A[k2]} | {b} / {B[k2]} | |"); continue
    scale = 1000 if "ms" in label else 1
    fmt = (lambda v: f"{v*scale:.2f}") if isinstance(a, float) else (lambda v: f"{v}")
    ratio = f"{b/a:.3f}" if isinstance(a, float) and a else ""
    print(f"| {label} | {fmt(a)} | {fmt(b)} | {ratio} |")
