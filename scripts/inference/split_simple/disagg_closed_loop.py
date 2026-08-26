#!/usr/bin/env python3
"""Closed-loop load against the disaggregated router: N requests in flight, always.

The inference study drove an open loop (Poisson arrivals), which is right for latency work
and wrong here -- it pins throughput at the offered rate, so throughput can never be a
dependent variable. With a closed loop the decode node runs at its ceiling and measured
throughput *is* its capacity.

Reports decode-side throughput, TTFT, ITL, and the achieved batch size sampled from the
decode instance's own metrics -- the causal chain under test is VRAM -> batch ->
throughput, so the middle term has to be measured, not inferred.

    scripts/decode/disagg_closed_loop.py -n 8 --requests 24 --isl 6144 --osl 128
"""

import argparse
import json
import statistics
import threading
import time
import urllib.request

MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"


def scrape(url, names):
    """Pull a few gauges out of a Prometheus endpoint."""
    out = {}
    try:
        with urllib.request.urlopen(f"{url}/metrics", timeout=5) as r:
            for line in r.read().decode().splitlines():
                for n in names:
                    if line.startswith(n + "{"):
                        out[n] = float(line.rsplit(" ", 1)[1])
    except Exception:
        pass
    return out


class Sampler(threading.Thread):
    """Samples the decode engine's running batch and preemption count while load runs."""

    GAUGES = ["vllm:num_requests_running", "vllm:num_requests_waiting",
              "vllm:num_preemptions_total", "vllm:kv_cache_usage_perc"]

    def __init__(self, url, interval=0.5, prefix=""):
        super().__init__(daemon=True)
        self.url, self.interval, self.samples, self.stop = url, interval, [], False
        self.prefix = prefix

    def run(self):
        while not self.stop:
            s = scrape(self.url, self.GAUGES)
            if s:
                self.samples.append(s)
            time.sleep(self.interval)

    def summary(self):
        if not self.samples:
            return {}
        running = [s.get("vllm:num_requests_running", 0) for s in self.samples]
        busy = [r for r in running if r > 0]
        k = self.prefix
        return {
            f"{k}batch_mean": round(statistics.fmean(busy), 2) if busy else 0.0,
            f"{k}batch_max": max(running),
            # Fraction of samples with anything running: the prefill node is a validity
            # check, not a result -- if it is busy all the time it is the bottleneck and
            # the decode curve is measuring the wrong thing.
            f"{k}busy_frac": round(len(busy) / len(self.samples), 3),
            f"{k}waiting_max": max(s.get("vllm:num_requests_waiting", 0)
                                   for s in self.samples),
            f"{k}kv_usage_max": round(max(s.get("vllm:kv_cache_usage_perc", 0)
                                          for s in self.samples), 4),
            f"{k}preemptions": max(s.get("vllm:num_preemptions_total", 0)
                                   for s in self.samples),
        }


def one_request(url, prompt, osl, timeout):
    """Streaming, so the first chunk gives a TTFT that includes the KV transfer."""
    body = {"model": MODEL, "prompt": prompt, "max_tokens": osl, "temperature": 0.0,
            "stream": True}
    req = urllib.request.Request(
        f"{url}/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    start = time.time()
    ttft, stamps, n = None, [], 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            if not raw.startswith(b"data: ") or raw.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(raw[6:])
            if not chunk["choices"][0]["text"]:
                continue
            now = time.time()
            if ttft is None:
                ttft = now - start
            stamps.append(now)
            n += 1
    itls = [b - a for a, b in zip(stamps, stamps[1:])]
    return {"ttft": ttft, "itl": itls, "tokens": n, "wall": time.time() - start}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--decode-url", default="http://127.0.0.1:8200")
    ap.add_argument("--prefill-url", default="http://127.0.0.1:8100")
    ap.add_argument("-n", "--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=24)
    ap.add_argument("--isl", type=int, default=6144)
    ap.add_argument("--osl", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default=None, help="write the result as json")
    args = ap.parse_args()

    # Unique prompts: a decode node has no reuse to exploit, and a prefix-cache hit would
    # confound the batch-size measurement. Prefix caching is off on both instances too.
    base = "The quick brown fox jumps over the lazy dog. " * max(1, args.isl // 9)
    prompts = [f"Document {i:06d}. " + base for i in range(args.requests)]

    results, errors = [], []
    lock = threading.Lock()
    nxt = iter(range(args.requests))

    def worker():
        while True:
            with lock:
                i = next(nxt, None)
            if i is None:
                return
            try:
                r = one_request(args.url, prompts[i], args.osl, args.timeout)
                with lock:
                    results.append(r)
            except Exception as e:
                with lock:
                    errors.append(f"{type(e).__name__}: {e}")

    sampler = Sampler(args.decode_url)
    prefill_sampler = Sampler(args.prefill_url, prefix="prefill_")
    sampler.start()
    prefill_sampler.start()
    before = scrape(args.decode_url, ["vllm:generation_tokens_total"])
    prefill_before = scrape(args.prefill_url, ["vllm:prompt_tokens_total"])
    t0 = time.time()
    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0
    sampler.stop = prefill_sampler.stop = True
    sampler.join(timeout=2)
    prefill_sampler.join(timeout=2)
    after = scrape(args.decode_url, ["vllm:generation_tokens_total"])
    prefill_after = scrape(args.prefill_url, ["vllm:prompt_tokens_total"])

    ttfts = sorted(r["ttft"] for r in results if r["ttft"] is not None)
    itls = sorted(x for r in results for x in r["itl"])
    tokens = sum(r["tokens"] for r in results)
    pct = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None

    out = {
        "concurrency": args.concurrency, "requests": args.requests,
        "isl": args.isl, "osl": args.osl,
        "completed": len(results), "errors": errors[:5], "error_count": len(errors),
        "elapsed_s": round(elapsed, 2),
        "output_tok_s": round(tokens / elapsed, 1) if elapsed else 0,
        "engine_gen_tok_s": round(
            (after.get("vllm:generation_tokens_total", 0)
             - before.get("vllm:generation_tokens_total", 0)) / elapsed, 1) if elapsed else 0,
        "ttft_p50": round(pct(ttfts, 0.50), 3) if ttfts else None,
        "ttft_p95": round(pct(ttfts, 0.95), 3) if ttfts else None,
        "itl_p50_ms": round(pct(itls, 0.50) * 1000, 2) if itls else None,
        "itl_p95_ms": round(pct(itls, 0.95) * 1000, 2) if itls else None,
        "prefill_tok_s": round(
            (prefill_after.get("vllm:prompt_tokens_total", 0)
             - prefill_before.get("vllm:prompt_tokens_total", 0)) / elapsed, 1)
        if elapsed else 0,
        **sampler.summary(),
        **prefill_sampler.summary(),
    }
    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
