#!/usr/bin/env python3
"""Open-loop async load generator against a vLLM OpenAI-compatible server.

Open loop (Poisson arrivals at a fixed rate, no think-time feedback) is deliberate: when the
GPU KV tier is too small, requests queue and latency grows. A closed-loop client would
throttle itself and hide exactly the effect we are trying to measure.

Per request it records TTFT, inter-token latencies, end-to-end latency, and output token
count, by streaming the completion and timestamping chunk arrivals.
"""

import asyncio
import json
import time

import aiohttp
import numpy as np


async def _one_request(session, url, model, req, max_tokens, timeout, results, sem=None):
    body = {
        "model": model,
        "prompt": req.token_ids,          # raw token IDs: exact, drift-free prompt length
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,               # every request emits exactly max_tokens
    }
    rec = {
        "index": req.index, "session_id": req.session_id, "prompt_len": req.prompt_len,
        "ttft": None, "e2e": None, "n_chunks": 0, "n_nonempty": 0, "itls": [],
        "ok": False, "error": None,
    }
    t0 = time.perf_counter()
    rec["t_submit"] = t0
    try:
        async with session.post(url, json=body,
                                timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                results.append(rec)
                return
            t_prev = None
            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                # One streaming chunk == one output token, even when the detokenized text is
                # empty. Random-token prompts make the model emit runs of incomplete-UTF-8
                # byte fragments, which the detokenizer renders as "" while still being real
                # tokens; testing truthiness of the text dropped those requests entirely.
                if not choices or "text" not in choices[0]:
                    continue
                if choices[0]["text"]:
                    rec["n_nonempty"] = rec.get("n_nonempty", 0) + 1
                now = time.perf_counter()
                if rec["ttft"] is None:
                    rec["ttft"] = now - t0
                else:
                    rec["itls"].append(now - t_prev)
                t_prev = now
                rec["n_chunks"] += 1
            rec["e2e"] = time.perf_counter() - t0
            rec["ok"] = rec["ttft"] is not None
    except asyncio.TimeoutError:
        rec["error"] = f"timeout after {timeout}s"
        rec["e2e"] = time.perf_counter() - t0
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["e2e"] = time.perf_counter() - t0
    results.append(rec)


async def run_load(base_url, model, requests, qps, max_tokens=128, seed=0,
                   timeout=600.0, label=""):
    """Fire `requests` at Poisson-spaced arrival times. Returns per-request records."""
    rng = np.random.default_rng(seed)
    # Pre-compute arrival offsets so the schedule is independent of server behaviour.
    gaps = rng.exponential(1.0 / qps, size=len(requests)) if qps > 0 else np.zeros(len(requests))
    arrivals = np.cumsum(gaps)

    url = f"{base_url.rstrip('/')}/v1/completions"
    results, tasks = [], []
    connector = aiohttp.TCPConnector(limit=0)  # never let the client be the bottleneck
    t_start = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector) as session:
        for req, at in zip(requests, arrivals):
            delay = at - (time.perf_counter() - t_start)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(
                _one_request(session, url, model, req, max_tokens, timeout, results)))
        await asyncio.gather(*tasks)

    duration = time.perf_counter() - t_start
    return results, duration


def summarize(results, duration, max_tokens):
    """Latency percentiles and throughput. Failed requests are counted, never averaged in."""
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]

    def pct(vals, p):
        return round(float(np.percentile(vals, p)) * 1e3, 2) if vals else None

    ttfts = [r["ttft"] for r in ok]
    e2es = [r["e2e"] for r in ok]
    itls = [x for r in ok for x in r["itls"]]
    # TPOT = per-REQUEST mean gap. With fixed 128-token outputs its mean equals the
    # all-gaps mean, but its percentiles differ from itl_ms (per-gap) whenever waiting
    # concentrates in a few gaps -- which is exactly the split/coloc failure signature.
    tpots = [float(np.mean(r["itls"])) for r in ok if r["itls"]]
    out_tokens = sum(r["n_chunks"] for r in ok)

    return {
        "n_requests": len(results),
        "n_ok": len(ok),
        "n_failed": len(fail),
        "failures": [r["error"] for r in fail][:5],
        "duration_s": round(duration, 2),
        "ttft_ms": {"p50": pct(ttfts, 50), "p90": pct(ttfts, 90),
                    "p95": pct(ttfts, 95), "p99": pct(ttfts, 99),
                    "mean": round(float(np.mean(ttfts)) * 1e3, 2) if ttfts else None},
        "itl_ms": {"p50": pct(itls, 50), "p95": pct(itls, 95),
                   "mean": round(float(np.mean(itls)) * 1e3, 2) if itls else None},
        "tpot_ms": {"p50": pct(tpots, 50), "p95": pct(tpots, 95),
                    "mean": round(float(np.mean(tpots)) * 1e3, 2) if tpots else None},
        "e2e_ms": {"p50": pct(e2es, 50), "p95": pct(e2es, 95),
                   "mean": round(float(np.mean(e2es)) * 1e3, 2) if e2es else None},
        "output_tokens": out_tokens,
        "output_tok_per_s": round(out_tokens / duration, 2) if duration > 0 else None,
        "request_tok_per_s": round(len(ok) / duration, 3) if duration > 0 else None,
        "prompt_tokens": sum(r["prompt_len"] for r in ok),
    }
