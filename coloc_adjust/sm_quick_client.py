#!/usr/bin/env python3
"""A quick closed-loop decode load: C requests kept in flight at a time (so the decode batch holds C requests), each a 64-token
random prompt and 200 output tokens, streamed. Per level: median and mean time per output token, from the first to the last
token of a request (requests that start after the first 2 s of the level).
  python sm_quick_client.py --url http://localhost:8000 --out results_sm/q_gpu0_64.json --levels 1,4,8 --secs 15"""
import argparse, asyncio, json, random, statistics, time
import aiohttp
ap = argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--levels", default="1,4,8"); ap.add_argument("--secs", type=float, default=15.0); ap.add_argument("--max-tokens", type=int, default=200)
a = ap.parse_args()
async def one(s, model, rng):
    body = dict(model=model, prompt=[rng.randrange(1000, 100000) for _ in range(64)], max_tokens=a.max_tokens, temperature=0, stream=True, ignore_eos=True)
    ts = []
    async with s.post(a.url + "/v1/completions", json=body) as r:
        async for line in r.content:
            if line.startswith(b"data:") and b"[DONE]" not in line:
                try:
                    if json.loads(line[5:])["choices"][0].get("text") is not None: ts.append(time.time())
                except Exception: pass
    return ts
async def main():
    rng = random.Random(0); out = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as s:
        async with s.get(a.url + "/v1/models") as r: model = (await r.json())["data"][0]["id"]
        await one(s, model, rng)   # warm-up
        for C in [int(x) for x in a.levels.split(",")]:
            t0 = time.time(); res = []
            async def worker():
                while time.time() - t0 < a.secs:
                    st = time.time(); ts = await one(s, model, rng)
                    if st - t0 >= 2.0 and len(ts) > 2: res.append((ts[-1] - ts[0]) / (len(ts) - 1) * 1e3)
            await asyncio.gather(*[worker() for _ in range(C)])
            out[C] = dict(n=len(res), median_ms=statistics.median(res) if res else None, mean_ms=statistics.fmean(res) if res else None)
            print(C, out[C], flush=True)
    json.dump(out, open(a.out, "w"))
asyncio.run(main())
