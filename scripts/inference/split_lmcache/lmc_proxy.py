#!/usr/bin/env python3
"""Router for the LMCache-based prefill/decode split.

Far simpler than the P2pNccl router: LMCache keys KV by content (token-chunk hashes), so
the two legs need no shared request id and the router carries no addressing at all. Leg 1
runs the prompt on the prefill instance with max_tokens=1, which stores the prompt KV
into the shared DRAM cache; leg 2 sends the same prompt to the decode instance, whose
connector looks the chunks up and retrieves them instead of prefilling.

Prints one LEGS line per request (gate/leg1/leg2_first) for TTFT decomposition, same
format as the split_simple router.
"""

import argparse
import asyncio
import time

import aiohttp
from aiohttp import web

TIMEOUT = aiohttp.ClientTimeout(total=1800)


async def handle(request: web.Request) -> web.StreamResponse:
    cfg = request.app["cfg"]
    t_arrive = time.perf_counter()
    async with request.app["gate"]:
        t_admit = time.perf_counter()
        body = await request.json()
        session: aiohttp.ClientSession = request.app["session"]
        path = request.path

        prefill_body = dict(body, max_tokens=1, stream=False)
        prefill_body.pop("max_completion_tokens", None)
        async with session.post(f"http://{cfg['prefill']}{path}",
                                json=prefill_body) as r:
            if r.status != 200:
                text = await r.text()
                return web.json_response(
                    {"error": f"prefill {r.status}: {text[:400]}"}, status=502)
            await r.read()
        t_prefill_done = time.perf_counter()

        first_chunk_at = None
        async with session.post(f"http://{cfg['decode']}{path}", json=body) as r:
            resp = web.StreamResponse(
                status=r.status,
                headers={"Content-Type": r.headers.get("Content-Type",
                                                       "application/json")})
            await resp.prepare(request)
            async for chunk in r.content.iter_any():
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                await resp.write(chunk)
            await resp.write_eof()

        if first_chunk_at is not None:
            print(f"LEGS gate_ms={1000*(t_admit-t_arrive):.1f} "
                  f"leg1_ms={1000*(t_prefill_done-t_admit):.1f} "
                  f"leg2_first_ms={1000*(first_chunk_at-t_prefill_done):.1f}",
                  flush=True)
        return resp


async def health(request: web.Request) -> web.Response:
    cfg, session = request.app["cfg"], request.app["session"]
    out = {}
    for name in ("prefill", "decode"):
        try:
            async with session.get(f"http://{cfg[name]}/health") as r:
                out[name] = r.status
        except Exception as e:
            out[name] = str(e)
    ok = all(v == 200 for v in out.values())
    out["max_inflight"] = cfg["max_inflight"]
    return web.json_response(out, status=200 if ok else 503)


async def on_startup(app):
    app["session"] = aiohttp.ClientSession(timeout=TIMEOUT)
    app["gate"] = asyncio.Semaphore(app["cfg"]["max_inflight"])


async def on_cleanup(app):
    await app["session"].close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", default="127.0.0.1:8100")
    ap.add_argument("--decode", default="127.0.0.1:8200")
    ap.add_argument("--max-inflight", type=int, default=14)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["cfg"] = {"prefill": args.prefill, "decode": args.decode,
                  "max_inflight": args.max_inflight}
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_post("/v1/completions", handle)
    app.router.add_post("/v1/chat/completions", handle)
    app.router.add_get("/health", health)
    print(f"lmcache proxy on :{args.port}  prefill={args.prefill}  "
          f"decode={args.decode}  max_inflight={args.max_inflight}", flush=True)
    web.run_app(app, host="0.0.0.0", port=args.port, print=None)


if __name__ == "__main__":
    main()
