#!/usr/bin/env python3
"""Router for the LMCache-based prefill/decode split.

Far simpler than the P2pNccl router: LMCache keys KV by content (token-chunk hashes), so
the two legs need no shared request id and the router carries no addressing at all. Leg 1
runs the prompt on the prefill instance with max_tokens=1, which stores the prompt KV
into the shared DRAM cache; leg 2 sends the same prompt to the decode instance, whose
connector looks the chunks up and retrieves them instead of prefilling.

Prints one LEGS line per request (gate/leg1/leg2_first) for TTFT decomposition, same
format as the split_simple router.

--forward-first-token changes what the client experiences, not what the engines do.
Prefill's final forward pass already produces token #1 (the leg-1 request samples it);
the naive router throws it away and the client waits out the decode node's KV retrieval
before seeing anything. With forwarding on, that token is streamed to the client the
moment leg 1 returns -- client TTFT becomes proxy + prefill, as in a colocated server --
and the decode leg's regenerated duplicate of token #1 is dropped so the client still
sees exactly N tokens. The KV retrieval cost does not disappear: it moves into the gap
between tokens #1 and #2 (the first ITL). Requires temperature 0, where the regenerated
token is identical by construction; the reference workload uses temperature 0.
"""

import argparse
import asyncio
import json
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
            leg1 = await r.json()
        t_prefill_done = time.perf_counter()

        forward = cfg["forward_first_token"]
        resp = None
        if forward:
            # Token #1 already exists -- hand it to the client before the decode leg
            # even starts. This chunk is what stops the client's TTFT clock.
            resp = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            first_tok = {"id": leg1.get("id", "fwd"), "object": "text_completion",
                         "created": leg1.get("created", 0),
                         "model": leg1.get("model", body.get("model", "")),
                         "choices": [{"index": 0,
                                      "text": leg1["choices"][0]["text"],
                                      "logprobs": None, "finish_reason": None}]}
            await resp.write(f"data: {json.dumps(first_tok)}\n\n".encode())
        t_forwarded = time.perf_counter()

        first_chunk_at = None
        dropped_dup = not forward   # with forwarding, drop leg 2's duplicate token #1
        async with session.post(f"http://{cfg['decode']}{path}", json=body) as r:
            if resp is None:
                resp = web.StreamResponse(
                    status=r.status,
                    headers={"Content-Type": r.headers.get("Content-Type",
                                                           "application/json")})
                await resp.prepare(request)
            async for raw in r.content:
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                if not dropped_dup and raw.startswith(b"data: ") and \
                        raw.strip() != b"data: [DONE]":
                    try:
                        if json.loads(raw[6:])["choices"][0]["text"]:
                            dropped_dup = True
                            continue   # the regenerated token #1
                    except (ValueError, KeyError, IndexError):
                        pass
                await resp.write(raw)
            await resp.write_eof()

        if first_chunk_at is not None:
            fwd_part = (f"first_tok_ms={1000*(t_forwarded-t_admit):.1f} "
                        if forward else "")
            print(f"LEGS gate_ms={1000*(t_admit-t_arrive):.1f} "
                  f"leg1_ms={1000*(t_prefill_done-t_admit):.1f} {fwd_part}"
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
    ap.add_argument("--forward-first-token", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["cfg"] = {"prefill": args.prefill, "decode": args.decode,
                  "max_inflight": args.max_inflight,
                  "forward_first_token": args.forward_first_token}
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_post("/v1/completions", handle)
    app.router.add_post("/v1/chat/completions", handle)
    app.router.add_get("/health", health)
    print(f"lmcache proxy on :{args.port}  prefill={args.prefill}  "
          f"decode={args.decode}  max_inflight={args.max_inflight}  "
          f"forward_first_token={args.forward_first_token}", flush=True)
    web.run_app(app, host="0.0.0.0", port=args.port, print=None)


if __name__ == "__main__":
    main()
