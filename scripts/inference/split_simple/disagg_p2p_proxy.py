#!/usr/bin/env python3
"""Router for prefill/decode disaggregation with vLLM's P2pNcclConnector.

vLLM's own `disagg_proxy_demo.py` implements a different handshake and does not work with this
connector. `P2pNcclConnector` addresses its peer by **parsing the request id**:

    prefill side:  re.search(r"___decode_addr_(.*):(\\d+)",  request_id)   -> where to send KV
    decode  side:  re.search(r"___prefill_addr_(.*):(\\d+)___", request_id) -> where to pull from

so the router's real job is to mint an id carrying both addresses and use it for both legs.
vLLM takes the id from the `X-Request-Id` header (no server flag needed -- the
`--enable-request-id-headers` flag only controls the *response* header).

**The addresses must be the engines' own ZMQ addresses, and those are never localhost.**
`P2pNcclEngine` binds its ROUTER socket to `f"{get_ip()}:{kv_port}"` -- one interface, the
LAN address -- so `127.0.0.1:22001` is not listening even though the instance is on this
machine. Pointing the id at localhost does not fail loudly: the DEALER queues the "NEW"
handshake forever, and the producer then blocks in `ncclCommInitRank`, which has no
timeout. The prefill engine wedges with nothing in its log but `set_p2p_nccl_context`.
Pass `--kv-host` the value of vLLM's own `get_ip()`; `disagg_launch.sh` does.

Flow per request:
  1. mint  ___prefill_addr_IP:KVPORT___decode_addr_IP:KVPORT_<uuid>
  2. POST to prefill with max_tokens=1  -> populates and ships the KV
  3. POST to decode with the original body -> pulls that KV, generates, streams back

Usage:
    python scripts/decode/disagg_p2p_proxy.py \\
        --prefill 127.0.0.1:8100 --prefill-kv-port 21001 \\
        --decode  127.0.0.1:8200 --decode-kv-port 22001 \\
        --kv-host "$(python -c 'from vllm.utils.network_utils import get_ip; print(get_ip())')" \\
        --port 8000
"""

import argparse
import asyncio
import time
import uuid

import aiohttp
from aiohttp import web

TIMEOUT = aiohttp.ClientTimeout(total=1800)


def make_request_id(cfg) -> str:
    """The id *is* the routing table for this connector."""
    return (
        f"___prefill_addr_{cfg['prefill_kv']}"
        f"___decode_addr_{cfg['decode_kv']}"
        f"_{uuid.uuid4().hex}"
    )


async def handle(request: web.Request) -> web.StreamResponse:
    cfg = request.app["cfg"]
    t_arrive = time.perf_counter()
    async with request.app["gate"]:
        t_admit = time.perf_counter()
        return await _handle(request, cfg, t_arrive, t_admit)


async def _handle(request: web.Request, cfg, t_arrive, t_admit) -> web.StreamResponse:
    body = await request.json()
    rid = make_request_id(cfg)
    headers = {"X-Request-Id": rid, "Content-Type": "application/json"}
    path = request.path

    session: aiohttp.ClientSession = request.app["session"]

    # --- leg 1: prefill. One token, non-streaming; we discard the output and keep the KV.
    prefill_body = dict(body)
    prefill_body["max_tokens"] = 1
    prefill_body["stream"] = False
    if "max_completion_tokens" in prefill_body:
        prefill_body["max_completion_tokens"] = 1
    async with session.post(f"http://{cfg['prefill']}{path}", json=prefill_body,
                            headers=headers) as r:
        if r.status != 200:
            text = await r.text()
            return web.json_response({"error": f"prefill {r.status}: {text[:400]}"},
                                     status=502)
        await r.read()
    t_prefill_done = time.perf_counter()

    # --- leg 2: decode. Same id, so the connector knows which prefill to pull from.
    first_chunk_at = None
    async with session.post(f"http://{cfg['decode']}{path}", json=body,
                            headers=headers) as r:
        resp = web.StreamResponse(status=r.status,
                                  headers={"Content-Type": r.headers.get(
                                      "Content-Type", "application/json")})
        await resp.prepare(request)
        async for chunk in r.content.iter_any():
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter()
            await resp.write(chunk)
        await resp.write_eof()

    # One line per request; this is the authoritative client-side TTFT decomposition.
    # gate = waiting for the in-flight slot, leg1 = the whole prefill request (queue +
    # retrieve + compute + KV export drain), leg2_first = decode admission + KV reload +
    # first step + first chunk back.
    if first_chunk_at is not None:
        print(f"LEGS gate_ms={1000*(t_admit-t_arrive):.1f} "
              f"leg1_ms={1000*(t_prefill_done-t_admit):.1f} "
              f"leg2_first_ms={1000*(first_chunk_at-t_prefill_done):.1f}", flush=True)
    return resp


async def health(request: web.Request) -> web.Response:
    cfg = request.app["cfg"]
    session = request.app["session"]
    out = {}
    for name in ("prefill", "decode"):
        try:
            async with session.get(f"http://{cfg[name]}/health") as r:
                out[name] = r.status
        except Exception as e:
            out[name] = str(e)
    ok = all(v == 200 for v in out.values())
    # The kv addresses are reported so probes and the sweep driver can mint ids that
    # match what the engines actually bound, instead of hardcoding a second copy.
    out["prefill_kv"] = cfg["prefill_kv"]
    out["decode_kv"] = cfg["decode_kv"]
    out["max_inflight"] = cfg["max_inflight"]
    return web.json_response(out, status=200 if ok else 503)


async def on_startup(app):
    app["session"] = aiohttp.ClientSession(timeout=TIMEOUT)
    app["gate"] = asyncio.Semaphore(app["cfg"]["max_inflight"])


async def on_cleanup(app):
    await app["session"].close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", default="127.0.0.1:8100", help="prefill HTTP host:port")
    ap.add_argument("--decode", default="127.0.0.1:8200", help="decode HTTP host:port")
    ap.add_argument("--prefill-kv-port", type=int, default=21001)
    ap.add_argument("--decode-kv-port", type=int, default=22001)
    ap.add_argument("--kv-host", default=None,
                    help="host the engines bound their ZMQ ROUTER sockets to, i.e. vLLM's "
                         "get_ip(). NOT localhost -- see the module docstring. Defaults to "
                         "the HTTP host, which is right only if that is already the LAN ip.")
    ap.add_argument("--max-inflight", type=int, default=28,
                    help="admission control. The consumer keeps a copy of every in-flight "
                         "request's KV until that request finishes, at 1.00 GiB of pinned "
                         "host memory each, and exhausting that pool raises inside the "
                         "connector's listener thread -- killing it and wedging the decode "
                         "engine. Under an open loop nothing else bounds in-flight, so the "
                         "router has to. Keep it a few below the pool size in GiB.")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["cfg"] = {
        "prefill": args.prefill,
        "decode": args.decode,
        "prefill_kv": f"{args.kv_host or args.prefill.split(':')[0]}:{args.prefill_kv_port}",
        "decode_kv": f"{args.kv_host or args.decode.split(':')[0]}:{args.decode_kv_port}",
        "max_inflight": args.max_inflight,
    }
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_post("/v1/completions", handle)
    app.router.add_post("/v1/chat/completions", handle)
    app.router.add_get("/health", health)

    print(f"proxy on :{args.port}  prefill={args.prefill} (kv {app['cfg']['prefill_kv']})  "
          f"decode={args.decode} (kv {app['cfg']['decode_kv']})  "
          f"max_inflight={args.max_inflight}", flush=True)
    web.run_app(app, host="0.0.0.0", port=args.port, print=None)


if __name__ == "__main__":
    main()
