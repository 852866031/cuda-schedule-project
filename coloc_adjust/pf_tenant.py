#!/usr/bin/env python
"""P3: adapter that makes a second vLLM engine (a small model doing prefill only) a budget-following
tenant, the way ft_worker.py does for training.

It is a streaming HTTP pass-through for the tenant's own requests (:8201 -> :8200) plus a loop
that reads the VRAM scheduler's budget file and turns the budget into
  tier 0  running: kvcached limit of the tenant engine = budget (its in-flight prompts are bounded by it)
  tier 1  paused: budget below --min-kv-gib -> new requests wait at the proxy, in-flight ones finish
  tier 2  asleep (with --sleep): paused for --sleep-after s and idle -> POST /sleep (level 1: weights to
          host, KV discarded); woken with /wake_up when the budget is back above min + --wake-margin-gib
and publishes its footprint to the state file the scheduler reads (fixed footprint + kvcached mapped).

    python pf_tenant.py --listen 8201 --upstream http://localhost:8200 --ipc vllm_c --fixed-gib 4.8 --log results/x_pf.csv
"""
import argparse, asyncio, csv, json, os, sys, time, urllib.request
from aiohttp import web, ClientSession, ClientTimeout, TCPConnector

GiB = 1024 ** 3
ap = argparse.ArgumentParser()
ap.add_argument("--listen", type=int, default=8201); ap.add_argument("--upstream", default="http://localhost:8200")
ap.add_argument("--ipc", default="vllm_c")
ap.add_argument("--budget-file", default="/dev/shm/coloc_ft_budget"); ap.add_argument("--state-file", default="/dev/shm/coloc_ft_state")
ap.add_argument("--fixed-gib", type=float, default=4.8, help="tenant engine's footprint outside KV (weights + runtime + context), for the state file")
ap.add_argument("--ctx-gib", type=float, default=0.6, help="what stays on the GPU while asleep")
ap.add_argument("--min-kv-gib", type=float, default=0.5); ap.add_argument("--max-kv-gib", type=float, default=8.0)
ap.add_argument("--sleep", action="store_true"); ap.add_argument("--sleep-after", type=float, default=3.0); ap.add_argument("--wake-margin-gib", type=float, default=1.0)
ap.add_argument("--tick", type=float, default=0.1); ap.add_argument("--log", required=True)
args = ap.parse_args()
from kvcached.control import set_instance_memory_limit
from kvcached.cli.utils import get_kv_cache_limit

st = dict(paused=False, asleep=False, paused_since=None, inflight=0, queued=0, admitted=0, limit=None, seq=0, used=0, tier=0, wakes=0, sleeps=0)
gate = asyncio.Event(); gate.set()
session = None

def kvc_used():
    try:
        info = get_kv_cache_limit(args.ipc); return int(info.used_size) if info is not None else 0
    except Exception: return 0

def read_budget():
    try:
        j = json.load(open(args.budget_file)); return float(j["bytes"]), int(j.get("seq", 0))
    except Exception: return None, 0

def write_state(extra=""):
    try:
        foot = (args.ctx_gib if st["asleep"] else args.fixed_gib) * GiB + st["used"]
        tmp = args.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(dict(seq_ack=st["seq"], tier=st["tier"], cfg=st["tier"], footprint_gib=round(foot / GiB, 3), ts=time.time(), extra=extra,
                           pf=dict(paused=st["paused"], asleep=st["asleep"], inflight=st["inflight"], queued=st["queued"], limit_gib=(st["limit"] or 0) / GiB)), f)
        os.replace(tmp, args.state_file)
    except Exception: pass

async def post(path):
    try:
        async with session.post(f"{args.upstream}{path}") as r: return r.status
    except Exception as e: return f"err:{e!r}"[:40]

async def is_sleeping():
    try:
        async with session.get(f"{args.upstream}/is_sleeping") as r: return bool((await r.json()).get("is_sleeping"))
    except Exception: return False

async def proxy_completions(request: web.Request):
    body = await request.read()
    st["queued"] += 1
    await gate.wait()
    st["queued"] -= 1; st["inflight"] += 1; st["admitted"] += 1
    try:
        async with session.post(f"{args.upstream}/v1/completions", data=body, headers={"Content-Type": "application/json"}) as up:
            if not json.loads(body).get("stream", False):
                data = await up.read(); return web.Response(status=up.status, body=data, content_type="application/json")
            resp = web.StreamResponse(status=up.status, headers={"Content-Type": up.headers.get("Content-Type", "text/event-stream")})
            await resp.prepare(request)
            async for chunk in up.content.iter_any(): await resp.write(chunk)
            await resp.write_eof(); return resp
    finally:
        st["inflight"] -= 1

async def proxy_any(request: web.Request):
    body = await request.read()
    async with session.request(request.method, f"{args.upstream}{request.rel_url}", data=body,
                               headers={k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}) as up:
        data = await up.read(); return web.Response(status=up.status, body=data, content_type=up.content_type)

async def control_loop(app):
    f = open(args.log, "w", newline=""); w = csv.writer(f)
    w.writerow(["ts", "budget_gib", "limit_gib", "kvc_used_gib", "tier", "paused", "asleep", "inflight", "queued", "admitted", "event"])
    quantum = 256 * 2**20
    while True:
        await asyncio.sleep(args.tick)
        budget, seq = read_budget(); st["seq"] = seq; st["used"] = kvc_used(); event = ""
        if budget is None: continue
        want = max(0, min(args.max_kv_gib * GiB, budget)) // quantum * quantum
        if budget >= args.min_kv_gib * GiB:
            if st["asleep"] and budget >= (args.min_kv_gib + args.wake_margin_gib) * GiB:
                t0 = time.time(); r = await post("/wake_up")
                for _ in range(200):
                    if not await is_sleeping(): break
                    await asyncio.sleep(0.05)
                st["asleep"] = False; st["wakes"] += 1; event = f"wake {r} {(time.time()-t0)*1e3:.0f}ms"
            if not st["asleep"]:
                if st["paused"]: st["paused"] = False; st["paused_since"] = None; gate.set(); event = (event + " resume").strip()
                if want != st["limit"] and want > 0:
                    try: set_instance_memory_limit(int(want), instance_name=args.ipc); st["limit"] = want
                    except Exception as e: event = f"limit_err:{e!r}"[:40]
        else:
            if not st["paused"]:
                st["paused"] = True; st["paused_since"] = time.time(); gate.clear(); event = "pause"
                try: set_instance_memory_limit(int(max(quantum, want)), instance_name=args.ipc); st["limit"] = max(quantum, want)
                except Exception as e: event = f"limit_err:{e!r}"[:40]
            if args.sleep and not st["asleep"] and st["inflight"] == 0 and time.time() - st["paused_since"] >= args.sleep_after:
                t0 = time.time(); r = await post("/sleep?level=1"); st["asleep"] = True; st["sleeps"] += 1; event = f"sleep {r} {(time.time()-t0)*1e3:.0f}ms"
        st["tier"] = 2 if st["asleep"] else (1 if st["paused"] else 0)
        write_state(event)
        w.writerow([f"{time.time():.3f}", f"{budget/GiB:.2f}", "" if st["limit"] is None else f"{st['limit']/GiB:.2f}", f"{st['used']/GiB:.2f}", st["tier"],
                    int(st["paused"]), int(st["asleep"]), st["inflight"], st["queued"], st["admitted"], event]); f.flush()

async def on_startup(app):
    global session
    session = ClientSession(connector=TCPConnector(limit=0), timeout=ClientTimeout(total=None, sock_read=900))
    t = asyncio.create_task(control_loop(app))
    def _done(task):
        if not task.cancelled() and task.exception() is not None:
            import traceback; print("PF CONTROL LOOP DIED:", repr(task.exception()), flush=True); traceback.print_exception(task.exception())
    t.add_done_callback(_done); app["ctl"] = t
    print(f"pf_tenant on :{args.listen} -> {args.upstream}; ipc {args.ipc}; sleep={'on' if args.sleep else 'off'}", flush=True)

app = web.Application(client_max_size=64 * 1024 * 1024)
app.router.add_post("/v1/completions", proxy_completions)
app.router.add_route("*", "/{tail:.*}", proxy_any)
app.on_startup.append(on_startup)
web.run_app(app, port=args.listen, access_log=None, print=None)
