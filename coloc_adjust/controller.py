#!/usr/bin/env python
"""Router + VRAM scheduler for the colocation VRAM-adjust prototype (design §3.2–3.3).

The router is a streaming HTTP pass-through in front of vLLM. It keeps a table of every
in-flight request (prompt tokens, max_tokens, tokens generated so far, admitted or waiting)
and derives the engine's KV block demand from it — nothing is read from vLLM internals. The
VRAM scheduler task ticks every --tick seconds, turns the demand into a KV limit for kvcached
(inference first: grows immediately, shrinks with hysteresis) and publishes what is left of
the GPU as the tenant's budget (/dev/shm/coloc_ft_budget).

    .venv-main/bin/python coloc_adjust/controller.py --listen 8001 --upstream http://localhost:8000 \
        --ipc vllm_a --log results/c_controller.csv            # adjusting
    ... --no-adjust                                             # router + logging only (decode-alone and static-split schemes)
    ... --no-adjust --fixed-budget-gib 6                        # static split: constant tenant budget
"""
import argparse, asyncio, csv, json, math, os, sys, time, urllib.request
from collections import deque
from aiohttp import web, ClientSession, ClientTimeout, TCPConnector

GiB = 1024 ** 3
ap = argparse.ArgumentParser()
ap.add_argument("--listen", type=int, default=8001)
ap.add_argument("--upstream", default="http://localhost:8000")
ap.add_argument("--ipc", default="vllm_a")
ap.add_argument("--model-path", default=None, help="tokenizer for text prompts (default: upstream model)")
ap.add_argument("--block-tokens", type=int, default=16)
ap.add_argument("--block-bytes", type=int, default=2 * 1024 * 1024, help="KV bytes per block, all layers (Llama-3-8B: 2 MiB)")
ap.add_argument("--total-gib", type=float, default=31.3536)
ap.add_argument("--engine-fixed-gib", type=float, default=17.7)
ap.add_argument("--safety-gib", type=float, default=1.0)
ap.add_argument("--min-limit-gib", type=float, default=2.0)
ap.add_argument("--max-limit-gib", type=float, default=12.65)
ap.add_argument("--headroom-gib", type=float, default=0.5)
ap.add_argument("--t-react", type=float, default=1.0, help="look-ahead horizon for arrivals (tenant release time)")
ap.add_argument("--hold-s", type=float, default=5.0, help="seconds a lower need must persist before shrinking")
ap.add_argument("--shrink-window", type=float, default=0.0, help="if > 0: shrink only when the MAX need over this many seconds is below limit - margin (smoother than the instantaneous hold)")
ap.add_argument("--shrink-margin-gib", type=float, default=1.0)
ap.add_argument("--tick", type=float, default=0.1)
ap.add_argument("--no-adjust", action="store_true")
ap.add_argument("--fixed-budget-gib", type=float, default=None, help="publish a constant tenant budget (static split)")
ap.add_argument("--budget-file", default="/dev/shm/coloc_ft_budget")
ap.add_argument("--state-file", default="/dev/shm/coloc_ft_state")
ap.add_argument("--log", required=True)
ap.add_argument("--sm-table", default="", help="tenant SM share by arrival rate, 'rps:sms,...' interpolated (e.g. '0:128,1:64,2:32,3:16'); empty = not published")
args = ap.parse_args()

try:
    from kvcached.control import set_instance_memory_limit, get_instance_memory_limit
    from kvcached.cli.utils import get_kv_cache_limit
    HAVE_KVC = True
except Exception as e:
    print("kvcached API unavailable:", e, file=sys.stderr); HAVE_KVC = False

# ---------------------------------------------------------------- request table
table = {}          # rid -> dict(prompt, max_tokens, gen, t_arrive, t_first)
arrivals = deque()  # timestamps of arrivals (for lambda-hat)
done_count = 0
_tok = None
def count_tokens(prompt):
    global _tok
    if isinstance(prompt, list):
        return len(prompt) if not prompt or isinstance(prompt[0], int) else sum(len(p) for p in prompt)
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(args.model_path or MODEL_NAME)
    return len(_tok(prompt)["input_ids"])

def blocks(n): return math.ceil(n / args.block_tokens)

def demand():
    """Block demand from the table (KV-slots model: admit = prompt blocks, +1 block per 16 generated)."""
    in_use = run_final = wait_admit = wait_final = 0; running = waiting = 0
    for r in table.values():
        final = blocks(r["prompt"] + r["max_tokens"])
        if r["t_first"] is not None or r["admitted"]:
            running += 1; in_use += blocks(r["prompt"] + r["gen"]); run_final += final
        else:
            waiting += 1; wait_admit += blocks(r["prompt"]); wait_final += final
    return dict(running=running, waiting=waiting, blocks_in_use=in_use, blocks_running_final=run_final,
                blocks_waiting_admit=wait_admit, blocks_waiting_final=wait_final)

# ---------------------------------------------------------------- router
MODEL_NAME = None
session = None
async def proxy_completions(request: web.Request):
    global done_count
    body = await request.read()
    try:
        j = json.loads(body)
    except Exception:
        return web.Response(status=400, text="bad json")
    rid = f"r{int(time.time()*1e6)}_{len(table)}"
    rec = dict(prompt=count_tokens(j.get("prompt", "")), max_tokens=int(j.get("max_tokens", 16)), gen=0,
               t_arrive=time.time(), t_first=None, admitted=False)
    table[rid] = rec; arrivals.append(rec["t_arrive"])
    stream = bool(j.get("stream", False))
    try:
        async with session.post(f"{args.upstream}/v1/completions", data=body,
                                headers={"Content-Type": "application/json"}) as up:
            if not stream:
                data = await up.read()
                try:
                    u = json.loads(data).get("usage", {}); rec["gen"] = int(u.get("completion_tokens", 0))
                except Exception: pass
                return web.Response(status=up.status, body=data, content_type="application/json")
            resp = web.StreamResponse(status=up.status, headers={"Content-Type": up.headers.get("Content-Type", "text/event-stream")})
            await resp.prepare(request)
            buf = b""
            async for chunk in up.content.iter_any():
                await resp.write(chunk)
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.startswith(b"data:"):
                        d = line[5:].strip()
                        if d and d != b"[DONE]":
                            try:
                                c = json.loads(d)["choices"][0]
                                n = len(c.get("token_ids") or []) or (1 if c.get("text") else 0)
                                if n and rec["t_first"] is None: rec["t_first"] = time.time()
                                rec["gen"] += n
                            except Exception:
                                pass
            await resp.write_eof()
            return resp
    finally:
        table.pop(rid, None); done_count += 1

async def proxy_any(request: web.Request):
    body = await request.read()
    async with session.request(request.method, f"{args.upstream}{request.rel_url}", data=body,
                               headers={k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}) as up:
        data = await up.read()
        return web.Response(status=up.status, body=data, content_type=up.content_type)

# ---------------------------------------------------------------- VRAM scheduler
def read_json(path):
    try:
        with open(path) as f: return json.load(f)
    except Exception: return None

def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f)
    os.replace(tmp, path)

SM_TABLE = sorted((float(a), float(b)) for a, b in (x.split(":") for x in args.sm_table.split(","))) if args.sm_table else None
def sm_share(lam):
    if not SM_TABLE: return None
    xs, ys = [a for a, _ in SM_TABLE], [b for _, b in SM_TABLE]
    def q(v): return max(8, int(round(v / 16.0)) * 16)   # quantise: fewer context switches
    if lam <= xs[0]: return q(ys[0])
    if lam >= xs[-1]: return q(ys[-1])
    for (x0, y0), (x1, y1) in zip(SM_TABLE, SM_TABLE[1:]):
        if x0 <= lam <= x1: return q(y0 + (y1 - y0) * (lam - x0) / (x1 - x0))
    return q(ys[-1])

def kvc_state():
    used = limit = None
    if HAVE_KVC:
        try:
            info = get_kv_cache_limit(args.ipc)
            if info is not None: used = info.used_size
            st = get_instance_memory_limit(args.ipc)
            lims = [g.get("effective_limit_bytes") or g.get("limit_bytes") for g in st.get("gpus", [])]
            if lims and lims[0]: limit = int(lims[0])
            if used is None and st.get("gpus"): used = int(st["gpus"][0].get("charged_bytes", 0))
        except Exception:
            pass
    return used, limit

def metrics():
    out = {}
    try:
        txt = urllib.request.urlopen(f"{args.upstream}/metrics", timeout=0.3).read().decode()
        for line in txt.splitlines():
            for k in ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:request_queue_time_seconds_sum",
                      "vllm:request_queue_time_seconds_count", "vllm:num_preemptions_total", "vllm:request_success_total"):
                if line.startswith(k + "{") or line.startswith(k + " "):
                    out[k.split(":")[1]] = float(line.rsplit(" ", 1)[-1])
    except Exception:
        pass
    return out

async def control_loop(app):
    total = args.total_gib * GiB; fixed = args.engine_fixed_gib * GiB; safety = args.safety_gib * GiB
    limit = None; seq = 0; below_since = None; last_metrics = {}; last_m_t = 0; pending_budget_up = None
    last_sms = None; need = 0; need_hist = deque()
    def publish(now, budget, limit, need, lam):
        nonlocal seq, last_sms
        seq += 1; last_sms = sm_share(lam)
        write_json(args.budget_file, dict(seq=seq, ts=now, bytes=int(budget), limit=limit, need=need, lam=lam, sms=last_sms))
    f = open(args.log, "w", newline=""); w = csv.writer(f)
    w.writerow(["t", "running", "waiting", "blocks_in_use", "blocks_running_final", "blocks_waiting_admit", "blocks_waiting_final",
                "lambda_hat", "need_gib", "limit_gib", "kvc_used_gib", "kvc_limit_gib", "ft_budget_gib", "ft_tier", "ft_footprint_gib",
                "ft_seq_ack", "action", "m_running", "m_waiting", "m_queue_sum", "m_queue_count", "m_preempt"])
    # initial limit from kvcached
    used0, lim0 = kvc_state()
    limit = lim0 if lim0 else int(args.max_limit_gib * GiB)
    budget = (args.fixed_budget_gib * GiB) if args.fixed_budget_gib is not None else max(0, total - fixed - safety - limit)
    need = 0; publish(time.time(), budget, limit, 0, 0.0)
    while True:
        await asyncio.sleep(args.tick)
        now = time.time()
        while arrivals and arrivals[0] < now - 5.0: arrivals.popleft()
        lam = len(arrivals) / 5.0
        if SM_TABLE and sm_share(lam) != last_sms and int(now * 10) % 10 == 0:   # re-publish the SM share at most 1/s
            publish(now, budget, limit, need, lam)
        d = demand()
        mean_final = (sum(blocks(r["prompt"] + r["max_tokens"]) for r in table.values()) / len(table)) if table else 400
        D = (d["blocks_running_final"] + d["blocks_waiting_final"]) * args.block_bytes
        A = lam * args.t_react * mean_final * args.block_bytes
        need = D + A + args.headroom_gib * GiB
        need = math.ceil(need / (0.5 * GiB)) * (0.5 * GiB)
        need = int(min(max(need, args.min_limit_gib * GiB), args.max_limit_gib * GiB))
        used, kvlim = kvc_state()
        action = ""
        need_hist.append((now, need))
        while need_hist and need_hist[0][0] < now - max(args.shrink_window, 0.1): need_hist.popleft()
        need_for_shrink = max(n for _, n in need_hist) if args.shrink_window > 0 else need
        if not args.no_adjust and HAVE_KVC:
            if need > limit:
                limit = need; below_since = None; action = "grow"
                try: set_instance_memory_limit(int(limit), instance_name=args.ipc)
                except Exception as e: action = f"grow_err:{e!r}"[:40]
                budget_new = max(0, total - fixed - safety - limit)
                if budget_new < budget:   # tenant must release first; publish at once
                    budget = budget_new; publish(now, budget, limit, need, lam)
            elif need_for_shrink < limit - args.shrink_margin_gib * GiB:
                if below_since is None: below_since = now
                floor = (d["blocks_in_use"] + d["blocks_waiting_admit"]) * args.block_bytes + args.headroom_gib * GiB
                if now - below_since >= args.hold_s and need_for_shrink >= floor:
                    limit = max(need_for_shrink, int(math.ceil(floor / (0.5 * GiB)) * 0.5 * GiB)); below_since = None; action = "shrink"
                    try: set_instance_memory_limit(int(limit), instance_name=args.ipc)
                    except Exception as e: action = f"shrink_err:{e!r}"[:40]
                    pending_budget_up = max(0, total - fixed - safety - limit)
            else:
                below_since = None
            # raise the tenant budget only once kvcached's mapped bytes are under the new limit
            if pending_budget_up is not None and (used is None or used <= limit):
                budget = pending_budget_up; pending_budget_up = None; action = (action + "+budget_up").strip("+")
                publish(now, budget, limit, need, lam)
        elif args.no_adjust:
            limit = kvlim or limit
        if now - last_m_t > 1.0:
            last_metrics = metrics(); last_m_t = now
        st = read_json(args.state_file) or {}
        w.writerow([f"{now:.3f}", d["running"], d["waiting"], d["blocks_in_use"], d["blocks_running_final"], d["blocks_waiting_admit"],
                    d["blocks_waiting_final"], f"{lam:.2f}", f"{need/GiB:.2f}", f"{limit/GiB:.2f}",
                    "" if used is None else f"{used/GiB:.2f}", "" if kvlim is None else f"{kvlim/GiB:.2f}", f"{budget/GiB:.2f}",
                    st.get("tier", ""), st.get("footprint_gib", ""), st.get("seq_ack", ""), action,
                    last_metrics.get("num_requests_running", ""), last_metrics.get("num_requests_waiting", ""),
                    last_metrics.get("request_queue_time_seconds_sum", ""), last_metrics.get("request_queue_time_seconds_count", ""),
                    last_metrics.get("num_preemptions_total", "")])
        f.flush()

async def on_startup(app):
    global session, MODEL_NAME
    session = ClientSession(connector=TCPConnector(limit=0), timeout=ClientTimeout(total=None, sock_read=900))
    try:
        async with session.get(f"{args.upstream}/v1/models") as r:
            MODEL_NAME = (await r.json())["data"][0]["id"]
    except Exception:
        MODEL_NAME = args.model_path
    t = asyncio.create_task(control_loop(app))
    def _done(task):
        if not task.cancelled() and task.exception() is not None:
            import traceback; print("CONTROL LOOP DIED:", repr(task.exception()), flush=True); traceback.print_exception(task.exception())
    t.add_done_callback(_done); app["ctl"] = t
    print(f"router on :{args.listen} -> {args.upstream}; model {MODEL_NAME}; adjust={'off' if args.no_adjust else 'on'}", flush=True)

app = web.Application(client_max_size=64 * 1024 * 1024)
app.router.add_post("/v1/completions", proxy_completions)
app.router.add_route("*", "/{tail:.*}", proxy_any)
app.on_startup.append(on_startup)
web.run_app(app, port=args.listen, access_log=None, print=None)
