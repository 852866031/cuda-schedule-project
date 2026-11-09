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
ap.add_argument("--engine-fixed-gib", type=float, default=None, help="engine non-KV footprint; default: measured from NVML at start (used - kvcached mapped)")
ap.add_argument("--tenant-fixed-gib", type=float, default=2.0, help="tenant overhead outside its budget: CUDA context (~0.8) + allocator floor when paused (~1.2)")
ap.add_argument("--safety-gib", type=float, default=0.5)
ap.add_argument("--gpu", type=int, default=0)
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
ap.add_argument("--tpot-slo", type=float, default=None, help="P1.1: cap the tenant's tokens/s so decode TPOT stays within this fraction above the baseline (e.g. 0.05)")
ap.add_argument("--tpot-ref", default=None, help="phases CSV of the baseline run (columns rps, tpot50) used as the TPOT reference")
ap.add_argument("--slo-period", type=float, default=2.0); ap.add_argument("--slo-gain", type=float, default=4.0)
ap.add_argument("--cap-min", type=float, default=500.0); ap.add_argument("--cap-max", type=float, default=20000.0)
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
completions = deque()  # (t_done, tpot_s) of finished streamed requests (for the TPOT SLO loop)
TPOT_REF = None
def load_tpot_ref(path):
    """baseline TPOT (ms) by arrival rate: mean of the phases at each rate, interpolated in rate."""
    import csv as _csv
    by = {}
    for r in _csv.DictReader(open(path)):
        try: by.setdefault(float(r["rps"]), []).append(float(r["tpot50"]))
        except Exception: pass
    pts = sorted((k, sum(v) / len(v)) for k, v in by.items())
    def f(lam):
        if lam <= pts[0][0]: return pts[0][1]
        if lam >= pts[-1][0]: return pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= lam <= x1: return y0 + (y1 - y0) * (lam - x0) / (x1 - x0)
        return pts[-1][1]
    return f
def tpot_now(now, window=10.0, min_n=8):
    while len(completions) > 200: completions.popleft()
    recent = [v for t, v in completions if t >= now - window]
    if len(recent) < min_n: recent = [v for _, v in list(completions)[-min_n:]]
    if not recent: return None
    recent.sort(); return recent[len(recent) // 2] * 1e3
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
        t_end = time.time()
        if rec["t_first"] is not None and rec["gen"] > 1:
            completions.append((t_end, (t_end - rec["t_first"]) / (rec["gen"] - 1)))
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

if args.tpot_ref: TPOT_REF = load_tpot_ref(args.tpot_ref)
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

try:
    import pynvml; pynvml.nvmlInit(); NVML_H = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
except Exception as e:
    print("NVML unavailable:", e, file=sys.stderr); NVML_H = None
def nvml_mem():
    """(used, free) bytes of the whole GPU, or (None, None)."""
    if NVML_H is None: return None, None
    try:
        m = pynvml.nvmlDeviceGetMemoryInfo(NVML_H); return int(m.used), int(m.free)
    except Exception: return None, None

async def control_loop(app):
    total = args.total_gib * GiB; safety = args.safety_gib * GiB; tenant_fixed = args.tenant_fixed_gib * GiB
    if args.engine_fixed_gib is not None:
        fixed = args.engine_fixed_gib * GiB
    else:   # engine alone on the GPU at this point: non-KV footprint = NVML used - kvcached mapped, median of 2 s
        samples = []
        for _ in range(20):
            u, _ = nvml_mem(); k, _ = kvc_state()
            if u is not None: samples.append(u - (k or 0))
            await asyncio.sleep(0.1)
        fixed = int(sorted(samples)[len(samples) // 2]) if samples else int(17.7 * GiB)
    print(f"VRAM scheduler: engine fixed {fixed/GiB:.2f} GiB, tenant fixed {tenant_fixed/GiB:.2f}, safety {safety/GiB:.2f} -> "
          f"budget pool {(total - fixed - tenant_fixed - safety)/GiB:.2f} GiB", flush=True)
    def budget_for(lim): return max(0, total - fixed - tenant_fixed - safety - lim)
    limit = None; seq = 0; below_since = None; last_metrics = {}; last_m_t = 0; pending_budget_up = None
    last_sms = None; need = 0; need_hist = deque(); want_limit = None; limit_cap = None; nv_free = None; arrivals20 = deque()
    slo = {"cap": args.cap_max if args.tpot_slo is not None else None, "t": 0.0, "tpot": None, "base": None}
    def publish(now, budget, limit, need, lam):
        nonlocal seq, last_sms
        seq += 1; last_sms = sm_share(lam)
        write_json(args.budget_file, dict(seq=seq, ts=now, bytes=int(budget), limit=limit, need=need, lam=lam, sms=last_sms,
                                          tok_s_cap=slo["cap"]))
    f = open(args.log, "w", newline=""); w = csv.writer(f)
    w.writerow(["t", "running", "waiting", "blocks_in_use", "blocks_running_final", "blocks_waiting_admit", "blocks_waiting_final",
                "lambda_hat", "need_gib", "limit_gib", "kvc_used_gib", "kvc_limit_gib", "ft_budget_gib", "ft_tier", "ft_footprint_gib",
                "ft_seq_ack", "action", "m_running", "m_waiting", "m_queue_sum", "m_queue_count", "m_preempt",
                "nvml_free_gib", "limit_cap_gib", "want_limit_gib", "engine_fixed_gib", "tpot_ms", "tpot_base_ms", "tok_s_cap"])
    # initial limit from kvcached
    used0, lim0 = kvc_state()
    limit = lim0 if lim0 else int(args.max_limit_gib * GiB)
    budget = (args.fixed_budget_gib * GiB) if args.fixed_budget_gib is not None else budget_for(limit)
    want_limit = limit; need = 0; publish(time.time(), budget, limit, 0, 0.0)
    while True:
        await asyncio.sleep(args.tick)
        now = time.time()
        while arrivals and arrivals[0] < now - 5.0: arrivals20.append(arrivals.popleft())
        while arrivals20 and arrivals20[0] < now - 20.0: arrivals20.popleft()
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
        _, nv_free = nvml_mem()
        # the limit may only promise memory that is free right now: what kvcached has mapped plus what NVML says is unused (minus safety)
        limit_cap = (int(used + max(0, nv_free - safety)) // (512 * 2**20)) * (512 * 2**20) if (used is not None and nv_free is not None) else None
        action = ""
        need_hist.append((now, need))
        while need_hist and need_hist[0][0] < now - max(args.shrink_window, 0.1): need_hist.popleft()
        need_for_shrink = max(n for _, n in need_hist) if args.shrink_window > 0 else need
        if not args.no_adjust and HAVE_KVC:
            if need <= limit and want_limit is not None and want_limit > limit:
                # D16: the demand fell back under the granted limit while a grow was still pending (the tenant had not
                # released, or the cap held it). Nothing is left to wait for: drop the stale request, and give the tenant
                # back the budget of the limit actually granted once kvcached's mapped bytes are under it.
                want_limit = limit; below_since = None
                if budget_for(limit) > budget: pending_budget_up = budget_for(limit)
                action = "grow_abandoned"
            if need > limit or (want_limit is not None and want_limit > limit):
                if need > limit: want_limit = need; below_since = None
                budget_new = budget_for(want_limit)
                if budget_new < budget:   # tenant must release first; publish at once
                    budget = budget_new; pending_budget_up = None; publish(now, budget, want_limit, need, lam)
                new_limit = want_limit if limit_cap is None else min(want_limit, max(limit, limit_cap))
                if new_limit > limit:
                    limit = new_limit; action = "grow" if limit == want_limit else "grow_partial"
                    try: set_instance_memory_limit(int(limit), instance_name=args.ipc)
                    except Exception as e: action = f"grow_err:{e!r}"[:40]
                else:
                    action = "grow_wait"   # tenant has not released yet
            elif need_for_shrink < limit - args.shrink_margin_gib * GiB and (want_limit is None or want_limit <= limit):
                if below_since is None: below_since = now
                floor = (d["blocks_in_use"] + d["blocks_waiting_admit"]) * args.block_bytes + args.headroom_gib * GiB
                if now - below_since >= args.hold_s and need_for_shrink >= floor:
                    limit = max(need_for_shrink, int(math.ceil(floor / (0.5 * GiB)) * 0.5 * GiB)); want_limit = limit; below_since = None; action = "shrink"
                    try: set_instance_memory_limit(int(limit), instance_name=args.ipc)
                    except Exception as e: action = f"shrink_err:{e!r}"[:40]
                    pending_budget_up = budget_for(limit)
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
        if args.tpot_slo is not None and TPOT_REF is not None and now - slo["t"] >= args.slo_period:
            # P1.1: multiplicative update of the tenant's tokens/s cap from the decode TPOT error against the baseline
            slo["t"] = now
            lam20 = sum(1 for t in arrivals20 if t >= now - 20.0) / 20.0
            slo["tpot"] = tpot_now(now); slo["base"] = TPOT_REF(lam20)
            if slo["tpot"] is not None and slo["base"]:
                e = (slo["tpot"] / slo["base"] - 1.0) - args.tpot_slo
                factor = min(1.5, max(0.5, math.exp(-args.slo_gain * e)))
                new_cap = min(args.cap_max, max(args.cap_min, slo["cap"] * factor))
                if abs(new_cap - slo["cap"]) / slo["cap"] > 0.02:
                    slo["cap"] = new_cap; publish(now, budget, limit, need, lam)
        st = read_json(args.state_file) or {}
        w.writerow([f"{now:.3f}", d["running"], d["waiting"], d["blocks_in_use"], d["blocks_running_final"], d["blocks_waiting_admit"],
                    d["blocks_waiting_final"], f"{lam:.2f}", f"{need/GiB:.2f}", f"{limit/GiB:.2f}",
                    "" if used is None else f"{used/GiB:.2f}", "" if kvlim is None else f"{kvlim/GiB:.2f}", f"{budget/GiB:.2f}",
                    st.get("tier", ""), st.get("footprint_gib", ""), st.get("seq_ack", ""), action,
                    last_metrics.get("num_requests_running", ""), last_metrics.get("num_requests_waiting", ""),
                    last_metrics.get("request_queue_time_seconds_sum", ""), last_metrics.get("request_queue_time_seconds_count", ""),
                    last_metrics.get("num_preemptions_total", ""),
                    "" if nv_free is None else f"{nv_free/GiB:.2f}", "" if limit_cap is None else f"{limit_cap/GiB:.2f}",
                    "" if want_limit is None else f"{want_limit/GiB:.2f}", f"{fixed/GiB:.2f}",
                    "" if slo["tpot"] is None else f"{slo['tpot']:.2f}", "" if slo["base"] is None else f"{slo['base']:.2f}", "" if slo["cap"] is None else f"{slo['cap']:.0f}"])
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
