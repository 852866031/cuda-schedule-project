#!/usr/bin/env python
"""Load generator against the vLLM OpenAI completions endpoint. Streams every request and
records per-token arrival times, so inter-token latency (≈ engine step time while the request
is in every batch) can be plotted against the memory trace. One JSON line per request.

modes
  stream   open-loop Poisson arrivals at --rate req/s for --duration s; each request is a
           fresh random-token prompt of --prompt-tokens tokens (no shared prefix), --max-tokens
           greedy output with ignore_eos. Used for Experiment A and the overhead comparison.
  session  closed-loop multi-turn sessions: --sessions concurrent sessions, each starting from
           a private random context of --prompt-tokens tokens; every turn appends --turn-tokens
           new random tokens to the context and generates --max-tokens, then waits an
           exponential think time with mean --think s. With vLLM prefix caching off, every turn's
           prefix must be re-fetched from LMCache (or recomputed), so loads happen continuously.
  check    send --n-check fixed natural-language prompts one at a time (greedy, 64 tokens) and
           store the texts for cross-run output comparison.
Prompts are passed as token-id lists; token ids are drawn from a seeded RNG so runs are
repeatable. Token ids avoid the first 1000 and last 1000 ids (special/rare tokens).
"""
import argparse, asyncio, json, random, time
import aiohttp

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
ap.add_argument("--mode", choices=["stream", "session", "check"], default="stream")
ap.add_argument("--out", required=True)
ap.add_argument("--rate", type=float, default=2.0)
ap.add_argument("--duration", type=float, default=120)
ap.add_argument("--prompt-tokens", type=int, default=6000)
ap.add_argument("--max-tokens", type=int, default=128)
ap.add_argument("--sessions", type=int, default=8)
ap.add_argument("--turn-tokens", type=int, default=256)
ap.add_argument("--think", type=float, default=4.0)
ap.add_argument("--max-context", type=int, default=7600)
ap.add_argument("--n-check", type=int, default=6)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--vocab", type=int, default=128256)
args = ap.parse_args()

rng = random.Random(args.seed)
LO, HI = 1000, args.vocab - 1000
out_f = open(args.out, "a")
out_lock = asyncio.Lock()

async def one_request(session, rid, prompt, max_tokens, extra=None):
    body = {"model": args.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "ignore_eos": True, "return_token_ids": True}
    if extra: body.update(extra)
    rec = {"rid": rid, "t_send": time.time(), "prompt_tokens": len(prompt) if isinstance(prompt, list) else None,
           "tok_t": [], "tok_n": [], "text": "", "status": None, "error": None}
    try:
        async with session.post(f"{args.url}/v1/completions", json=body) as resp:
            rec["status"] = resp.status
            if resp.status != 200:
                rec["error"] = (await resp.text())[:500]
            else:
                async for raw in resp.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:"): continue
                    data = line[5:].strip()
                    if data == "[DONE]": break
                    j = json.loads(data)
                    ch = j["choices"][0]
                    txt = ch.get("text", "")
                    ids = ch.get("token_ids") or []
                    # vLLM may deliver several generated tokens in one SSE chunk; keep the count so
                    # per-token (per-step) latency can be recovered as gap / tokens-in-chunk.
                    if txt or ids or ch.get("finish_reason"):
                        rec["tok_t"].append(time.time()); rec["tok_n"].append(max(1, len(ids))); rec["text"] += txt
                    if ch.get("finish_reason"): rec["finish_reason"] = ch["finish_reason"]
    except Exception as e:
        rec["error"] = repr(e)[:500]
    rec["t_done"] = time.time()
    rec["n_out"] = sum(rec["tok_n"]); rec["n_chunks"] = len(rec["tok_t"])
    async with out_lock:
        out_f.write(json.dumps(rec) + "\n"); out_f.flush()
    return rec

def rand_tokens(n, r):
    return [r.randrange(LO, HI) for _ in range(n)]

async def run_stream():
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        tasks = []; t0 = time.time(); rid = 0; t_next = t0
        while True:
            now = time.time()
            if now >= t0 + args.duration: break
            if now < t_next:
                await asyncio.sleep(min(t_next - now, 0.05)); continue
            prompt = rand_tokens(args.prompt_tokens, rng)
            tasks.append(asyncio.create_task(one_request(session, rid, prompt, args.max_tokens)))
            rid += 1
            t_next += rng.expovariate(args.rate)
        print(f"sent {rid} requests; waiting for completion", flush=True)
        await asyncio.gather(*tasks)

async def run_session(session, sid, deadline):
    r = random.Random(args.seed * 1000 + sid)
    ctx = rand_tokens(args.prompt_tokens, r)
    turn = 0
    while time.time() < deadline:
        rec = await one_request(session, f"s{sid}t{turn}", ctx, args.max_tokens)
        turn += 1
        if rec["error"]: await asyncio.sleep(1.0)
        ctx = ctx + rand_tokens(args.turn_tokens, r)
        if len(ctx) + args.max_tokens > args.max_context:   # start a new conversation
            ctx = rand_tokens(args.prompt_tokens, r)
        await asyncio.sleep(r.expovariate(1.0 / args.think))

async def run_sessions():
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        deadline = time.time() + args.duration
        await asyncio.gather(*[run_session(session, i, deadline) for i in range(args.sessions)])

CHECK_PROMPTS = [
    "The history of the Roman Empire begins with",
    "In computer science, a hash table is",
    "The theory of general relativity, proposed by Albert Einstein,",
    "Photosynthesis is the process by which plants",
    "The city of Paris is known for",
    "A central processing unit executes instructions by",
    "The Pacific Ocean is the largest ocean on Earth and",
    "In economics, inflation refers to",
]
async def run_check():
    async with aiohttp.ClientSession() as session:
        for i, p in enumerate(CHECK_PROMPTS[: args.n_check]):
            rec = await one_request(session, f"c{i}", p, 64)
            print(f"c{i}: {rec['text'][:80]!r}")

if args.mode == "stream": asyncio.run(run_stream())
elif args.mode == "session": asyncio.run(run_sessions())
else: asyncio.run(run_check())
