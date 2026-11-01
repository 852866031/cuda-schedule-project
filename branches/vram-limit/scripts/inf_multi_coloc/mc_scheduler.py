#!/usr/bin/env python3
"""External cross-model scheduler for the prefill-only config.

Sits IN FRONT of N engines (:8400+i) and intercepts requests before they reach them:
clients post to /m/<i>/v1/completions, the scheduler queues them per model and
dispatches under a GLOBAL in-flight cap K across all models. K=1 runs the N models'
prefills strictly sequentially; K=0 means no cap (pass-through: every request reaches its
engine immediately and the N prefills contend on GPU1 under MPS).

Policies (who goes next when a slot frees):
  fcfs  global arrival order, ignoring model
  rr    round-robin across models with a non-empty queue (per-model fairness)
  prio  model 0 strictly first, then the rest in arrival order (priority tenant)

Responses are streamed through unchanged, so client-side TTFT includes scheduler queueing
-- that is the point: the queue is where an admission policy spends latency.

    mc_scheduler.py --n 4 --k 1 --policy rr --port 8390
"""

import argparse
import asyncio
import collections
import itertools
import json
import time

import aiohttp
from aiohttp import web


class Scheduler:
    def __init__(self, n, k, policy):
        self.n, self.k, self.policy = n, k, policy
        self.queues = [collections.deque() for _ in range(n)]
        self.inflight = 0
        self.rr_next = 0
        self.seq = itertools.count()
        self.stats = {"admitted": [0] * n, "queue_wait_s": [0.0] * n, "max_qlen": 0}

    def _pick(self):
        nonempty = [i for i in range(self.n) if self.queues[i]]
        if not nonempty:
            return None
        if self.policy == "prio" and self.queues[0]:
            return 0
        if self.policy == "rr":
            for d in range(self.n):
                i = (self.rr_next + d) % self.n
                if self.queues[i]:
                    self.rr_next = (i + 1) % self.n
                    return i
        # fcfs (and prio's non-priority tail): oldest head across queues
        return min(nonempty, key=lambda i: self.queues[i][0][0])

    def _dispatch(self):
        while (self.k == 0 or self.inflight < self.k):
            i = self._pick()
            if i is None:
                return
            _, t_enq, fut = self.queues[i].popleft()
            self.inflight += 1
            self.stats["admitted"][i] += 1
            self.stats["queue_wait_s"][i] += time.perf_counter() - t_enq
            fut.set_result(None)

    async def admit(self, i):
        fut = asyncio.get_running_loop().create_future()
        self.queues[i].append((next(self.seq), time.perf_counter(), fut))
        self.stats["max_qlen"] = max(self.stats["max_qlen"], sum(map(len, self.queues)))
        self._dispatch()
        await fut

    def release(self):
        self.inflight -= 1
        self._dispatch()


async def handle(request):
    sched: Scheduler = request.app["sched"]
    i = int(request.match_info["i"])
    body = await request.read()
    await sched.admit(i)
    try:
        sess: aiohttp.ClientSession = request.app["session"]
        async with sess.post(f"http://127.0.0.1:{8400 + i}/v1/completions", data=body,
                             headers={"Content-Type": "application/json"}) as up:
            resp = web.StreamResponse(status=up.status,
                                      headers={"Content-Type": up.headers.get(
                                          "Content-Type", "application/json")})
            await resp.prepare(request)
            async for chunk in up.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
            return resp
    finally:
        sched.release()


async def stats(request):
    return web.json_response(request.app["sched"].stats)


async def health(request):
    return web.Response(text="ok")


async def on_startup(app):
    app["session"] = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0),
        timeout=aiohttp.ClientTimeout(total=600))


async def on_cleanup(app):
    await app["session"].close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, default=1, help="global in-flight cap (0 = none)")
    ap.add_argument("--policy", choices=["fcfs", "rr", "prio"], default="fcfs")
    ap.add_argument("--port", type=int, default=8390)
    args = ap.parse_args()
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["sched"] = Scheduler(args.n, args.k, args.policy)
    app.router.add_post("/m/{i}/v1/completions", handle)
    app.router.add_get("/stats", stats)
    app.router.add_get("/health", health)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    print(json.dumps(vars(args)), flush=True)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
