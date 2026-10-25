"""Record-only timeline of the decode engine (E1). Inert unless KV_TRACE=1.

Loaded via PYTHONPATH sitecustomize into the decode engine's processes only (kv_launch.sh
sets the env for that engine alone). Patches land on import, in whichever process
imports the module, which is the EngineCore (scheduler + the in-process worker):

  vllm.v1.core.sched.scheduler.Scheduler
      add_request        -> "arr": request reaches the scheduler; free blocks then
      schedule           -> "sch": one line per call (running, waiting, free blocks,
                            newly admitted request ids + their LMCache-hit tokens)
      update_from_output -> "upd": one line per executed batch (wall time; ids whose
                            first output token appeared in it)
      _free_request      -> "fin": request finished and released its blocks
  vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector.LMCacheConnectorV1
      start_load_kv      -> "load": CPU wall time of the connector's load call and the
                            number of requests / tokens it loaded (no CUDA sync added --
                            LMCache's retrieve already blocks the host)

Nothing here changes scheduling or adds device synchronization: every wrapper calls the
original and only reads state. Lines are JSON, appended to KV_TRACE_PATH and flushed at
most every 0.5 s (and at exit). Times are time.monotonic(), the same CLOCK_MONOTONIC as
the client's perf_counter, so client and engine records share one time axis.
A request is identified by its 4 suffix tokens (prompt[6144:6148]); the driver maps
them to the workload index. Pinned to vllm==0.15.1 / lmcache 0.4.4.
"""

import os
import sys

if os.environ.get("KV_TRACE") == "1":
    import atexit
    import json
    import time
    from importlib.machinery import PathFinder

    _PATH = os.environ.get("KV_TRACE_PATH", f"/dev/shm/kvv_trace_{os.getpid()}.jsonl")
    _SCHED = "vllm.v1.core.sched.scheduler"
    _CONN = "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector"
    _KEY_AT = int(os.environ.get("KV_TRACE_KEY_AT", "6144"))
    _now = time.monotonic

    class _Writer:
        def __init__(self):
            self.f = None
            self.buf = []
            self.t_flush = 0.0

        def put(self, rec):
            self.buf.append(rec)
            t = rec["t"]
            if t - self.t_flush >= 0.5:
                self.flush()
                self.t_flush = t

        def flush(self):
            if not self.buf:
                return
            if self.f is None:
                self.f = open(_PATH, "a")
                self.f.write(json.dumps({"ev": "start", "pid": os.getpid(), "t": _now(),
                                         "wall": time.time()}) + "\n")
            self.f.write("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in self.buf))
            self.f.flush()
            self.buf.clear()

    _W = _Writer()
    atexit.register(_W.flush)

    def _key(req):
        p = req.prompt_token_ids or []
        return p[_KEY_AT:_KEY_AT + 4] if len(p) >= _KEY_AT + 4 else p[:4]

    def _patch_scheduler(mod):
        S = mod.Scheduler
        o_add, o_sch, o_upd, o_free = (S.add_request, S.schedule, S.update_from_output,
                                       S._free_request)
        first_seen = set()

        def free_blocks(self):
            return self.kv_cache_manager.block_pool.get_num_free_blocks()

        def add_request(self, request):
            t = _now()
            fb = free_blocks(self)
            r = o_add(self, request)
            _W.put({"ev": "arr", "t": t, "id": request.request_id, "key": _key(request),
                    "n_prompt": len(request.prompt_token_ids or []), "free": fb,
                    "run": len(self.running), "wait": len(self.waiting)})
            return r

        def schedule(self):
            t0 = _now()
            out = o_sch(self)
            new = [(nr.req_id, self.requests[nr.req_id].num_external_computed_tokens
                    if nr.req_id in self.requests else None,
                    len(nr.block_ids[0]) if nr.block_ids else 0)
                   for nr in out.scheduled_new_reqs]
            rec = {"ev": "sch", "t": t0, "dt": _now() - t0, "run": len(self.running),
                   "wait": len(self.waiting), "free": free_blocks(self),
                   "ntok": out.total_num_scheduled_tokens}
            if new:
                rec["new"] = new                       # [id, external tokens, blocks]
            if out.preempted_req_ids:
                rec["pre"] = list(out.preempted_req_ids)
            _W.put(rec)
            return out

        def update_from_output(self, scheduler_output, model_runner_output):
            t = _now()
            res = o_upd(self, scheduler_output, model_runner_output)
            firsts = []
            for rid in scheduler_output.num_scheduled_tokens:
                if rid in first_seen:
                    continue
                req = self.requests.get(rid)
                if req is not None and req.num_output_tokens > 0:
                    first_seen.add(rid)
                    firsts.append(rid)
            rec = {"ev": "upd", "t": t, "n": len(scheduler_output.num_scheduled_tokens),
                   "ntok": scheduler_output.total_num_scheduled_tokens,
                   "nnew": len(scheduler_output.scheduled_new_reqs)}
            if firsts:
                rec["first"] = firsts
            _W.put(rec)
            return res

        def _free_request(self, request):
            _W.put({"ev": "fin", "t": _now(), "id": request.request_id,
                    "nout": request.num_output_tokens, "free": free_blocks(self)})
            first_seen.discard(request.request_id)
            return o_free(self, request)

        S.add_request, S.schedule = add_request, schedule
        S.update_from_output, S._free_request = update_from_output, _free_request
        print("[kv_trace] scheduler wrapped ->", _PATH, file=sys.stderr, flush=True)

    def _patch_connector(mod):
        C = mod.LMCacheConnectorV1
        o_load = C.start_load_kv

        def start_load_kv(self, forward_context, **kw):
            t0 = _now()
            o_load(self, forward_context, **kw)
            t1 = _now()
            n = ntok = 0
            try:
                for r in self._get_connector_metadata().requests:
                    ls = getattr(r, "load_spec", None)
                    if ls is not None and ls.can_load:
                        n += 1
                        ntok += ls.lmcache_cached_tokens - ls.vllm_cached_tokens
            except Exception:
                n = -1
            if n:
                _W.put({"ev": "load", "t": t0, "dt": t1 - t0, "n": n, "ntok": ntok})

        C.start_load_kv = start_load_kv
        print("[kv_trace] connector start_load_kv wrapped", file=sys.stderr, flush=True)

    _HOOKS = {_SCHED: _patch_scheduler, _CONN: _patch_connector}

    class _PatchOnImport:
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in _HOOKS:
                return None
            sys.meta_path.remove(self)
            try:
                spec = PathFinder.find_spec(fullname, path, target)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module
            hook = _HOOKS[fullname]

            def exec_module(module):
                orig_exec(module)
                hook(module)

            spec.loader.exec_module = exec_module
            return spec

    if not any(isinstance(f, _PatchOnImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchOnImport())
