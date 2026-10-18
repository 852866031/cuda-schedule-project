"""Asynchronous KV save for LMCacheConnectorV1, loaded into a vLLM engine via PYTHONPATH.

CPython imports `sitecustomize` at startup, so putting this directory on PYTHONPATH
reaches the API server and the EngineCore it spawns. Active only with LMC_ASYNC_SAVE=1
(pfp_launch.sh sets both for the 3B prefill engine only). Pinned to vllm==0.15.1 +
lmcache==0.4.4.

What upstream does. vLLM calls `kv_connector.wait_for_save()` inside `execute_model`,
before the step's outputs are returned, and LMCache does the whole GPU->host copy there
(`lmcache_engine.store`, ~55 ms for a 6144-token Qwen2.5-3B prompt). So token #1 of a
prefill is computed, then held back for the copy, and the engine is blocked for it.
LMCache's `get_finished()` returns (None, None): it never uses vLLM's delayed-free path.

What this changes (three methods of LMCacheConnectorV1Impl):

  wait_for_save     requests with something to store are handed to ONE background thread
                    (which runs the unmodified upstream code on a shallow copy of the
                    connector bound to this step's metadata) and the call returns at
                    once -- the step's outputs, token #1 included, go out immediately.
  request_finished  a request whose copy is still running returns delay_free=True, so the
                    scheduler keeps its KV blocks instead of handing them to the next
                    request while the copy is reading them.
  get_finished      reports those requests as finished_sending once their copy is done;
                    the scheduler then frees the blocks (vLLM's own mechanism).

Limits, stated: blocks are only protected by the delayed free at request end. A request
PREEMPTED while its copy is in flight would lose its blocks under the copy -- not handled
(prefill-only engines here have zero preemptions; the driver records the counter). A
finished copy is reported at the engine's next step; an idle engine reports it when the
next request arrives, so up to one request's blocks stay held while idle.

Every background copy prints one `ASYNC_SAVE` line (requests, ms, backlog) and, when
done, touches $LMC_ASYNC_SAVED_DIR/<request id> -- the split router waits for that file
before sending the decode leg (lmc_proxy.py --wait-saved-dir), because token #1 now
leaves before the KV exists in the host tier.
"""

import os
import sys
import threading
import time
from importlib.machinery import PathFinder

TARGET = "lmcache.integration.vllm.vllm_v1_adapter"
SAVED_DIR = os.environ.get("LMC_ASYNC_SAVED_DIR", "/dev/shm/pfp_saved")


def _patch(module):
    import copy
    import queue

    import torch

    os.makedirs(SAVED_DIR, exist_ok=True)

    Impl = module.LMCacheConnectorV1Impl
    orig_wait, orig_finished = Impl.wait_for_save, Impl.request_finished

    lock = threading.Lock()
    saving = {}          # req_id -> copies submitted and not yet finished
    delayed = set()      # req_ids whose block free the scheduler is holding for us
    jobs = queue.Queue()

    class _Parent:       # stands in for the vLLM connector: just this step's metadata
        def __init__(self, md):
            self._md = md

        def _get_connector_metadata(self):
            return self._md

    def _worker():
        while True:
            shim, ids, t_sub = jobs.get()
            t0 = time.perf_counter()
            released = False
            try:
                with torch.inference_mode():     # as on the engine's own thread
                    orig_wait(shim)
                t_copy = time.perf_counter()
                with lock:                       # the copy is done: blocks may go
                    for r in ids:
                        saving[r] -= 1
                        if saving[r] <= 0:
                            del saving[r]
                    released = True
                # LMCache pushes L1 -> the store server on its own loop. Wait for that
                # too before telling the router: the decode engine reads the STORE.
                backs = [b for b in shim.lmcache_engine.storage_manager
                         .storage_backends.values() if hasattr(b, "put_tasks")]
                while any(b.put_tasks for b in backs) and \
                        time.perf_counter() - t_copy < 5.0:
                    time.sleep(0.001)
                for r in ids:                    # "KV is in the store" marker for the
                    try:                         # router's decode-leg gate
                        open(os.path.join(SAVED_DIR, r), "w").close()
                    except OSError:
                        pass
            except Exception as e:   # never leave a request's blocks held forever
                print(f"ASYNC_SAVE ERROR {type(e).__name__}: {e}", flush=True)
            finally:
                if not released:
                    t_copy = time.perf_counter()
                    with lock:
                        for r in ids:
                            saving[r] -= 1
                            if saving[r] <= 0:
                                del saving[r]
                print(f"ASYNC_SAVE reqs={len(ids)} wait_ms={1000 * (t0 - t_sub):.1f} "
                      f"copy_ms={1000 * (t_copy - t0):.1f} "
                      f"put_ms={1000 * (time.perf_counter() - t_copy):.1f} "
                      f"backlog={jobs.qsize()}", flush=True)

    threading.Thread(target=_worker, name="lmc-async-save", daemon=True).start()

    def wait_for_save(self):
        if self.kv_role == "kv_consumer" or getattr(self, "use_layerwise", False):
            return orig_wait(self)
        md = self._parent._get_connector_metadata()
        ids = []
        for req in md.requests:
            spec = req.save_spec
            if spec is not None and spec.can_save and \
                    spec.skip_leading_tokens != len(req.token_ids):
                ids.append(req.req_id)
        if not ids:                  # nothing to copy: upstream path (just unpins)
            return orig_wait(self)
        with lock:
            for r in ids:
                saving[r] = saving.get(r, 0) + 1
        shim = copy.copy(self)
        shim._parent = _Parent(md)
        jobs.put((shim, ids, time.perf_counter()))

    def request_finished(self, request, block_ids):
        delay, params = orig_finished(self, request, block_ids)
        with lock:
            if saving.get(request.request_id, 0) > 0:
                delayed.add(request.request_id)
                return True, params
        return delay, params

    def get_finished(self, finished_req_ids):
        with lock:
            done = {r for r in delayed if r not in saving}
            delayed.difference_update(done)
        return (done or None), None

    Impl.wait_for_save = wait_for_save
    Impl.request_finished = request_finished
    Impl.get_finished = get_finished
    print("async_save_patch: LMCacheConnectorV1Impl patched (async KV save)", flush=True)


class _PatchOnImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = PathFinder.find_spec(fullname, path, target)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        orig_exec_module = spec.loader.exec_module

        def exec_module(module):
            orig_exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


if os.environ.get("LMC_ASYNC_SAVE") == "1" and \
        not any(isinstance(f, _PatchOnImport) for f in sys.meta_path):
    sys.meta_path.insert(0, _PatchOnImport())
