"""Observation-only hooks for the vLLM 0.30 engine, loaded through PYTHONPATH (no vLLM file is
modified; nothing here schedules anything).

COLOC_HP_SIGNAL=1   publish the engine step's GPU-busy window to /dev/shm/coloc_hp_busy for the
                    tenant's idle-window gate. Model Runner V2 splits a step into
                    GPUModelRunner.execute_model (launches the forward) and sample_tokens (syncs
                    on the sampled ids), so busy is raised at execute_model entry and cleared
                    when sample_tokens returns. Page: 7 little-endian int64s —
                    [0] magic 0x434f4c4f43 [1] heartbeat_ns [2] busy [3] last_step_end_ns
                    [4] steps [5] busy_ns_total [6] spare.
COLOC_XFER_TRACE=1  record every native-offload transfer (vllm.v1.kv_offload.cpu.gpu_worker
                    SingleDirectionOffloadingHandler): bytes from the copy plan, direction from
                    the handler, submit/done wall time, CUDA-event elapsed ms.
"""
import os
import sys

_MAGIC = 0x434F4C4F43


def _install_meta_hook(modname, patcher):
    from importlib.machinery import PathFinder

    class _PatchOnImport:
        def find_spec(self, fullname, path=None, target=None):
            if fullname != modname:
                return None
            sys.meta_path.remove(self)
            try:
                spec = PathFinder.find_spec(fullname, path, target)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                try:
                    patcher(module)
                except Exception as e:  # never break the engine
                    print(f"[coloc hp_patch] patch of {modname} failed: {e!r}", file=sys.stderr)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _PatchOnImport())


if os.environ.get("COLOC_ENGINE_SMS"):
    def _patch_engine_green(mod):
        import torch
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from green_ctx import GreenCtx
        n = int(os.environ["COLOC_ENGINE_SMS"])
        R = mod.GPUModelRunner
        state = {"g": None}
        def _stream(self):
            if state["g"] is None:
                state["g"] = GreenCtx(n, device=0, remaining=bool(os.environ.get("COLOC_ENGINE_SMS_REMAINING")))   # REMAINING: the SMs left after an n-SM split (a second engine disjoint from one on the first n)
                print(f"[coloc hp_patch] engine steps on a green context of {state['g'].sm_count} SMs{' (remaining partition)' if os.environ.get('COLOC_ENGINE_SMS_REMAINING') else ''}", file=sys.stderr)
            return state["g"].torch_stream()
        for name in ("execute_model", "sample_tokens"):
            orig = getattr(R, name)
            def wrapped(self, *a, _orig=orig, **kw):
                with torch.cuda.stream(_stream(self)):
                    return _orig(self, *a, **kw)
            setattr(R, name, wrapped)
    _install_meta_hook("vllm.v1.worker.gpu.model_runner", _patch_engine_green)

if os.environ.get("COLOC_HP_SIGNAL") == "1":
    def _patch_runner(mod):
        import mmap, struct, threading, time
        fd = os.open("/dev/shm/coloc_hp_busy", os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(fd, 7 * 8)
        page = mmap.mmap(fd, 7 * 8)
        os.close(fd)

        def wr(i, v): struct.pack_into("<q", page, i * 8, v)
        def rd(i): return struct.unpack_from("<q", page, i * 8)[0]
        wr(2, 0); wr(4, 0); wr(5, 0); wr(0, _MAGIC)

        def heartbeat():
            while True:
                wr(1, time.monotonic_ns()); time.sleep(0.1)
        threading.Thread(target=heartbeat, daemon=True).start()

        R = mod.GPUModelRunner
        orig_exec, orig_sample = R.execute_model, R.sample_tokens
        import queue, torch
        state = {"started": 0, "busy_since": 0}
        q = queue.Queue()

        def waiter():
            # Model Runner V2 launches a step asynchronously (execute_model/sample_tokens return before
            # the GPU finishes), so the end of the GPU work is taken from a CUDA event recorded after
            # the step's launches; busy clears only when the newest started step has completed.
            while True:
                n, ev = q.get()
                try:
                    ev.synchronize()
                except Exception:
                    pass
                t = time.monotonic_ns()
                wr(4, rd(4) + 1); wr(3, t); wr(1, t)
                if n == state["started"]:          # newest step done: GPU idle from here
                    wr(5, rd(5) + (t - state["busy_since"])); wr(2, 0)
        threading.Thread(target=waiter, daemon=True).start()

        def execute_model(self, *a, **kw):
            if not kw.get("dummy_run", False):
                state["started"] += 1
                now = time.monotonic_ns()
                if not rd(2): state["busy_since"] = now   # idle -> busy transition starts the clock
                wr(2, 1); wr(1, now)
            return orig_exec(self, *a, **kw)

        def sample_tokens(self, *a, **kw):
            try:
                return orig_sample(self, *a, **kw)
            finally:
                if rd(2):
                    try:
                        ev = torch.cuda.Event(); ev.record(torch.cuda.current_stream()); q.put((state["started"], ev))
                    except Exception:
                        wr(2, 0)

        R.execute_model, R.sample_tokens = execute_model, sample_tokens
        print("[coloc hp_patch] busy page live (execute_model -> GPU completion event)", file=sys.stderr)

    _install_meta_hook("vllm.v1.worker.gpu.model_runner", _patch_runner)


if os.environ.get("COLOC_XFER_TRACE") == "1":
    def _patch_offload(mod):
        import time
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from xfer_trace import XferTrace, default_path
        trace = XferTrace(default_path("inf"), "inf")
        H = mod.SingleDirectionOffloadingHandler
        orig_submit, orig_finished = H.transfer_async, H.get_finished

        def transfer_async(self, job_id, src_spec, dst_spec):
            t = time.time()
            ok = orig_submit(self, job_id, src_spec, dst_spec)
            if ok:
                if not hasattr(self, "_coloc_pending"):
                    self._coloc_pending = {}
                self._coloc_pending[job_id] = t
            return ok

        def get_finished(self):
            res = orig_finished(self)
            now = time.time()
            pend = getattr(self, "_coloc_pending", {})
            for r in res:
                t0 = pend.pop(r.job_id, now)
                trace.record("offload", "d2h" if self.gpu_to_cpu else "h2d", r.transfer_size,
                             t0, now, r.transfer_time * 1e3, f"job={r.job_id}")
            return res

        H.transfer_async, H.get_finished = transfer_async, get_finished
        print("[coloc hp_patch] offload transfer tracer installed", file=sys.stderr)

    _install_meta_hook("vllm.v1.kv_offload.cpu.gpu_worker", _patch_offload)
