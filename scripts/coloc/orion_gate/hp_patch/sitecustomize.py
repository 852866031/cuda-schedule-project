"""Publish the decode engine's GPU-busy window to /dev/shm for the Orion-lite gate.

Loaded via PYTHONPATH into every process of the vLLM stack (same mechanism as
split_simple/p2p_patch); does nothing unless COLOC_HP_SIGNAL=1. Only the engine
process ever imports vllm.v1.worker.gpu_model_runner, so the patch lands exactly
where the GPU work happens.

The signal: `GPUModelRunner.execute_model` is wall-clock synchronous — it returns
when the step's GPU work is done (vLLM replays decode steps as CUDA graphs and the
sampler syncs on the output tokens). Wrapping it therefore gives a completion-aware
busy window with no CUDA interception at all — the thing an LD_PRELOAD shim cannot
get, because cuBLAS/Triton launch via driver-API entry points obtained through
cuGetProcAddress (measured: a runtime-API shim sees ~none of the launches).

Page layout (/dev/shm/coloc_hp_busy, 7 little-endian int64s), shared with
scripts/coloc/ft_train.py's --gate reader:
  [0] magic 0x434f4c4f43   [1] heartbeat_ns (CLOCK_MONOTONIC)   [2] busy
  [3] last_step_end_ns     [4] steps                            [5,6] spare
Pinned to vllm==0.15.1.
"""

import os
import sys

if os.environ.get("COLOC_HP_SIGNAL") == "1":
    from importlib.machinery import PathFinder

    _RUNNER = "vllm.v1.worker.gpu_model_runner"
    _MAGIC = 0x434F4C4F43

    def _open_page():
        import mmap
        path = "/dev/shm/coloc_hp_busy"
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(fd, 7 * 8)
        page = mmap.mmap(fd, 7 * 8)
        os.close(fd)
        return page

    def _patch_runner(mod):
        import struct
        import threading
        import time

        page = _open_page()

        def wr(i, v):
            struct.pack_into("<q", page, i * 8, v)

        wr(2, 0)
        wr(4, 0)
        wr(0, _MAGIC)

        def heartbeat():
            while True:
                wr(1, time.monotonic_ns())
                time.sleep(0.1)

        threading.Thread(target=heartbeat, daemon=True).start()

        orig = mod.GPUModelRunner.execute_model

        def execute_model(self, *a, **kw):
            wr(2, 1)
            wr(1, time.monotonic_ns())
            try:
                return orig(self, *a, **kw)
            finally:
                t = time.monotonic_ns()
                wr(3, t)
                wr(1, t)
                wr(2, 0)
                struct.pack_into("<q", page, 4 * 8,
                                 struct.unpack_from("<q", page, 4 * 8)[0] + 1)

        mod.GPUModelRunner.execute_model = execute_model
        print("[hp_signal] execute_model wrapped; busy page live", file=sys.stderr)

    class _PatchOnImport:
        def find_spec(self, fullname, path=None, target=None):
            if fullname != _RUNNER:
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
                _patch_runner(module)

            spec.loader.exec_module = exec_module
            return spec

    if not any(isinstance(f, _PatchOnImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchOnImport())
