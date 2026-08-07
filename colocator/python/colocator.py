"""ctypes frontend for the colocator (mirrors Orion's scheduler_frontend.py).

Drives the two C++ libraries:
  build/libcolocator.so — LD_PRELOAD interposition layer (must be preloaded;
      `ensure_managed_env()` re-execs the interpreter with the right env so
      callers can just run `python run_demo.py ...`).
  build/libsched.so — scheduler core (streams, FCFS loop, issue log).

Usage sketch (see demo/run_demo.py for the real thing):

    ensure_managed_env()                      # no-op if env already correct
    col = Colocator(num_clients=2, priorities=[0, 0])
    results = col.launch([client_fn_a, client_fn_b])
    col.dump_issue_log("runs/x/issue_log.csv")
"""

import ctypes
import os
import sys
import threading
import time

_COL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBCOL = os.path.join(_COL_ROOT, "build", "libcolocator.so")
LIBSCHED = os.path.join(_COL_ROOT, "build", "libsched.so")


def ensure_managed_env():
    """Re-exec the interpreter with LD_PRELOAD + COLOCATOR_MODE=managed.

    LD_PRELOAD only takes effect at process start, so it cannot be set from
    within a running interpreter — hence the exec. Returns normally if the
    environment is already correct.
    """
    if LIBCOL in os.environ.get("LD_PRELOAD", "") \
            and os.environ.get("COLOCATOR_MODE") == "managed":
        return
    if "torch" in sys.modules:
        raise RuntimeError("ensure_managed_env() must run before importing torch")
    env = dict(os.environ)
    prev = env.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = LIBCOL + (":" + prev if prev else "")
    env["COLOCATOR_MODE"] = "managed"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


class Colocator:
    """Owns the scheduler thread and the client threads for one run."""

    def __init__(self, num_clients, priorities=None):
        assert os.environ.get("COLOCATOR_MODE") == "managed", \
            "call ensure_managed_env() first"
        self.num_clients = num_clients
        self.priorities = list(priorities or [0] * num_clients)
        assert len(self.priorities) == num_clients

        # LD_PRELOAD already mapped libcolocator; CDLL just returns a handle.
        self._icept = ctypes.CDLL(LIBCOL)
        self._icept.col_register_client.argtypes = [ctypes.c_int]
        self._icept.col_register_client.restype = ctypes.c_int
        self._icept.col_client_submitted.argtypes = [ctypes.c_int]
        self._icept.col_client_submitted.restype = ctypes.c_uint64

        self._sched = ctypes.CDLL(LIBSCHED)
        self._sched.col_setup.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self._sched.col_setup.restype = ctypes.c_int
        self._sched.col_run.argtypes = []
        self._sched.col_stop.argtypes = []
        self._sched.col_is_running.restype = ctypes.c_int
        self._sched.col_issued_count.argtypes = [ctypes.c_int]
        self._sched.col_issued_count.restype = ctypes.c_uint64
        self._sched.col_dump_issue_log.argtypes = [ctypes.c_char_p]
        self._sched.col_dump_issue_log.restype = ctypes.c_long

    def launch(self, client_fns):
        """Run one callable per client on its own registered thread.

        The main thread (unregistered -> passthrough) initializes the CUDA
        context and the scheduler before any client touches the GPU. Returns
        the callables' return values, re-raises the first client exception.
        """
        assert len(client_fns) == self.num_clients
        import torch  # deferred: must be imported post-exec
        torch.cuda.init()
        torch.ones(1, device="cuda:0")  # force full context creation

        prio = (ctypes.c_int * self.num_clients)(*self.priorities)
        if self._sched.col_setup(self.num_clients, prio) != 0:
            raise RuntimeError("col_setup failed")

        # ctypes releases the GIL around col_run, so the busy-wait loop and
        # Python client threads coexist.
        sched_thread = threading.Thread(target=self._sched.col_run, name="sched")
        sched_thread.start()
        while not self._sched.col_is_running():
            time.sleep(0.001)
        self.sched_tid = sched_thread.native_id  # for observer correlation
        self.client_tids = [None] * self.num_clients

        start_barrier = threading.Barrier(self.num_clients)
        results = [None] * self.num_clients
        errors = [None] * self.num_clients

        def run_client(i, fn):
            try:
                assert self._icept.col_register_client(i) == 0
                self.client_tids[i] = threading.get_native_id()
                start_barrier.wait()
                results[i] = fn()
            except BaseException as e:  # noqa: BLE001 — reported to caller
                errors[i] = e
                try:
                    start_barrier.abort()
                except Exception:
                    pass

        threads = [threading.Thread(target=run_client, args=(i, fn), name=f"client{i}")
                   for i, fn in enumerate(client_fns)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        torch.cuda.synchronize()  # main thread: device-wide, passthrough
        self._sched.col_stop()
        sched_thread.join()

        for e in errors:
            if e is not None:
                raise e

        # Accounting invariant: everything submitted was issued.
        self.stats = []
        for i in range(self.num_clients):
            sub = self._icept.col_client_submitted(i)
            iss = self._sched.col_issued_count(i)
            self.stats.append({"client": i, "submitted": sub, "issued": iss})
            if sub != iss:
                raise RuntimeError(f"client {i}: submitted {sub} != issued {iss}")
        return results

    def dump_issue_log(self, path):
        n = self._sched.col_dump_issue_log(path.encode())
        if n < 0:
            raise RuntimeError(f"could not write issue log to {path}")
        return n
