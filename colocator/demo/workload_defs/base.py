"""Base class for demo workloads (one workload per file in this directory).

A workload file defines a class and exports it as the module-level name
``WORKLOAD``; ../workloads.py discovers and registers every such file.

Execution model — DEPTH-2 PIPELINE (same code path solo and colocated):
    run_iter() admits one iteration of GPU work WITHOUT waiting for it:
    it waits for the iteration from DEPTH launches ago (torch.cuda.Event,
    whose cudaEventRecord the colocator redirects onto the client's stream),
    then launches the next one. So two iterations are in flight at any
    moment — the GPU always has the next kernel queued back-to-back instead
    of draining between iterations.

Contract (used by run_demo.py's client_body):
    setup()          allocate host/device state
    launch_iter()    launch ONE iteration's GPU work, NO sync; return the
                     result buffer (or None)
    calibrate()      pick self.iters for a ~TARGET_MS timed section,
                     measured at pipelined steady state
    check()          one verified iteration (uses run_iter_sync)
    run_iter()       pipelined admission (above)
    drain()          wait for everything in flight (end of loops, barriers)
    describe()       HTML fragment for the replay's description boxes
"""

import collections
import time

import torch

TARGET_MS = 500       # calibrate() sizes the timed section to about this
CALIBRATE_PROBES = 4  # steady-state iterations measured during calibration
DEPTH = 2             # iterations in flight


class WorkloadBase:
    name = "?"
    DEPTH = DEPTH
    # False = this workload launches kernels the interceptor cannot see
    # (e.g. cuBLAS via statically-linked runtime / driver API). Colocating it
    # would split its dependency chains across unsynchronized streams (wrong
    # results), so run_demo/run_pairs refuse it in colocated mode.
    colocatable = True

    def __init__(self, iters=None, device="cuda:0"):
        self.iters = iters          # None -> calibrate() decides
        self.device = torch.device(device)
        self._events = collections.deque()
        self._slot = 0              # cycles 0..DEPTH-1 for double buffers
        # Named CLOCK_MONOTONIC_RAW timestamps a workload may stamp during
        # setup (e.g. llmdecode's "load_done"); recorded into results.json and
        # used by the visualizer (e.g. to trim model loading from a timeline).
        self.marks = {}

    # -- to override -----------------------------------------------------
    def setup(self):
        raise NotImplementedError

    def launch_iter(self):
        raise NotImplementedError

    def check(self):
        return True

    def params_html(self):
        return ""

    def summary(self):
        return ""

    # -- pipelined execution --------------------------------------------
    def run_iter(self):
        """Admit one iteration; keep DEPTH iterations in flight."""
        if len(self._events) >= self.DEPTH:
            self._events.popleft().synchronize()  # wait for iter from DEPTH ago
        res = self.launch_iter()
        self._slot = (self._slot + 1) % self.DEPTH
        ev = torch.cuda.Event()
        ev.record()                                # redirected under the colocator
        self._events.append(ev)
        return res

    def drain(self):
        torch.cuda.current_stream().synchronize()  # redirected under the colocator
        self._events.clear()

    def run_iter_sync(self):
        """One fully-completed iteration (for checks)."""
        self.drain()
        res = self.launch_iter()
        self._slot = (self._slot + 1) % self.DEPTH
        self.drain()
        return res

    # -- common ----------------------------------------------------------
    def calibrate(self, target_ms=TARGET_MS):
        """Set self.iters for ~target_ms, measured at pipeline steady state."""
        if self.iters is not None:
            return
        for _ in range(self.DEPTH):  # fill the pipeline first
            self.run_iter()
        # Measure a BLOCK of admissions, not individual ones: pipelined
        # admissions jitter (one blocks ~a full kernel, the next returns
        # instantly), so a per-sample min under-estimates wildly — measured:
        # a min of ~0 ms calibrated a 9 ms/iter workload to the 5000 clamp.
        t0 = time.perf_counter()
        for _ in range(CALIBRATE_PROBES):
            self.run_iter()
        per_iter_ms = (time.perf_counter() - t0) / CALIBRATE_PROBES * 1e3
        self.drain()
        self.iters = max(3, min(5000, round(target_ms / max(per_iter_ms, 1e-3))))
        self.calibrated_iter_ms = per_iter_ms

    def describe(self):
        cal = ""
        if getattr(self, "calibrated_iter_ms", None) is not None:
            cal = (f"<li>calibrated: <b>{self.iters} iterations</b> × "
                   f"{self.calibrated_iter_ms:.2f} ms/iter ≈ "
                   f"{self.iters * self.calibrated_iter_ms:.0f} ms timed section</li>")
        elif self.iters is not None:
            cal = f"<li>iterations: <b>{self.iters}</b> (forced)</li>"
        pipe = (f"<li>pipelined: <b>{self.DEPTH} iterations in flight</b> — the next "
                f"launch is admitted when the one from {self.DEPTH} ago completes</li>")
        return f"<p>{self.summary()}</p><ul>{self.params_html()}{pipe}{cal}</ul>"


def sm_count(device=0):
    return torch.cuda.get_device_properties(device).multi_processor_count
