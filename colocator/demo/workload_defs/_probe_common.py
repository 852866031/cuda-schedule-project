"""Shared bits for the resource-probe workloads (ported from
~/Documents/Projects/gpu-interfere/profiler/code/probe.cu).

Each probe launches ONE kernel per iteration — grid = one block per SM, like
the original harness — sized (inner-loop count) so a single launch takes a
few to a few tens of ms; calibrate() then picks the iteration count for a
~500 ms timed section. Iterations are pipelined DEPTH-2 by the base class:
two kernels in flight, back-to-back on the client's stream."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ext"))
import colocator_kernels  # noqa: E402

from base import WorkloadBase, sm_count  # noqa: E402


class ProbeBase(WorkloadBase):
    THREADS = 128     # per-block threads (override per probe)
    resource = "?"    # what this probe saturates (for describe())

    def setup(self):
        self.blocks = sm_count(self.device.index or 0)
        self.alloc()

    def alloc(self):
        pass

    def launch(self):
        raise NotImplementedError

    def launch_iter(self):
        self.launch()

    def summary(self):
        return (f"Resource probe (from the gpu-interfere profiler): saturates "
                f"<b>{self.resource}</b>. One kernel launch per iteration, one block "
                f"per SM — isolates the resource from block-scheduler effects.")

    def geometry_html(self):
        return (f"<li>launch geometry: grid ({self.blocks}) = 1 block/SM, "
                f"block ({self.THREADS}) threads</li>")
