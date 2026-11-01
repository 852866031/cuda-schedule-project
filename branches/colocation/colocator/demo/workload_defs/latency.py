"""latency — inference-like client: many short kernels, latency-sensitive."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ext"))
import colocator_kernels  # noqa: E402

from base import WorkloadBase  # noqa: E402


class LatencyClient(WorkloadBase):
    name = "latency"
    BATCH, DIM, LAYERS = 256, 1024, 4

    def setup(self):
        g = torch.Generator().manual_seed(0)
        self.w_host = [torch.randn(self.DIM, self.DIM, generator=g) * (self.DIM ** -0.5)
                       for _ in range(self.LAYERS)]
        self.x_host = torch.randn(self.BATCH, self.DIM, generator=g).pin_memory()
        # Pinned result buffers, one per pipeline slot: pageable D2H would
        # make cudaMemcpyAsync synchronous for the caller — under the
        # colocator that caller is the scheduler, and the stall would block
        # BOTH clients' queues. One buffer per in-flight iteration avoids
        # racing two D2H copies into the same memory.
        self.out_host = [torch.empty(self.BATCH, pin_memory=True)
                         for _ in range(self.DEPTH)]
        self.weights = [w.to(self.device) for w in self.w_host]

    def launch_iter(self):
        x = self.x_host.to(self.device, non_blocking=True)   # H2D
        for w in self.weights:
            x = torch.relu(colocator_kernels.matmul(x, w))   # SGEMM + relu
        out = x.sum(dim=1)                                   # reduction
        dst = self.out_host[self._slot]
        dst.copy_(out, non_blocking=True)                    # D2H (pinned, async)
        return dst

    def reference(self):
        x = self.x_host.double()
        for w in self.w_host:
            x = torch.relu(x @ w.double())
        return x.sum(dim=1)

    def check(self):
        got = self.run_iter_sync().double().clone()
        ok = torch.allclose(got, self.reference(), rtol=1e-3, atol=1e-2)
        if not ok:
            print(f"[{self.name}] check mismatch")
        return ok

    def summary(self):
        return ("Inference-like <b>latency-sensitive</b> client: per iteration, copy a small "
                "input batch H2D, run a 4-layer chain of tiled-SGEMM → relu, row-sum, and "
                "retrieve the result D2H (pinned). Many short kernels; per-iteration latency "
                "is its metric.")

    def params_html(self):
        return (f"<li>sgemm_tiled ×{self.LAYERS}/iter: C({self.BATCH}×{self.DIM}) = "
                f"A({self.BATCH}×{self.DIM}) @ W({self.DIM}×{self.DIM}), "
                f"grid ({self.DIM // 32}×{self.BATCH // 32}), block (32×32)</li>"
                f"<li>+ {self.LAYERS} relu (elementwise) + 1 row-sum (reduce) per iter</li>"
                f"<li>copies: {self.BATCH * self.DIM * 4 // 1024} KiB H2D in, "
                f"{self.BATCH * 4 / 1024:.0f} KiB D2H out (pinned)</li>"
                f"<li>weights preloaded: {self.LAYERS}×{self.DIM}×{self.DIM} fp32 "
                f"({self.LAYERS * self.DIM * self.DIM * 4 >> 20} MiB)</li>")


WORKLOAD = LatencyClient
