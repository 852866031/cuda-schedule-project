"""throughput — batch-like client: one SM-saturating SGEMM + big copies."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ext"))
import colocator_kernels  # noqa: E402

from base import WorkloadBase  # noqa: E402


class ThroughputClient(WorkloadBase):
    name = "throughput"
    DIM = 4096

    def setup(self):
        g = torch.Generator().manual_seed(1)
        self.a_host = torch.randn(self.DIM, self.DIM, generator=g) * (self.DIM ** -0.5)
        self.shard_host = (torch.randn(self.DIM, self.DIM, generator=g)
                           * (self.DIM ** -0.5)).pin_memory()
        self.out_host = [torch.empty(self.DIM, pin_memory=True)
                         for _ in range(self.DEPTH)]  # one per pipeline slot
        self.a = self.a_host.to(self.device)

    def launch_iter(self):
        shard = self.shard_host.to(self.device, non_blocking=True)  # 64 MiB H2D
        c = colocator_kernels.matmul(self.a, shard)                 # big SGEMM
        out = c.sum(dim=1)
        dst = self.out_host[self._slot]
        dst.copy_(out, non_blocking=True)
        return dst

    def reference(self):
        return (self.a_host.double() @ self.shard_host.double()).sum(dim=1)

    def check(self):
        got = self.run_iter_sync().double().clone()
        ok = torch.allclose(got, self.reference(), rtol=1e-3, atol=1e-1)
        if not ok:
            print(f"[{self.name}] check mismatch")
        return ok

    def summary(self):
        return ("Batch-like <b>throughput</b> client: per iteration, copy a 64 MiB weight "
                "shard H2D, run one huge SM-saturating tiled SGEMM, row-sum, retrieve D2H. "
                "Few very long kernels; iterations/sec is its metric.")

    def params_html(self):
        g = self.DIM // 32
        return (f"<li>sgemm_tiled ×1/iter: C({self.DIM}×{self.DIM}) = "
                f"A({self.DIM}×{self.DIM}) @ W({self.DIM}×{self.DIM}), "
                f"grid ({g}×{g}) = {g * g} blocks, block (32×32) — ~16 ms, fills all SMs</li>"
                f"<li>+ 1 row-sum (reduce) per iter</li>"
                f"<li>copies: {self.DIM * self.DIM * 4 >> 20} MiB H2D shard in, "
                f"{self.DIM * 4 / 1024:.0f} KiB D2H out (pinned)</li>")


WORKLOAD = ThroughputClient
