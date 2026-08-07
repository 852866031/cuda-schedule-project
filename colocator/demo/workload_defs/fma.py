"""fma — high-ILP FP32 FMA loop: saturates the FMA pipe and issue slots."""

import torch
import colocator_kernels

from _probe_common import ProbeBase


class FmaProbe(ProbeBase):
    name = "fma"
    THREADS = 128
    resource = "FP32 FMA pipeline + warp-scheduler issue slots (per-SMSP)"
    IT = 5_000_000  # 4 independent FMA chains per inner iteration -> ~15-20 ms/launch

    def alloc(self):
        self.a = torch.rand(self.THREADS, device=self.device)
        self.b = torch.rand(self.THREADS, device=self.device)
        self.c = torch.zeros(self.THREADS, device=self.device)

    def launch(self):
        colocator_kernels.probe_fma32(self.a, self.b, self.c, self.IT,
                                      self.blocks, self.THREADS)

    def check(self):
        self.run_iter_sync()
        return bool(torch.isfinite(self.c).all())

    def params_html(self):
        flops = self.IT * 4 * 2 * self.THREADS * self.blocks
        return (self.geometry_html() +
                f"<li>kernel: k_fma32 — {self.IT:,} × 4 independent FP32 FMA chains "
                f"(high ILP, registers only): ~{flops / 1e12:.1f} TFLOP/launch, "
                f"no memory traffic after the first cache line</li>")


WORKLOAD = FmaProbe
