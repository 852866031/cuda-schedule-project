"""fp64 — high-ILP FP64 FMA loop: saturates the (narrow) FP64 pipe."""

import torch
import colocator_kernels

from _probe_common import ProbeBase


class Fp64Probe(ProbeBase):
    name = "fp64"
    THREADS = 128
    resource = "FP64 pipeline (heavily cut down on GeForce: 1/64 rate)"
    IT = 150_000

    def alloc(self):
        self.a = torch.rand(self.THREADS, device=self.device, dtype=torch.float64)
        self.b = torch.rand(self.THREADS, device=self.device, dtype=torch.float64)
        self.c = torch.zeros(self.THREADS, device=self.device, dtype=torch.float64)

    def launch(self):
        colocator_kernels.probe_fma64(self.a, self.b, self.c, self.IT,
                                      self.blocks, self.THREADS)

    def check(self):
        self.run_iter_sync()
        return bool(torch.isfinite(self.c).all())

    def params_html(self):
        return (self.geometry_html() +
                f"<li>kernel: k_fma64 — {self.IT:,} × 4 independent FP64 FMA chains; "
                f"stresses only the FP64 units, which GeForce parts run at 1/64 of "
                f"FP32 rate — long kernel for little math</li>")


WORKLOAD = Fp64Probe
