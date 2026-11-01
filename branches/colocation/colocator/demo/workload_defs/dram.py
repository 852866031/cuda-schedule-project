"""dram — streaming copy of a >L2 array: saturates DRAM bandwidth."""

import torch
import colocator_kernels

from _probe_common import ProbeBase


class DramProbe(ProbeBase):
    name = "dram"
    THREADS = 512
    resource = "DRAM bandwidth (GPU-wide)"
    N = 128 * 1024 * 1024   # 512 MiB fp32 array, far larger than the 96 MiB L2
    IT = 16                 # passes per launch -> ~11 ms/launch

    def alloc(self):
        self.inp = torch.rand(self.N, device=self.device)
        self.out = torch.zeros(self.N, device=self.device)

    def launch(self):
        colocator_kernels.probe_copy(self.inp, self.out, self.IT, self.blocks, self.THREADS)

    def check(self):
        self.run_iter_sync()
        return bool(torch.equal(self.out, self.inp))

    def params_html(self):
        return (self.geometry_html() +
                f"<li>kernel: k_copy (grid-stride) — {self.IT} passes over a "
                f"{self.N * 4 >> 20} MiB fp32 array (&gt; L2, so every access hits DRAM); "
                f"~{self.IT * self.N * 8 / 1e9:.0f} GB traffic/launch</li>")


WORKLOAD = DramProbe
