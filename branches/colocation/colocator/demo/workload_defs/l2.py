"""l2 — streaming copy of an L2-resident array: saturates L2 bandwidth."""

import torch
import colocator_kernels

from _probe_common import ProbeBase


class L2Probe(ProbeBase):
    name = "l2"
    THREADS = 512
    resource = "L2 cache bandwidth (GPU-wide)"
    N = 4 * 1024 * 1024   # 16 MiB fp32 array — resident in the 96 MiB L2
    IT = 600              # passes per launch -> a few ms/launch

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
                f"{self.N * 4 >> 20} MiB fp32 array (L2-resident, DRAM barely touched); "
                f"~{self.IT * self.N * 8 / 1e9:.0f} GB L2 traffic/launch</li>")


WORKLOAD = L2Probe
