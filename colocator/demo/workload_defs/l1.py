"""l1 — per-block copy of a small region: L1-cache-resident per-SM traffic."""

import torch
import colocator_kernels

from _probe_common import ProbeBase


class L1Probe(ProbeBase):
    name = "l1"
    THREADS = 64
    resource = "L1 cache (per-SM working set)"
    NPB = 8192            # floats copied per block: 32 KiB working set
    REGION = 128 * 1024   # region stride per block (bytes)
    IT = 15000            # passes per launch

    def alloc(self):
        fpr = self.REGION // 4
        rpt = (self.NPB + fpr - 1) // fpr
        total = self.blocks * fpr * rpt
        self.inp = torch.rand(total, device=self.device)
        self.out = torch.zeros(total, device=self.device)

    def launch(self):
        colocator_kernels.probe_copy_block(self.inp, self.out, self.NPB, self.REGION,
                                           self.IT, self.blocks, self.THREADS)

    def check(self):
        self.run_iter_sync()
        fpr = self.REGION // 4
        ok = True
        for b in range(0, self.blocks, 37):  # spot-check a spread of blocks
            b0 = b * fpr
            ok &= bool(torch.equal(self.out[b0:b0 + self.NPB], self.inp[b0:b0 + self.NPB]))
        return ok

    def params_html(self):
        return (self.geometry_html() +
                f"<li>kernel: k_copy_tb — each block copies its own "
                f"{self.NPB * 4 // 1024} KiB region ({self.IT:,} passes/launch): "
                f"working set fits in L1, so traffic stays inside each SM</li>")


WORKLOAD = L1Probe
