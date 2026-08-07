"""sleep — occupies SM residency slots without using memory or compute."""

import colocator_kernels

from _probe_common import ProbeBase


class SleepProbe(ProbeBase):
    name = "sleep"
    THREADS = 768
    resource = "scheduler / SM occupancy only (nanosleep loop — no memory, no compute)"
    IT = 20000  # ~1 µs nanosleep per inner iteration -> ~20 ms/launch

    def launch(self):
        colocator_kernels.probe_sleep(self.blocks, self.THREADS, self.IT)

    def params_html(self):
        return (self.geometry_html() +
                f"<li>kernel: k_sleep — {self.IT:,} × nanosleep(1 µs); "
                f"holds {self.THREADS} threads/SM resident doing nothing</li>")


WORKLOAD = SleepProbe
