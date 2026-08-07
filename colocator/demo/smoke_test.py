#!/usr/bin/env python
"""Phase 1 smoke test: exercise the interception set with unmodified PyTorch.

Runs a few iterations of H2D copy -> ATen elementwise kernels -> D2H retrieve
(no matmul/conv, so no cuBLAS/cuDNN paths) and checks results against CPU.

Run it twice and compare behavior:

    python colocator/demo/smoke_test.py                                 # bare
    LD_PRELOAD=$PWD/colocator/build/libcolocator.so \
        python colocator/demo/smoke_test.py                             # intercepted

Both must print OK; the intercepted run must additionally show a
"[colocator] intercept summary" with nonzero kernel and memcpy counts.
"""

import sys

import torch

ITERS = 5
N = 1 << 20  # 1M floats


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    torch.manual_seed(0)
    dev = torch.device("cuda:0")

    for it in range(ITERS):
        x_host = torch.randn(N, pin_memory=True)

        x = x_host.to(dev, non_blocking=True)          # H2D (cudaMemcpyAsync)
        y = torch.relu(x) * 2.0 + 0.5                  # elementwise kernels
        z = y.sigmoid()
        total = z.sum()                                # reduction kernel

        got = total.item()                             # D2H + sync
        ref = (torch.relu(x_host.float()) * 2.0 + 0.5).sigmoid().sum().item()

        if abs(got - ref) > 1e-2 * max(1.0, abs(ref)):
            print(f"MISMATCH at iter {it}: gpu={got} cpu={ref}")
            sys.exit(1)

    torch.cuda.synchronize()
    print(f"OK ({ITERS} iters, {N} elems)")


if __name__ == "__main__":
    main()
