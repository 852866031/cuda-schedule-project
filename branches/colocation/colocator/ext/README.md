# ext/ — colocator_kernels torch extension

Custom CUDA kernels for the demo workloads. One module, one op:

- `colocator_kernels.matmul(a, b)` — fp32 2-D matmul, tiled shared-memory
  SGEMM (`csrc/matmul.cu`, 32×32 tiles, one output element per thread, bounds
  checked, launched on the caller's current torch stream).

## Why not `torch.matmul`?

`torch.matmul` dispatches to cuBLAS. cuBLAS launches its kernels internally
through the CUDA **driver** API, so they never pass through the
`cudaLaunchKernel` runtime symbol that `libcolocator.so` interposes — the
colocator would be blind to them (Orion solves this by interposing cuBLAS
entry points; see PROPOSAL.md §5 R1). A `__global__` kernel launched from this
extension goes through `cudaLaunchKernel` like any ATen native kernel, so the
colocator sees and schedules it.

Performance is intentionally "honest but naive" (~6–9 TFLOPS fp32 on RTX
5090): kernel duration scales with problem size, giving the demo short kernels
(latency client) and long SM-saturating kernels (throughput client).

## Build

```bash
cd colocator/ext && python setup.py build_ext --inplace
```

Produces `colocator_kernels.cpython-*.so` in this directory (arch auto-detected
from the local GPU, sm_120 on RTX 5090). `demo/workloads.py` adds `ext/` to
`sys.path` and imports it directly.
