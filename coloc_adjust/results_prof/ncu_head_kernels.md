# Nsight Compute, per kernel: three segments of one tenant step (2026-11-21)

Usual tenant, step 12, green context of 32 SMs, no MPS (Nsight Compute does not separate MPS clients). Counters `dram__bytes_read.sum` and
`dram__bytes_write.sum` of every kernel launched inside the segment, caches not flushed between kernels (`--cache-control none`); about 0.19 s per kernel.
The trial used a wrapper around `ft_worker.py` that opened one NVTX start/end range per segment; `ncu --nvtx --nvtx-include <range>` selected the kernels.
The wrapper was a scratch script and is not in the repository; `ft_worker.py --nvtx` now emits marks, not ranges.

| Segment | Kernels | Read GB | Written GB | Total GB | Same segment from Nsight Systems (`segment_bytes.py`) |
|---|---|---|---|---|---|
| last block's forward, output layer, loss, their backward (the 8 GB stretch) | 106 | 4.829 | 3.155 | 7.984 | 7.97 |
| one block, forward | 70 | 0.181 | 0.000 | 0.181 | 0.11 |
| one block, backward | 115 | 0.293 | 0.000 | 0.293 | 0.15 |

The large segment agrees (7.98 against 7.97 GB). For a single block the per-kernel counters read 1.6 to 1.9 times what the device sampler reads, and no
writes at all: the block's tensors are a few MB, and run one kernel at a time with pauses between them they behave differently in the cache than in the
running tenant. The per-segment table therefore comes from Nsight Systems; Nsight Compute is used only to see which kernels a large segment is made of.

## Kernels of the 8 GB stretch that move more than 20 MB, in launch order

| Kernel | Read GB | Written GB | Total GB | Kernel time ms |
|---|---|---|---|---|
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(T1::` | 0.275 | 0.315 | 0.590 | 9.10 |
| `void at::unrolled_elementwise_kernel<at::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lam` | 0.311 | 0.626 | 0.937 | 2.16 |
| `void at::<unnamed>::cunn_SoftMaxForward<4, float, float, float, at::<unnamed>::LogSoftMaxForward` | 0.622 | 0.625 | 1.248 | 1.74 |
| `void at::vectorized_elementwise_kernel<4, at::FillFunctor<float>, std::array<char *, 1>>(int, T2` | 0.000 | 0.619 | 0.620 | 0.61 |
| `void at::<unnamed>::cunn_SoftMaxBackward<4, float, float, float, at::<unnamed>::LogSoftMaxBackwa` | 1.352 | 0.634 | 1.986 | 1.58 |
| `void at::vectorized_elementwise_kernel<4, at::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &` | 0.622 | 0.321 | 0.944 | 0.63 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_nn_align8>(T1::P` | 1.420 | 0.014 | 1.434 | 8.98 |

These seven kernels are 7.76 of the segment's 7.98 GB. Kernel time is under the profiler and is longer than in the running tenant.
