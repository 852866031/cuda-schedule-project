# One tenant micro-batch, operator by operator (2 x 512 tokens, LoRA r=16, all layers)

Model Qwen/Qwen2.5-0.5B: 24 layers, hidden 896, attention implementation `sdpa`; all parameters = 946 MB at 2 bytes per element. Counted on the CPU; no GPU was used.

| pass | operator calls that move data | weights read | intermediates read | written | total | per token |
|---|---|---|---|---|---|---|
| forward | 1,318 | 1,206 MB | 3,114 MB | 2,573 MB | 6,893 MB | 6.73 MB |
| backward | 1,984 | 946 MB | 6,000 MB | 3,533 MB | 10,479 MB | 10.23 MB |
| all | 3,302 | 2,152 MB | 9,114 MB | 6,106 MB | 17,372 MB | 16.96 MB |

Weights are 12 % of the bytes; intermediate results (read and written) are 88 %.

## The operators that move the most

| pass | operator | calls | weights read | intermediates read | written | share of all bytes |
|---|---|---|---|---|---|---|
| backward | `mm.default` | 553 | 946 MB | 1,260 MB | 657 MB | 16.5 % |
| backward | `mul.Tensor` | 485 | 0 MB | 1,625 MB | 991 MB | 15.1 % |
| forward | `mm.default` | 289 | 899 MB | 527 MB | 936 MB | 13.6 % |
| backward | `add.Tensor` | 361 | 0 MB | 1,120 MB | 560 MB | 9.7 % |
| forward | `mul.Tensor` | 317 | 0 MB | 826 MB | 592 MB | 8.2 % |
| backward | `_log_softmax_backward_data.default` | 1 | 0 MB | 594 MB | 297 MB | 5.1 % |
| forward | `add.Tensor` | 242 | 0 MB | 456 MB | 228 MB | 3.9 % |
| backward | `silu_backward.default` | 24 | 0 MB | 456 MB | 228 MB | 3.9 % |
| backward | `nll_loss_backward.default` | 1 | 0 MB | 297 MB | 297 MB | 3.4 % |
| forward | `_log_softmax.default` | 1 | 0 MB | 297 MB | 297 MB | 3.4 % |
| forward | `silu.default` | 24 | 0 MB | 228 MB | 228 MB | 2.6 % |
| forward | `nll_loss_forward.default` | 1 | 0 MB | 297 MB | 0 MB | 1.7 % |
| forward | `embedding.default` | 1 | 260 MB | 0 MB | 2 MB | 1.5 % |
| forward | `addmm.default` | 72 | 47 MB | 126 MB | 54 MB | 1.3 % |
| backward | `_scaled_dot_product_flash_attention_for_cpu_backward.default` | 24 | 0 MB | 139 MB | 54 MB | 1.1 % |
| backward | `pow.Tensor_Scalar` | 98 | 0 MB | 86 MB | 86 MB | 1.0 % |
| backward | `mul.Scalar` | 98 | 0 MB | 86 MB | 86 MB | 1.0 % |
| forward | `pow.Tensor_Scalar` | 49 | 0 MB | 86 MB | 86 MB | 1.0 % |

## Variants (same count, one run each)

| variant | operator calls that move data | MB per token | forward / backward | weights' share |
|---|---|---|---|---|
| as the tenant runs (2 x 512 tokens, LoRA on all 24 layers, sdpa) | 3,302 | 16.96 | 6.73 / 10.23 | 12 % |
| `--forward-only` | 1,318 | 6.73 | 6.73 / 0 | 17 % |
| `--top-layers 8` | 1,710 | 10.92 | 6.24 / 4.68 | 15 % |
| `--top-layers 4` | 1,318 | 9.43 | 6.12 / 3.32 | 16 % |
| `--seqs 8` (one micro-batch) | 3,302 | 15.38 | 5.84 / 9.53 | 3 % |
| `--seqs 1` (eight micro-batches) | 3,301 | 19.08 | 7.92 / 11.17 | 22 % |
| `--attn eager` | 3,720 | 23.29 | 9.78 / 13.51 | 9 % |

Measured on the GPU for the first row's shape (k = 4): 13.4–14.3 MB per token (`knee_bw2_summary.md`).
