# Bytes per segment of a tenant step (2026-11-22)

Tenant alone, green context of 32 SMs, Nsight Systems GPU metrics at 10000 Hz; 1 % = 17.63 GB/s read, 17.63 GB/s write.
In 'blocks' the last block's forward segment also holds whatever follows it without a checkpoint (with the model's own loss: the whole output layer and loss).
A compiled tenant has no checkpoints in its body: 'blocks' is 0, and 'head' counts only the two passes over the vocabulary (blank where the head cannot be told apart).

| variant | step ms | tokens/s | GB per step | MB per token | blocks forward GB | blocks backward GB | head segments GB | largest segment | its GB | its ms | its GB/s | loss at step 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 356.0 | 11511 | 57.17 | 13.95 | 41.89 | 15.13 | 0.00 | blk23:fwd | 7.969 | 19.27 | 413 | 12.2168 |
| split | 358.5 | 11425 | 57.00 | 13.92 | 10.39 | 15.03 | 31.44 | head:logsoftmax_bwd | 2.037 | 1.44 | 1414 | 12.2165 |
| vocab1024 | 471.5 | 8687 | 29.60 | 7.23 | 10.40 | 15.06 | 3.99 | blk3:bwd | 0.176 | 1.58 | 111 | 12.2154 |
| vocab2048 | 429.8 | 9533 | 30.42 | 7.42 | 10.40 | 15.07 | 4.80 | blk3:bwd | 0.177 | 1.58 | 112 | 12.2165 |
| vocab4096 | 410.7 | 9975 | 35.00 | 8.54 | 10.39 | 15.56 | 8.91 | blk6:bwd | 0.179 | 1.61 | 111 | 12.2164 |
| vocab8192 | 402.2 | 10184 | 40.41 | 9.87 | 10.41 | 15.17 | 14.68 | blk8:bwd | 0.171 | 1.60 | 107 | 12.2165 |
| vocab16384 | 397.9 | 10297 | 53.08 | 12.95 | 10.40 | 15.10 | 27.43 | head:pass2 | 0.420 | 1.85 | 227 | 12.2168 |
| vocab32768 | 409.8 | 9996 | 71.44 | 17.44 | 10.41 | 15.08 | 45.80 | head:pass2 | 1.471 | 4.11 | 358 | 12.2168 |
| compile | 286.5 | 14294 | 34.38 | 8.39 | 0.00 | 0.00 |  | mb_end | 8.582 | 72.63 | 118 | 12.2309 |
| compile_split | 301.3 | 13593 | 51.68 | 12.62 | 0.00 | 0.00 |  | head:linear_bwd | 4.562 | 36.46 | 125 | 12.2316 |
| compile_vocab4096 | 354.5 | 11553 | 28.58 | 6.98 | 0.00 | 0.00 | 9.06 | head:surrogate | 2.934 | 30.90 | 95 | 12.2314 |
| compile_vocab2048 | 373.9 | 10957 | 24.16 | 5.90 | 0.00 | 0.00 | 4.60 | head:surrogate | 2.939 | 30.94 | 95 | 12.2317 |
| cb | 317.6 | 12897 | 50.95 | 12.44 | 39.95 | 10.86 | 0.00 | blk23:fwd | 8.133 | 19.24 | 423 | 12.2319 |
| cb_vocab4096 | 372.9 | 10984 | 28.08 | 6.86 | 7.88 | 10.77 | 9.29 | blk0:bwd | 0.127 | 1.30 | 97 | 12.2314 |
| cb_vocab2048 | 391.6 | 10460 | 23.00 | 5.62 | 7.85 | 10.69 | 4.33 | blk5:bwd | 0.121 | 1.27 | 95 | 12.2312 |
| vocab4096f | 395.1 | 10367 | 29.02 | 7.09 | 10.46 | 15.07 | 3.34 | blk3:bwd | 0.176 | 1.59 | 111 | 12.2338 |
| vocab2048f | 415.3 | 9863 | 28.93 | 7.06 | 10.45 | 15.02 | 3.31 | blk3:bwd | 0.176 | 1.58 | 111 | 12.2334 |
| vocab1024f | 454.6 | 9010 | 29.28 | 7.15 | 10.43 | 15.03 | 3.68 | blk3:bwd | 0.175 | 1.59 | 111 | 12.2332 |
| cb_vocab4096f | 354.5 | 11555 | 21.87 | 5.34 | 7.83 | 10.76 | 3.16 | blk11:bwd | 0.122 | 1.27 | 96 | 12.2314 |
| cb_vocab2048f | 374.9 | 10925 | 21.99 | 5.37 | 7.81 | 10.75 | 3.30 | blk11:bwd | 0.123 | 1.27 | 96 | 12.2312 |
| cb_vocab4096f_t1 | 298.1 | 13741 | 20.48 | 5.00 | 6.69 | 11.23 | 2.45 | blk7:bwd | 0.242 | 2.14 | 113 | 12.2300 |
| cb_vocab4096f_t3 | 444.0 | 9225 | 29.13 | 7.11 | 9.95 | 13.33 | 5.68 | mb_end | 0.078 | 1.54 | 51 | 12.2313 |
| cb_vocab4096f_t0 | 268.7 | 15243 | 26.90 | 6.57 | 8.77 | 14.43 | 3.68 | blk0:bwd | 0.676 | 5.45 | 124 | 12.2307 |
