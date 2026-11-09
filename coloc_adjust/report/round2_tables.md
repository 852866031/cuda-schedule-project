### Baseline (inference alone, KV follows load): 600 W round 1 vs 500 W round 2

| phase | TPOT p50 600 W | 500 W | queue 600 W | 500 W | GPU-memory busy % | power W |
|---|---|---|---|---|---|---|
| 1 req/s (0 s) | 12.9 | 12.9 | 107 | 108 | 65 | 335 |
| 2 req/s (90 s) | 15.2 | 15.2 | 124 | 124 | 73 | 375 |
| 3 req/s (180 s) | 21.2 | 21.3 | 270 | 271 | 66 | 378 |
| 1 req/s (270 s) | 12.5 | 12.5 | 105 | 105 | 60 | 319 |
| 4 req/s (360 s) | 26.0 | 26.1 | 1228 | 1292 | 60 | 376 |
| 0.5 req/s (420 s) | 12.0 | 12.0 | 94 | 95 | 51 | 284 |
| 2 req/s (480 s) | 15.5 | 15.5 | 123 | 121 | 60 | 354 |
| 3 req/s (510 s) | 20.0 | 20.2 | 159 | 157 | 69 | 367 |
| 2 req/s (540 s) | 18.1 | 18.1 | 134 | 135 | 68 | 372 |
| 1 req/s (570 s) | 12.7 | 12.7 | 109 | 109 | 69 | 352 |
| 1 req/s (600 s) | 12.4 | 12.4 | 100 | 100 | 56 | 307 |

### Decode per scheme (TPOT p50 ms / scheduler queue wait ms), 500 W

| phase | Inference alone, KV follows load (baseline) | Static split: KV 6.7 GiB, FT 6 GiB | Static split with the idle gate (tenant starves) | Adaptive KV + FT waits for idle GPU (gate) | Adaptive KV + FT shares GPU via MPS | Adaptive KV + FT capped at 20 % of SMs (MPS) | Adaptive KV + disjoint SMs: engine 128, FT 42 | Adaptive KV + FT throttled to a 5 % decode-latency SLO | Adaptive KV + FT throttled to a 15 % decode-latency SLO | Adaptive KV + FT SM count from load (prebuilt green contexts) | Inference alone, KV follows load, 25.5 GiB host tier (prefill-scheme baseline) | Adaptive KV + 3B prefill tenant, shared SMs (MPS) | Adaptive KV + 3B prefill tenant capped at 20 % of SMs (MPS) | Adaptive KV + 3B prefill tenant on disjoint SMs: engine 128, tenant 42 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 req/s (0 s) | 12.9 / 108 | 16.9 / 125 | 13.1 / 108 | 13.1 / 108 | 16.4 / 123 | 14.3 / 113 | 16.3 / 122 | 13.3 / 108 | 13.6 / 112 | 14.7 / 114 | 13.0 / 108 | 14.1 / 126 | 13.7 / 112 | 13.6 / 110 |
| 2 req/s (90 s) | 15.2 / 124 | 21.5 / 332 | 15.6 / 172 | 15.3 / 125 | 18.4 / 140 | 17.0 / 134 | 17.5 / 137 | 15.5 / 131 | 16.2 / 134 | 17.0 / 135 | 15.3 / 122 | 15.4 / 122 | 15.4 / 123 | 15.3 / 123 |
| 3 req/s (180 s) | 21.3 / 271 | 22.8 / 8164 | 18.6 / 1402 | 21.4 / 301 | 21.4 / 311 | 21.6 / 317 | 21.4 / 381 | 21.4 / 308 | 21.5 / 326 | 21.4 / 314 | 21.3 / 243 | 21.5 / 300 | 21.4 / 269 | 21.5 / 302 |
| 1 req/s (270 s) | 12.5 / 105 | 17.3 / 8489 | 12.7 / 106 | 12.6 / 106 | 16.1 / 123 | 13.8 / 114 | 16.0 / 123 | 12.8 / 107 | 13.8 / 114 | 14.3 / 115 | 12.5 / 104 | 13.3 / 142 | 13.7 / 110 | 13.4 / 108 |
| 4 req/s (360 s) | 26.1 / 1292 | 22.9 / 13472 | 18.6 / 9225 | 25.2 / 2262 | 25.0 / 2302 | 25.7 / 2940 | 24.1 / 3367 | 22.3 / 2425 | 24.5 / 2277 | 24.5 / 2216 | 27.5 / 1303 | 24.9 / 3360 | 25.3 / 2176 | 25.4 / 6659 |
| 0.5 req/s (420 s) | 12.0 / 95 | 21.8 / 26176 | 12.5 / 10871 | 12.1 / 94 | 15.4 / 105 | 13.2 / 98 | 15.3 / 106 | 12.8 / 97 | 12.5 / 100 | 13.7 / 100 | 12.1 / 94 | 12.7 / 110 | 12.5 / 97 | 12.6 / 469 |
| 2 req/s (480 s) | 15.5 / 121 | 20.9 / 145 | 15.8 / 124 | 15.7 / 124 | 18.8 / 141 | 17.5 / 142 | 18.9 / 164 | 16.7 / 131 | 17.2 / 138 | 17.3 / 132 | 15.5 / 124 | 15.5 / 122 | 15.7 / 125 | 15.5 / 122 |
| 3 req/s (510 s) | 20.2 / 157 | 23.2 / 1719 | 18.6 / 636 | 20.3 / 175 | 21.4 / 203 | 20.8 / 191 | 21.3 / 234 | 20.2 / 304 | 20.3 / 233 | 20.5 / 186 | 21.3 / 194 | 21.2 / 219 | 21.4 / 210 | 21.1 / 221 |
| 2 req/s (540 s) | 18.1 / 135 | 23.1 / 4405 | 18.1 / 262 | 18.3 / 138 | 19.5 / 140 | 19.4 / 140 | 19.5 / 138 | 18.3 / 144 | 18.6 / 138 | 19.0 / 138 | 18.1 / 135 | 18.2 / 134 | 18.2 / 136 | 18.1 / 135 |
| 1 req/s (570 s) | 12.7 / 109 | 16.6 / 116 | 12.8 / 102 | 12.8 / 109 | 16.7 / 117 | 14.2 / 108 | 16.6 / 115 | 13.3 / 110 | 14.5 / 115 | 14.6 / 114 | 12.9 / 104 | 13.5 / 143 | 14.6 / 100 | 14.1 / 109 |
| 1 req/s (600 s) | 12.4 / 100 | 16.0 / 119 | 12.4 / 101 | 12.5 / 101 | 16.0 / 120 | 13.7 / 110 | 16.0 / 120 | 12.8 / 100 | 14.1 / 108 | 14.3 / 108 | 12.4 / 98 | 12.8 / 106 | 12.8 / 102 | 12.7 / 98 |

### Tenant per scheme (tok/s; prefill schemes: prompt tok/s), 500 W

| phase | Static split: KV 6.7 GiB, FT 6 GiB | Static split with the idle gate (tenant starves) | Adaptive KV + FT waits for idle GPU (gate) | Adaptive KV + FT shares GPU via MPS | Adaptive KV + FT capped at 20 % of SMs (MPS) | Adaptive KV + disjoint SMs: engine 128, FT 42 | Adaptive KV + FT throttled to a 5 % decode-latency SLO | Adaptive KV + FT throttled to a 15 % decode-latency SLO | Adaptive KV + FT SM count from load (prebuilt green contexts) | Inference alone, KV follows load, 25.5 GiB host tier (prefill-scheme baseline) | Adaptive KV + 3B prefill tenant, shared SMs (MPS) | Adaptive KV + 3B prefill tenant capped at 20 % of SMs (MPS) | Adaptive KV + 3B prefill tenant on disjoint SMs: engine 128, tenant 42 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 req/s (0 s) | 12.2k | 0.1k | 0.1k | 9.8k | 6.3k | 9.8k | 0.6k | 1.8k | 8.3k |  | 4.1k | 4.4k | 4.1k |
| 2 req/s (90 s) | 12.8k | 0.1k | 0.0k | 3.4k | 2.8k | 3.2k | 0.6k | 1.3k | 2.8k |  | 0.0k | 0.0k | 0.0k |
| 3 req/s (180 s) | 12.7k | 0.1k | 0.0k | 0.0k | 0.0k | 0.0k | 0.0k | 0.1k | 0.0k |  | 0.0k | 0.0k | 0.0k |
| 1 req/s (270 s) | 12.5k | 0.0k | 0.0k | 12.6k | 6.8k | 12.8k | 0.8k | 4.0k | 9.7k |  | 7.1k | 9.5k | 7.3k |
| 4 req/s (360 s) | 12.9k | 0.1k | 0.0k | 0.0k | 0.0k | 0.0k | 0.0k | 0.0k | 0.0k |  | 0.0k | 0.0k | 2.0k |
| 0.5 req/s (420 s) | 13.2k | 0.0k | 0.0k | 14.5k | 7.0k | 13.7k | 3.8k | 3.8k | 11.9k |  | 9.3k | 8.9k | 9.3k |
| 2 req/s (480 s) | 12.7k | 0.0k | 0.0k | 4.3k | 2.9k | 4.3k | 0.4k | 3.1k | 3.3k |  | 0.0k | 1.7k | 0.0k |
| 3 req/s (510 s) | 13.1k | 0.2k | 0.0k | 0.4k | 0.6k | 0.4k | 0.2k | 0.2k | 0.4k |  | 0.0k | 0.0k | 0.0k |
| 2 req/s (540 s) | 13.1k | 0.0k | 0.0k | 1.2k | 0.8k | 1.2k | 0.2k | 0.2k | 0.8k |  | 0.0k | 0.0k | 0.0k |
| 1 req/s (570 s) | 10.9k | 0.0k | 0.0k | 11.5k | 6.1k | 11.1k | 1.2k | 5.3k | 8.8k |  | 10.9k | 11.1k | 13.4k |
| 1 req/s (600 s) | 13.1k | 0.1k | 0.0k | 12.8k | 6.9k | 12.8k | 2.0k | 5.0k | 10.3k |  | 2.6k | 3.4k | 3.0k |

### VRAM-scheduler accounting per scheme (from its log)

| scheme | engine fixed GiB | max limit GiB | tenant > budget (s) | KV used + tenant + fixed > total (s) | grow_wait (s) | grow_partial | shrinks | tenant steps (preempted) |
|---|---|---|---|---|---|---|---|---|
| Inference alone, KV follows load (baseline) | 16.97 | 12.65 | 0.0 | 0.0 | 0.0 | 0 | 28 |  |
| Static split: KV 6.7 GiB, FT 6 GiB | 16.97 | 6.69 | 0.0 | 0.0 | 0.0 | 0 | 0 | 2089 (0) |
| Static split with the idle gate (tenant starves) | 16.97 | 6.69 | 0.0 | 0.0 | 0.0 | 0 | 0 | 8 (0) |
| Adaptive KV + FT waits for idle GPU (gate) | 16.97 | 12.00 | 194.0 | 0.0 | 97.6 | 7 | 28 | 2 (21) |
| Adaptive KV + FT shares GPU via MPS | 16.97 | 12.00 | 198.9 | 0.0 | 99.7 | 9 | 30 | 1124 (26) |
| Adaptive KV + FT capped at 20 % of SMs (MPS) | 16.97 | 12.50 | 197.1 | 0.0 | 95.7 | 15 | 30 | 640 (24) |
| Adaptive KV + disjoint SMs: engine 128, FT 42 | 16.97 | 11.00 | 207.6 | 0.0 | 111.2 | 21 | 29 | 1115 (26) |
| Adaptive KV + FT throttled to a 5 % decode-latency SLO | 16.97 | 12.00 | 205.5 | 0.0 | 105.2 | 17 | 28 | 141 (6) |
| Adaptive KV + FT throttled to a 15 % decode-latency SLO | 16.97 | 12.00 | 196.9 | 0.0 | 101.6 | 15 | 27 | 340 (7) |
| Adaptive KV + FT SM count from load (prebuilt green contexts) | 16.97 | 12.00 | 175.8 | 0.0 | 98.4 | 16 | 29 | 898 (24) |
| Inference alone, KV follows load, 25.5 GiB host tier (prefill-scheme baseline) | 17.85 | 12.65 | 0.0 | 0.0 | 0.0 | 0 | 28 |  |
| Adaptive KV + 3B prefill tenant, shared SMs (MPS) | 21.46 | 12.00 | 641.2 | 101.0 | 98.8 | 7 | 30 |  |
| Adaptive KV + 3B prefill tenant capped at 20 % of SMs (MPS) | 21.02 | 12.50 | 629.7 | 88.6 | 92.1 | 9 | 27 |  |
| Adaptive KV + 3B prefill tenant on disjoint SMs: engine 128, tenant 42 | 21.50 | 12.00 | 633.3 | 103.5 | 101.8 | 7 | 27 |  |

### GPU-memory-controller busy and power at 1 and 2 req/s (phase means)

| scheme | GPU-memory busy % @1 | @2 | power W @1 | @2 | tenant tok/s @1 | @2 |
|---|---|---|---|---|---|---|
| Inference alone, KV follows load (baseline) | 65 | 73 | 335 | 375 |  |  |
| Static split: KV 6.7 GiB, FT 6 GiB | 71 | 72 | 424 | 440 | 12235 | 12800 |
| Static split with the idle gate (tenant starves) | 68 | 73 | 344 | 383 | 51 | 51 |
| Adaptive KV + FT waits for idle GPU (gate) | 64 | 76 | 337 | 372 | 51 | 0 |
| Adaptive KV + FT shares GPU via MPS | 71 | 69 | 407 | 388 | 9829 | 3379 |
| Adaptive KV + FT capped at 20 % of SMs (MPS) | 74 | 69 | 405 | 394 | 6298 | 2765 |
| Adaptive KV + disjoint SMs: engine 128, FT 42 | 71 | 72 | 411 | 390 | 9778 | 3226 |
| Adaptive KV + FT throttled to a 5 % decode-latency SLO | 68 | 71 | 340 | 377 | 614 | 563 |
| Adaptive KV + FT throttled to a 15 % decode-latency SLO | 68 | 71 | 347 | 374 | 1792 | 1331 |
| Adaptive KV + FT SM count from load (prebuilt green contexts) | 75 | 72 | 405 | 391 | 8345 | 2816 |
| Inference alone, KV follows load, 25.5 GiB host tier (prefill-scheme baseline) | 63 | 78 | 337 | 379 |  |  |
| Adaptive KV + 3B prefill tenant, shared SMs (MPS) | 60 | 73 | 355 | 377 | 4147 | 0 |
| Adaptive KV + 3B prefill tenant capped at 20 % of SMs (MPS) | 65 | 76 | 376 | 379 | 4378 | 0 |
| Adaptive KV + 3B prefill tenant on disjoint SMs: engine 128, tenant 42 | 66 | 74 | 373 | 379 | 4147 | 0 |
