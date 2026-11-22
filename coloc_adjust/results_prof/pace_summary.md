# The paced tenant alone: bytes against the budget (2026-11-21)

Tenant alone on a green context of 32 SMs; memory read + write throughput from Nsight Systems at 10 kHz (1 % = 17.63 GB/s).
Case = variant_mode: free = no budget; step = byte ledger, one pause after the step; wN = byte ledger, chunks of budget x N ms; blockN = the pacer of Nov 16.

| case | budget GB/s | period ms | expected ms | GB per step | mean GB/s | tok/s | 12 ms windows: p50 / p90 / p99 / max GB/s | 100 ms windows: p50 / p90 / max | time over 200 / 500 GB/s | pauses per step | late per step (max ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base_free |  | 350.9 |  | 57.00 | 162.5 | 11673 | 99 / 490 / 606 / 610 | 155 / 209 / 217 | 17.5 % / 6.25 % | 0 | 0.0 (0.0) |
| base_step | 50 | 1144.6 | 1143.4 | 60.47 | 52.8 | 3579 | 5 / 101 / 587 / 610 | 4 / 155 / 214 | 5.4 % / 1.92 % | 0 | 0.0 (0.0) |
| base_w12 | 50 | 1143.8 | 1143.4 | 60.56 | 52.9 | 3581 | 43 / 63 / 580 / 610 | 51 / 110 / 123 | 5.4 % / 1.92 % | 46 | 1.0 (1.5) |
| vocab4096_free |  | 399.8 |  | 34.85 | 87.2 | 10244 | 84 / 103 / 106 / 108 | 87 / 88 / 88 | 4.6 % / 0.01 % | 0 | 0.0 (0.0) |
| vocab4096_step | 50 | 700.9 | 700.0 | 36.44 | 52.0 | 5844 | 72 / 102 / 106 / 107 | 81 / 87 / 88 | 2.6 % / 0.01 % | 0 | 0.0 (0.0) |
| vocab4096_w12 | 50 | 700.5 | 700.0 | 37.79 | 54.0 | 5848 | 54 / 61 / 66 / 72 | 54 / 56 / 57 | 2.6 % / 0.00 % | 58 | 1.0 (1.7) |
| vocab4096_w6 | 50 | 700.5 | 700.0 | 37.80 | 54.0 | 5848 | 54 / 60 / 66 / 70 | 54 / 55 / 56 | 2.6 % / 0.00 % | 116 | 1.1 (1.8) |
| vocab2048_w12 | 50 | 609.0 | 608.3 | 32.03 | 52.6 | 6726 | 53 / 69 / 95 / 102 | 53 / 57 / 59 | 2.8 % / 0.00 % | 38 | 12.0 (10.6) |
| base_block1 | 50 | 1163.1 | 1143.4 | 65.29 | 56.1 | 3522 | 31 / 44 / 578 / 608 | 34 / 107 / 108 | 5.3 % / 1.89 % | 193 | 0.0 (0.0) |
