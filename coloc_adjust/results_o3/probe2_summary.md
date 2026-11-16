# R5: every tenant shape under the same byte allowance (probe2), real engine at 1 req/s

No-tenant windows: 12.53 ms, 12.36 ms, 12.70 ms (mean token gap; all three together 12.53 ms, p90 12.50).

| shape | pass | tenant tok/s | while computing | share computing | GB/s while computing | GB/s on average | decode gap mean | p90 | while tenant computes | while it does not | mean, same number of running requests | per 1 k tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 micro-batches | 1 | 3,701 | 5,986 | 62 % | 81 | 50 | +5.1 % | +7.4 % | +7.2 % | +0.5 % | +4.7 % | 1.38 % |
| 8 micro-batches | 1 | 3,317 | 4,374 | 76 % | 65 | 49 | +7.9 % | +10.7 % | +8.5 % | +2.5 % | +7.1 % | 2.39 % |
| 8 mb + activations in host RAM | 1 | 2,078 | 2,081 | 100 % | 32 | 32 | +10.5 % | +19.6 % | +10.5 % | – | +9.8 % | 5.07 % |
| + 12 layers in host RAM | 1 | 1,824 | 1,826 | 100 % | 30 | 30 | +11.7 % | +21.4 % | +11.7 % | – | +11.6 % | 6.44 % |
| + 20 layers in host RAM | 1 | 1,663 | 1,665 | 100 % | 29 | 29 | +13.9 % | +20.3 % | +13.9 % | – | +13.0 % | 8.36 % |
| + 20 layers in host RAM | 2 | 1,676 | 1,678 | 100 % | 30 | 30 | +12.1 % | +20.0 % | +12.1 % | – | +12.3 % | 7.24 % |
| + 12 layers in host RAM | 2 | 1,812 | 1,814 | 100 % | 30 | 30 | +16.2 % | +21.8 % | +16.1 % | – | +15.7 % | 8.93 % |
| 8 mb + activations in host RAM | 2 | 2,094 | 2,097 | 100 % | 32 | 32 | +9.5 % | +17.2 % | +9.6 % | – | +8.9 % | 4.55 % |
| 8 micro-batches | 2 | 3,325 | 4,444 | 75 % | 66 | 49 | +7.1 % | +10.3 % | +7.9 % | +1.2 % | +7.0 % | 2.13 % |
| 4 micro-batches | 2 | 3,701 | 5,956 | 62 % | 80 | 50 | +6.8 % | +8.0 % | +8.1 % | +1.7 % | +6.4 % | 1.84 % |

## Both passes averaged

| shape | MB per token (R2) | tenant tok/s | share computing | GB/s while computing | decode gap mean | while tenant computes | per 1 k tok/s | per 100 GB/s on average |
|---|---|---|---|---|---|---|---|---|
| 4 micro-batches | 13.5 | 3,701 | 62 % | 80 | +6.0 % | +7.6 % | 1.61 % | 12.0 % |
| 8 micro-batches | 14.9 | 3,321 | 75 % | 66 | +7.5 % | +8.2 % | 2.26 % | 15.2 % |
| 8 mb + activations in host RAM | 15.2 | 2,086 | 100 % | 32 | +10.0 % | +10.1 % | 4.81 % | 31.6 % |
| + 12 layers in host RAM | 16.6 | 1,818 | 100 % | 30 | +14.0 % | +13.9 % | 7.68 % | 46.3 % |
| + 20 layers in host RAM | 17.6 | 1,670 | 100 % | 29 | +13.0 % | +13.0 % | 7.80 % | 44.3 % |
