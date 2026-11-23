# vLLM-engine runs in results_b50: every number against the engine alone on the same night, arrival rates 0.5,1 req/s

Baseline runs pooled: adjust_notenant_a. Baseline values: mean gap 12.64 ms, median gap 11.99 ms; TPOT median 12.48 ms, p90 14.36 ms; single steps with 1 / 2 / 3 requests generating: median 10.66 / 11.93 / 12.30 ms, p99 11.18 / 12.54 / 13.07 ms; all-gap p90 / p95 / p99 12.86 / 14.25 / 66.47 ms.

## Every run

| run | mean gap | TPOT p50 | TPOT p90 | step p50 | step p90 | step p99 | all-gap p90 | all-gap p95 | all-gap p99 | tenant tok/s | computing | tenant tok/s, whole run | pauses per step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cvfm_step_b50 | +7.8 % | +7.4 % | +8.7 % | +6.2 % | +6.9 % | +7.4 % | +11.6 % | +8.9 % | +1.7 % | 9,013 | 94 % | 5,853 | 0 |
| cvfm_w12_b50 | +7.1 % | +6.8 % | +8.1 % | +5.8 % | +6.2 % | +6.3 % | +10.2 % | +7.5 % | +1.3 % | 8,350 | 89 % | 5,455 | 12 |
| adjust_notenant_a | +0.0 % | +0.0 % | +0.0 % | +0.0 % | +0.0 % | +0.0 % | +0.0 % | +0.0 % | +0.0 % | | | | |

## Repeats: mean (lowest to highest)

| run | n | mean gap | TPOT p50 | TPOT p90 | step p50 | step p90 | step p99 | all-gap p90 | all-gap p95 | all-gap p99 | tenant tok/s | computing | tenant tok/s, whole run | pauses per step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cvfm_step_b50 | 1 | +7.8 % | +7.4 % | +8.7 % | +6.2 % | +6.9 % | +7.4 % | +11.6 % | +8.9 % | +1.7 % | 9,013 | 94 % | 5,853 | 0 |
| cvfm_w12_b50 | 1 | +7.1 % | +6.8 % | +8.1 % | +5.8 % | +6.2 % | +6.3 % | +10.2 % | +7.5 % | +1.3 % | 8,350 | 89 % | 5,455 | 12 |
