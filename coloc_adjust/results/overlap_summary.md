# Decode inter-token time by tenant state

Mean decode inter-token time against the no-tenant run; `n` = token gaps; a cell needs at least 300 gaps and 5 % of the gaps of its row.

## A. Against the same phases (total effect), 0.5 and 1 req/s

| scheme | req/s | tenant running: share of time | all gaps | while tenant runs | while tenant does not run | n run / not |
|---|---|---|---|---|---|---|
| adjust_nogate | 0.5 | 99 % | +26.0 % | +24.9 % | – | 3864 / 0 |
| adjust_nogate | 1 | 98 % | +26.6 % | +25.8 % | – | 27360 / 0 |
| adjust_mps20 | 0.5 | 99 % | +9.5 % | +9.4 % | – | 3868 / 0 |
| adjust_mps20 | 1 | 99 % | +10.9 % | +10.5 % | – | 27555 / 0 |
| green_split | 0.5 | 95 % | +25.0 % | +25.2 % | – | 3735 / 0 |
| green_split | 1 | 98 % | +26.2 % | +25.2 % | – | 27296 / 0 |
| green_dyn2 | 0.5 | 98 % | +13.1 % | +12.5 % | – | 3818 / 0 |
| green_dyn2 | 1 | 98 % | +14.1 % | +13.5 % | – | 27216 / 0 |
| tpot_slo15 | 0.5 | 36 % | +10.4 % | +30.0 % | -1.2 % | 1398 / 2492 |
| tpot_slo15 | 1 | 33 % | +9.6 % | +23.0 % | +2.0 % | 8961 / 18631 |
| tpot_slo5 | 0.5 | 40 % | +11.2 % | +28.6 % | -0.7 % | 1546 / 2331 |
| tpot_slo5 | 1 | 9 % | +2.8 % | +17.2 % | +1.2 % | 2543 / 25284 |

## B. Against the same number of running requests (per-step effect)

| scheme | req/s | tenant running: share of time | all gaps | while tenant runs | while tenant does not run | n run / not |
|---|---|---|---|---|---|---|
| adjust_nogate | 0.5 | 99 % | +16.3 % | +15.3 % | – | 3864 / 0 |
| adjust_nogate | 1 | 98 % | +17.2 % | +16.6 % | – | 27360 / 0 |
| adjust_nogate | 2 | 56 % | +7.9 % | +14.4 % | -1.0 % | 16986 / 13435 |
| adjust_mps20 | 0.5 | 99 % | +2.5 % | +2.5 % | – | 3868 / 0 |
| adjust_mps20 | 1 | 99 % | +4.8 % | +4.5 % | – | 27555 / 0 |
| adjust_mps20 | 2 | 57 % | +4.6 % | +8.3 % | -1.1 % | 17389 / 13095 |
| green_split | 0.5 | 95 % | +15.2 % | +15.1 % | – | 3735 / 0 |
| green_split | 1 | 98 % | +16.8 % | +16.2 % | – | 27296 / 0 |
| green_split | 2 | 48 % | +7.1 % | +14.8 % | -0.6 % | 14456 / 15996 |
| green_dyn2 | 0.5 | 98 % | +5.5 % | +5.1 % | – | 3818 / 0 |
| green_dyn2 | 1 | 98 % | +7.3 % | +6.8 % | – | 27216 / 0 |
| green_dyn2 | 2 | 56 % | +4.6 % | +9.2 % | -1.8 % | 17127 / 13342 |
| tpot_slo15 | 0.5 | 36 % | +2.7 % | +16.6 % | -5.6 % | 1398 / 2492 |
| tpot_slo15 | 1 | 33 % | +3.7 % | +15.8 % | -3.1 % | 8961 / 18631 |
| tpot_slo15 | 2 | 26 % | +3.6 % | +15.0 % | -0.5 % | 7816 / 22557 |
| tpot_slo5 | 0.5 | 40 % | +3.3 % | +14.7 % | -4.6 % | 1546 / 2331 |
| tpot_slo5 | 1 | 9 % | -1.7 % | +12.7 % | -3.3 % | 2543 / 25284 |
| tpot_slo5 | 2 | 10 % | +0.2 % | +9.5 % | -1.0 % | 3055 / 27601 |

## Shape probe (1 req/s, 4k tok/s cap), against its own no-tenant window

`plain` = against the window mean (total effect); `same n` = against the same number of running requests. Tenant tok/s while running = tokens / time spent computing.

| shape | tenant running: share of time | tenant tok/s while running | all gaps, plain | while tenant runs, plain | while tenant does not run, plain | while tenant runs, same n | while not, same n | n run / not |
|---|---|---|---|---|---|---|---|---|
| cfg 2 | 31 % | 11,413 | +6.3 % | +22.3 % | -1.1 % | +18.7 % | -3.6 % | 1303 / 2964 |
| cfg 3 | 61 % | 6,196 | +10.8 % | +15.6 % | +1.7 % | +15.7 % | +1.1 % | 3421 / 2169 |
| cfg 4 | 100 % | 2,772 | +16.8 % | +16.8 % | – | +14.8 % | – | 4289 / 0 |
| cfg 5 | 100 % | 2,207 | +21.6 % | +21.4 % | – | +20.5 % | – | 4645 / 0 |
| cfg 6 | 100 % | 2,016 | +19.5 % | +19.6 % | – | +17.7 % | – | 4340 / 0 |

## Per-request TPOT, median and p90 (ms; change against the no-tenant run)

| scheme | p50 @0.5 | p50 @1 | p50 @2 | p90 @0.5 | p90 @1 | p90 @2 |
|---|---|---|---|---|---|---|
| adjust_notenant | 12.0 | 12.6 | 16.3 | 13.3 | 14.1 | 18.9 |
| static_split | 21.8 (+82 %) | 16.7 (+32 %) | 21.8 (+34 %) | 23.1 (+74 %) | 19.2 (+37 %) | 22.8 (+21 %) |
| adjust | 12.1 (+1 %) | 12.8 (+1 %) | 16.5 (+1 %) | 13.3 (+0 %) | 14.3 (+1 %) | 19.1 (+1 %) |
| adjust_nogate | 15.4 (+29 %) | 16.3 (+29 %) | 18.9 (+16 %) | 17.0 (+27 %) | 18.4 (+31 %) | 21.3 (+13 %) |
| adjust_mps20 | 13.2 (+10 %) | 14.0 (+11 %) | 18.0 (+10 %) | 14.7 (+10 %) | 15.8 (+12 %) | 20.3 (+8 %) |
| green_split | 15.3 (+28 %) | 16.2 (+28 %) | 18.6 (+14 %) | 17.0 (+28 %) | 18.2 (+30 %) | 21.5 (+14 %) |
| green_dyn2 | 13.7 (+14 %) | 14.5 (+15 %) | 17.8 (+9 %) | 15.3 (+15 %) | 16.3 (+16 %) | 20.5 (+9 %) |
| tpot_slo15 | 12.5 (+5 %) | 14.0 (+11 %) | 17.3 (+6 %) | 17.1 (+29 %) | 16.3 (+16 %) | 20.2 (+7 %) |
| tpot_slo5 | 12.8 (+7 %) | 13.1 (+3 %) | 16.9 (+4 %) | 17.0 (+28 %) | 14.6 (+4 %) | 20.4 (+8 %) |
