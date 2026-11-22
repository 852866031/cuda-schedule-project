# Decode steps next to each tenant, rates 0.5,1 req/s

Baseline: adjust_notenant (pooled). Step counts of the baseline by requests generating: 1: 7542, 2: 9415, 3: 5852.

| run | requests | mean token gap | ordinary steps p10 | p50 | p90 | p99 | tenant tok/s | computing | pauses per step | late per step (max ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| c1_fix | 1185 | +6.5 % | +2.3 % | +4.9 % | +7.2 % | +19.4 % | 3,377 | 59 % | 0 | 0.0 (0) |
| v_step | 1185 | +5.9 % | +2.4 % | +5.1 % | +5.6 % | +3.8 % | 3,277 | 67 % | 0 | 0.0 (84) |
| v_w12 | 1185 | +6.0 % | +4.3 % | +4.8 % | +5.1 % | +1.6 % | 3,240 | 69 % | 56 | 15.7 (124) |

## By number of requests generating (1 / 2 / 3)

| run | p10 | p50 | p90 | p99 |
|---|---|---|---|---|
| c1_fix | +2.2 / +2.0 / +2.5 | +3.7 / +5.7 / +5.3 | +5.5 / +8.3 / +7.7 | +23.7 / +20.9 / +13.5 |
| v_step | +2.5 / +2.3 / +2.4 | +4.0 / +5.9 / +5.4 | +4.4 / +6.4 / +5.8 | +3.1 / +5.7 / +2.5 |
| v_w12 | +3.7 / +4.7 / +4.6 | +4.3 / +5.1 / +4.9 | +4.5 / +5.5 / +5.2 | +2.8 / +3.5 / -1.4 |
