## expA1

requests: 480 total, 480 ok, 0 failed/errored; output tokens: 45406
TTFT s: p50 82.784  p90 145.113  p99 153.725
E2E  s: p50 91.600  p90 153.943  max 161.640
ITL  s: p50 0.0196  p90 0.0392  p99 0.9531  max 1.430
vLLM preemptions counter: 0 → 0 (Δ 0)
resize t=41.1s → 3G (kvctl rc=0, kvctl took 1.37s, limit visible 1.25s after invocation): NVML used 29.66 → 20.66 GiB; first movement after 0.50s, within 2% of plateau after 2.70s; kvcached used 11.94 → 3.00 GiB; running reqs at cmd: 15.0
    ITL median 15 s before: 0.0196 s; 2–17 s after: 0.0128 s; during first 2 s after: p50 0.0158 max 0.942
resize t=91.1s → 12G (kvctl rc=0, kvctl took 1.39s, limit visible 1.25s after invocation): NVML used 20.66 → 29.66 GiB; first movement after 0.50s, within 2% of plateau after 6.20s; kvcached used 3.00 → 11.88 GiB; running reqs at cmd: 4.0
    ITL median 15 s before: 0.0133 s; 2–17 s after: 0.0356 s; during first 2 s after: p50 0.4656 max 1.078
