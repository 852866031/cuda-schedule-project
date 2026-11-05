## expB_engine_driven_plain

requests: 97 total, 97 ok, 0 failed/errored; output tokens: 12416; SSE chunks: 12410 (ITL below is per token = chunk gap / tokens in chunk)
TTFT s: p50 2.494  p90 5.499  p99 7.206
E2E  s: p50 5.491  p90 8.701  max 10.391
ITL  s: p50 0.0131  p90 0.0149  p99 0.1334  max 4.456
token-stream stalls > 1 s: 55 in 32 requests; longest 4.46 s at t=150.8 s
vLLM preemptions counter: 0 → 0 (Δ 0)
resize t=41.2s → 2G (kvctl rc=0, kvctl took 1.38s, limit visible 1.38s after invocation): NVML used 30.15 → 30.15 GiB; first movement after nans, within 2% of plateau after nans; kvcached used nan → nan GiB (fits under the limit after nans); running reqs at cmd: 4.0
    ITL median 15 s before: 0.0123 s; 2–17 s after: 0.0121 s; during first 2 s after: p50 nan max nan
resize t=91.5s → 12G (kvctl rc=0, kvctl took 1.67s, limit visible 1.67s after invocation): NVML used 30.15 → 30.15 GiB; first movement after nans, within 2% of plateau after nans; kvcached used nan → nan GiB (fits under the limit after nans); running reqs at cmd: 4.0
    ITL median 15 s before: 0.0137 s; 2–17 s after: 0.0141 s; during first 2 s after: p50 0.0138 max 1.193
