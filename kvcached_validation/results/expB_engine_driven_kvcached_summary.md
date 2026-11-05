## expB_engine_driven_kvcached

requests: 91 total, 91 ok, 0 failed/errored; output tokens: 11648; SSE chunks: 11514 (ITL below is per token = chunk gap / tokens in chunk)
TTFT s: p50 2.648  p90 5.624  p99 8.589
E2E  s: p50 6.313  p90 9.051  max 10.580
ITL  s: p50 0.0129  p90 0.0148  p99 0.1570  max 4.285
token-stream stalls > 1 s: 74 in 42 requests; longest 4.29 s at t=134.0 s
vLLM preemptions counter: 0 → 0 (Δ 0)
resize t=41.7s → 2G (kvctl rc=0, kvctl took 2.09s, limit visible 1.87s after invocation): NVML used 19.60 → 20.35 GiB; first movement after 0.00s, within 2% of plateau after 1.10s; kvcached used 0.94 → 1.75 GiB (fits under the limit after 0.00s); running reqs at cmd: 1.0
    ITL median 15 s before: 0.0131 s; 2–17 s after: 0.0118 s; during first 2 s after: p50 1.0979 max 1.098
resize t=91.6s → 12G (kvctl rc=0, kvctl took 2.11s, limit visible 1.78s after invocation): NVML used 20.35 → 21.60 GiB; first movement after 1.30s, within 2% of plateau after 30.70s; kvcached used 1.88 → 2.94 GiB (fits under the limit after 0.00s); running reqs at cmd: 1.0
    ITL median 15 s before: 0.0120 s; 2–17 s after: 0.0139 s; during first 2 s after: p50 0.0033 max 0.007
