## expA2

requests: 160 total, 160 ok, 0 failed/errored; output tokens: 81920; SSE chunks: 81903 (ITL below is per token = chunk gap / tokens in chunk)
TTFT s: p50 29.055  p90 48.119  p99 53.335
E2E  s: p50 44.815  p90 62.495  max 66.443
ITL  s: p50 0.0183  p90 0.0199  p99 0.4754  max 7.008
token-stream stalls > 1 s: 13 in 13 requests; longest 7.01 s at t=158.9 s
vLLM preemptions counter: 0 → 15 (Δ 15)
resize t=41.1s → 3G (kvctl rc=0, kvctl took 1.42s, limit visible 1.28s after invocation): NVML used 29.47 → 20.66 GiB; first movement after 0.50s, within 2% of plateau after 24.20s; kvcached used 11.50 → 2.38 GiB (fits under the limit after 21.00s); running reqs at cmd: 15.0
    ITL median 15 s before: 0.0186 s; 2–17 s after: 0.0143 s; during first 2 s after: p50 0.0173 max 0.834
resize t=91.1s → 12G (kvctl rc=0, kvctl took 1.42s, limit visible 1.28s after invocation): NVML used 20.66 → 29.66 GiB; first movement after 0.40s, within 2% of plateau after 5.40s; kvcached used 2.44 → 12.00 GiB (fits under the limit after 0.00s); running reqs at cmd: 3.0
    ITL median 15 s before: 0.0122 s; 2–17 s after: 0.0187 s; during first 2 s after: p50 0.0122 max 0.624
