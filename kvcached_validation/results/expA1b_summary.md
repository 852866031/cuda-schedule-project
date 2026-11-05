## expA1b

requests: 160 total, 160 ok, 0 failed/errored; output tokens: 14941; SSE chunks: 14941 (ITL below is per token = chunk gap / tokens in chunk)
TTFT s: p50 0.841  p90 3.338  p99 6.383
E2E  s: p50 4.400  p90 10.100  max 14.044
ITL  s: p50 0.0181  p90 0.0371  p99 0.4944  max 1.264
vLLM preemptions counter: 0 → 0 (Δ 0)
resize t=41.1s → 2G (kvctl rc=0, kvctl took 1.42s, limit visible 1.29s after invocation): NVML used 18.72 → 19.47 GiB; first movement after 0.10s, within 2% of plateau after 0.10s; kvcached used 0.75 → 1.50 GiB; running reqs at cmd: 0.0
    ITL median 15 s before: 0.0184 s; 2–17 s after: 0.0118 s; during first 2 s after: p50 0.0115 max 0.453
resize t=91.1s → 12G (kvctl rc=0, kvctl took 1.39s, limit visible 1.29s after invocation): NVML used 19.47 → 20.72 GiB; first movement after 0.40s, within 2% of plateau after 26.10s; kvcached used 1.50 → 2.44 GiB; running reqs at cmd: 2.0
    ITL median 15 s before: 0.0116 s; 2–17 s after: 0.0190 s; during first 2 s after: p50 0.0115 max 0.623
