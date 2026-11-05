## expB_engine_driven_kvcached_stress

requests: 84 total, 79 ok, 5 failed/errored; output tokens: 10112; SSE chunks: 10105 (ITL below is per token = chunk gap / tokens in chunk)
TTFT s: p50 4.492  p90 10.305  p99 12.376
E2E  s: p50 8.949  p90 13.625  max 18.740
ITL  s: p50 0.0143  p90 0.0156  p99 0.3819  max 7.697
token-stream stalls > 1 s: 42 in 30 requests; longest 7.70 s at t=29.9 s
failed examples: s4t13: 200 KeyError('choices'); s3t14: 200 KeyError('choices'); s1t12: 200 KeyError('choices')
vLLM preemptions counter: 0 → 0 (Δ 0)
resize t=31.2s → 2G (kvctl rc=0, kvctl took 1.45s, limit visible 1.39s after invocation): NVML used 23.72 → 20.09 GiB; first movement after 1.10s, within 2% of plateau after 1.40s; kvcached used 5.19 → 1.56 GiB (fits under the limit after 1.10s); running reqs at cmd: 6.0
    ITL median 15 s before: 0.0147 s; 2–17 s after: 0.0117 s; during first 2 s after: p50 0.0148 max 0.029
resize t=46.2s → 12G (kvctl rc=0, kvctl took 1.60s, limit visible 1.39s after invocation): NVML used 20.03 → 22.47 GiB; first movement after 0.90s, within 2% of plateau after 10.70s; kvcached used 1.50 → 3.94 GiB (fits under the limit after 0.00s); running reqs at cmd: 1.0
    ITL median 15 s before: 0.0119 s; 2–17 s after: 0.0146 s; during first 2 s after: p50 1.1276 max 2.255
resize t=61.2s → 2G (kvctl rc=0, kvctl took 1.60s, limit visible 1.39s after invocation): NVML used 22.47 → 20.09 GiB; first movement after 3.10s, within 2% of plateau after 11.80s; kvcached used 3.94 → 1.56 GiB (fits under the limit after 6.00s); running reqs at cmd: 1.0
    ITL median 15 s before: 0.0146 s; 2–17 s after: 0.0116 s; during first 2 s after: p50 0.0106 max 6.135
resize t=76.1s → 12G (kvctl rc=0, kvctl took 1.52s, limit visible 1.29s after invocation): NVML used 20.09 → 23.41 GiB; first movement after 0.10s, within 2% of plateau after 12.10s; kvcached used 1.56 → 4.88 GiB (fits under the limit after 0.00s); running reqs at cmd: 1.0
    ITL median 15 s before: 0.0115 s; 2–17 s after: 0.0146 s; during first 2 s after: p50 0.0117 max 0.012
resize t=91.2s → 2G (kvctl rc=0, kvctl took 1.53s, limit visible 1.39s after invocation): NVML used 23.47 → 20.22 GiB; first movement after 0.00s, within 2% of plateau after 3.30s; kvcached used 4.94 → 1.75 GiB (fits under the limit after 0.00s); running reqs at cmd: 6.0
    ITL median 15 s before: 0.0146 s; 2–17 s after: 0.0117 s; during first 2 s after: p50 0.1202 max 0.120
resize t=106.1s → 12G (kvctl rc=0, kvctl took 1.50s, limit visible 1.29s after invocation): NVML used 20.22 → 22.47 GiB; first movement after 1.20s, within 2% of plateau after 2.60s; kvcached used 1.75 → 3.94 GiB (fits under the limit after 0.00s); running reqs at cmd: 2.0
    ITL median 15 s before: 0.0117 s; 2–17 s after: 0.0144 s; during first 2 s after: p50 0.7127 max 1.425
