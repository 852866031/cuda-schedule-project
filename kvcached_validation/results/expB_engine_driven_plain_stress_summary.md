## expB_engine_driven_plain_stress

requests: 94 total, 94 ok, 0 failed/errored; output tokens: 12032; SSE chunks: 12032 (ITL below is per token = chunk gap / tokens in chunk)
TTFT s: p50 2.853  p90 6.639  p99 8.830
E2E  s: p50 8.226  p90 10.803  max 11.315
ITL  s: p50 0.0145  p90 0.0157  p99 0.5128  max 7.140
token-stream stalls > 1 s: 88 in 53 requests; longest 7.14 s at t=27.5 s
vLLM preemptions counter: 0 → 0 (Δ 0)
