## expOH_kvcached

requests: 100 total, 100 ok, 0 failed/errored; output tokens: 25600; SSE chunks: 25600 (ITL below is per token = chunk gap / tokens in chunk)
TTFT s: p50 0.488  p90 1.138  p99 3.178
E2E  s: p50 6.017  p90 8.197  max 9.935
ITL  s: p50 0.0140  p90 0.0163  p99 0.4576  max 0.619
token-stream stalls > 1 s: 0 in 0 requests
vLLM preemptions counter: 16 → 16 (Δ 0)
