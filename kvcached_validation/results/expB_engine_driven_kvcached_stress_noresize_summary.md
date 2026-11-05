## expB_engine_driven_kvcached_stress_noresize

requests: 184 total, 52 ok, 132 failed/errored; output tokens: 6656; SSE chunks: 6652 (ITL below is per token = chunk gap / tokens in chunk)
TTFT s: p50 3.064  p90 8.631  p99 9.012
E2E  s: p50 8.469  p90 10.695  max 11.132
ITL  s: p50 0.0145  p90 0.0158  p99 0.4512  max 7.200
token-stream stalls > 1 s: 34 in 21 requests; longest 7.20 s at t=27.0 s
failed examples: s5t9: 200 KeyError('choices'); s2t9: 200 KeyError('choices'); s3t10: 200 KeyError('choices')
vLLM preemptions counter: 0 → 0 (Δ 0)
