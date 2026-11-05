## rehearsal

requests: 45 total, 45 ok, 0 failed/errored; output tokens: 4457
TTFT s: p50 0.645  p90 1.941  p99 3.109
E2E  s: p50 3.631  p90 5.408  max 7.011
ITL  s: p50 0.0139  p90 0.0275  p99 0.4638  max 0.906
vLLM preemptions counter: 0 → 0 (Δ 0)
resize t=24.8s → 4G (kvctl rc=0, cmd took 1.54s): NVML used 20.66 → 18.34 GiB; first movement after 0.04s, within 2% of plateau after 3.84s; kvcached used 2.38 → 0.06 GiB; running reqs at cmd: 4.0
    ITL median 15 s before: 0.0138 s; 2–17 s after: 0.0141 s; during first 2 s after: p50 0.0125 max 0.464
resize t=44.8s → 12G (kvctl rc=0, cmd took 1.45s): NVML used 19.41 → 18.34 GiB; first movement after 0.04s, within 2% of plateau after 0.04s; kvcached used 1.12 → 0.06 GiB; running reqs at cmd: 1.0
    ITL median 15 s before: 0.0141 s; 2–17 s after: 0.0107 s; during first 2 s after: p50 nan max nan
