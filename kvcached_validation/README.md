# kvcached live KV-budget resizing — validation

Self-contained validation (own venv, own scripts, nothing reused from other branches) of two
questions the research proposal depends on:

1. Can the kvcached plugin (ovg-project) shrink and grow a *running* vLLM engine's KV-cache
   budget live, and what does that cost (latency to release/re-acquire memory, effect on
   running requests, per-step overhead)?
2. Does kvcached coexist with LMCache's multi-process connector doing asynchronous KV
   offload/load to host RAM at the same time?

Results: [`REPORT.md`](REPORT.md). Run-by-run predictions and outcomes: [`DECISIONS.md`](DECISIONS.md).

## Environment

`scripts/setup_env.sh` builds `.venv` (Python 3.12) with `vllm==0.30.0`, `kvcached==0.1.6`,
`lmcache==0.5.6`, `torch==2.13.0+cu130`. Exact versions and the reasoning are in `DECISIONS.md` D1.
Hardware: 2× RTX 5090 32 GB, driver 580, CUDA 12.8 toolkit on the host (the venv brings its own
CUDA 13 runtime). Model: `meta-llama/Meta-Llama-3-8B`, bf16.

## Scripts

| script | purpose |
|---|---|
| `scripts/env.sh` | shared environment (activates the venv, sets `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=8`, ports, IPC name) |
| `scripts/serve.sh <tag> <plain\|kvcached> [--lmcache MODE] [--apc] [--kv-bytes N]` | launch one vLLM engine; logs to `logs/vllm_<tag>.log` |
| `scripts/lmcache_server.sh <tag> [--l1-gb N] [--transfer MODE]` | launch the LMCache MP server |
| `scripts/wait_ready.sh <tag>` / `scripts/stop.sh` | wait for `/health`; stop everything |
| `scripts/sampler.py` | 100 ms samples of NVML memory, kvcached's segment (limit/used), vLLM gauges → CSV |
| `scripts/resize.py` | timed `kvctl limit` commands, logged with wall-clock instants |
| `scripts/loadgen.py` | streaming load generator (open-loop stream, multi-turn sessions, output check) |
| `scripts/run_resize_exp.sh` | one shrink/grow experiment = sampler + resize + load + analysis |
| `scripts/analyze.py` | joins the traces, prints the summary, draws `results/<name>.png` |

## Reproduce

```bash
scripts/setup_env.sh
scripts/serve.sh a kvcached && scripts/wait_ready.sh a
scripts/run_resize_exp.sh expA1 stream 3.5 150 40:3G 90:12G
scripts/stop.sh
```
