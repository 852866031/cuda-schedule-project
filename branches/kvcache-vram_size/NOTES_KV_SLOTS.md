# KV slots — starting facts

Everything below comes from data that already exists: `output/raw/pfp_*.json` and the
decode-engine logs in `output/logs/pfp_archive/<cell>/pfp_big_decode.log` (ninth study,
Llama-3-8B split with a Qwen2.5-3B prefill neighbour on GPU1). Items marked
**hypothesis** are not yet measured; this study tests them with no neighbour on GPU1.

## Workload (fixed)

32 sessions × 6144-token fixed prefix + 128 unique suffix tokens, 128 output tokens,
zipf-1.1 session popularity, open-loop Poisson arrivals, seed 0, 300 requests. Router
`--forward-first-token --max-inflight 999`.

## Arithmetic

| quantity | value |
|---|---|
| Llama-3-8B KV per token | 32 layers × 2 (K,V) × 8 KV heads × 128 dim × 2 B = **128 KiB** |
| one request (6400 tokens) | 400 blocks of 16 tokens = **0.78 GiB** |
| KV grant vs decode budget | 23 GiB → 6.77 GiB (55,488 tokens); 25 GiB → 8.77 GiB (71,872 tokens); grant ≈ budget − 16.2 GiB |
| whole requests that fit | 23 GiB → ⌊55,488 / 6400⌋ = **8**; 25 GiB → ⌊71,872 / 6400⌋ = **11** |
| extrapolated (to be probed) | 24 / 26 / 28 GiB → ≈ 7.77 / 9.77 / 11.77 GiB → ≈ **9 / 12 / 15** requests |

Correction to the ninth study's report: its "8.7 slots" (grant ÷ 0.78 GiB) ignores that
the scheduler admits whole requests only. The usable count is the floor: 8 at 23 GiB.

## What the ninth study already shows (23 vs 25 GiB, fp8 3B neighbour, 8 req/s of 3B)

| decode-side stage | 23 GiB | 25 GiB |
|---|---|---|
| KV load (scheduled → first token) | 397 ms | 413 ms |
| time in decode (first → last token) | 4.14 s | 4.89 s |
| queue before decode (arrival → scheduled) | **5.6 s** | **1.3 s** |

The difference is all queueing. The engine's 10 s stats lines agree: at 23 GiB Running
never exceeds 8 (KV usage 91–92%, ~11.4% per request) and Waiting climbs to 27; at
25 GiB Running reaches 11 (KV 97%) and Waiting drains. Rough queueing check at 2 QPS:
2 × 4.14 s = 8.3 busy slots > 8 (unstable, linear queue growth); 2 × 4.89 s = 9.8 < 11
(ρ ≈ 0.89, stable).

## Metric definitions (vLLM 0.15.1, `vllm/v1/metrics/stats.py`)

- queued time = scheduled − queued
- prefill time = first token − scheduled (on the decode engine this is the KV load)
- decode time = last token − first token

## How KV reaches the decode GPU today

Read from source (`lmcache/integration/vllm/vllm_v1_adapter.py:741`, `start_load_kv`):

- When a request is scheduled, the decode engine allocates all 400 of its blocks at
  once; a block id indexes the same slot in all 32 per-layer KV tensors.
- LMCache runs with `use_layerwise=False` and `enable_async_loading=False` (defaults):
  all 32 layers are copied into the paged blocks **synchronously, before the forward
  pass** of the step that admits the request.
- The decode engine runs with `--no-enable-prefix-caching`. No reason is recorded; the
  split report says only that decode VRAM is a working area, not a cache.
- Measured load: 0.78 GiB in ~400 ms ≈ 2 GB/s, far below PCIe. LMCache's compiled ops
  do not load against torch 2.9.1, so it runs the Python fallback over a TCP store.

## Two suspected sources of waste (hypotheses)

1. **No prefix sharing.** With prefix caching off, concurrent requests from the same
   session each hold their own 6144-token copy. Simulating the arrival sequence: on
   average 7.2 concurrent requests span only 5.1 distinct sessions; at ≥ 9 concurrent,
   10.3 requests span 6.8 sessions; the hottest session issues 32% of all requests.
2. **Synchronous load stalls the whole batch.** 8B alone: ITL p50 14.8 ms but mean
   28.4 ms (fp8b25 p95 155 ms). Guess: each admission stalls every running request for
   ~200 ms. If removed, time in decode ~4.1 s → ~2.3 s and the slots needed at 2 QPS
   fall from 8.3 to ~4.7.

## Why decode cannot stream KV layer by layer

Every decode step attends, in every layer, over the request's full history, so it needs
all layers' KV every step. Streaming instead of holding it would move 0.78 GiB per
request per step: ≈ 60 ms at ~13 GB/s (to be measured with
`scripts/common/calibrate_pcie.py`; an earlier calibration in
`output/calibration_pcie.json` read ~14 GB/s pinned H2D), against a ~15 ms step.
Layer-wise pipelining (LMCache `use_layerwise`, CachedAttention) can only speed up the
one-time *load*; it does not reduce resident VRAM.
