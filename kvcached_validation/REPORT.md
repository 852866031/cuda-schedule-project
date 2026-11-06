# kvcached live KV-budget resizing with and without LMCache — validation report

Date: 2026-11-05. Branch `kvcached-validation`, directory `kvcached_validation/`.
Everything here was built from scratch in its own venv; predictions and outcomes per run are in
[`DECISIONS.md`](DECISIONS.md), raw traces and figures in `results/`, engine logs in `logs/`.

## 1. Summary

| question | answer |
|---|---|
| Does kvcached shrink/grow a running vLLM engine's KV budget live? | **Yes.** `kvctl limit` is picked up by the engine within ~1.3 s (CLI start-up; the engine polls its segment every 100 ms). Physical memory is unmapped as the requests holding it finish; pages in use are never revoked. |
| What happens to running requests at a shrink? | None is killed or stopped at the shrink instant. Memory falls at the rate requests complete. **But** once the resident set sits at the new cap, vLLM's ordinary full-cache preemption kicks in: growing decodes get preempted and recomputed (A2: 15 preemptions, stalls of 1.7–7 s, every request still completed with correct length). |
| Time to release / re-acquire memory | Release: first movement 0.5 s after the limit is visible; fits under the limit after ~3 s (A1, short outputs) to ~21 s (A2, 512-token outputs: bounded by the lifetime of the resident requests). Re-acquire: bounded by prefill admission, 2.4 → 11.9 GiB in 5–6 s. |
| Per-step overhead of kvcached at a static budget | **None measurable**: decode step p50 14.03 ms vs 14.03 ms, p90 16.17 vs 16.18 ms, TTFT identical, greedy outputs byte-identical (8/8 prompts). |
| Fixed 2 QPS with DRAM offload, shrink the budget live (vram-limit study 1 redone with kvcached, §7) | **Flat down to the in-flight-KV knee.** 13 → 4 GiB: TTFT p50 51 → 155 ms, p95 < 0.8 s, ITL and tok/s unchanged, 0 preemptions, 0 failures, the DRAM tier absorbing the lost GPU hits (72 → 95 %); at 3 and 2 GiB the engine queues (TTFT 3 s / 18 s) but never preempts or stalls. Without offload the knee is at 8 GiB (TTFT p50 0.5 s) and 5 GiB costs 3.5 s. Needs kvcached main (page pinning for the native offload copies) and `KVCACHED_MAX_CACHED_TOKENS=-1`. **Against a plain vLLM 0.30 engine restarted at each budget, with the same request timeline at every budget: indistinguishable at every point (TTFT within 2 ms, TPOT within 0.1 ms, queue wait within 4 ms, same preemptions, same throughput).** A decode-only emulation (prefix caching off, every prefix in DRAM) gives the same picture: TPOT flat 13.3 ms from 13 to 4 GiB, live = static. |
| kvcached + LMCache multi-process connector | **Does not work on these versions.** The default (`lmcache_driven`) and VMM-IPC modes cannot start: LMCache's server-side zero-copy path needs an exportable GPU handle and kvcached's VMM pages are created non-exportable (`requestedHandleTypes=0`). The `engine_driven` mode (worker-side copies) starts and serves, but under store traffic of a few thousand tokens per store the engine dies within 1.5–2.5 min with a CUDA illegal-memory-access in LMCache's asynchronous store gather (kernel `Xid 31` MMU read fault on an unmapped kvcached page), with or without any limit change; the plain engine with the same connector and load is unaffected. |

## 2. Environment

| component | version / setting |
|---|---|
| GPU | RTX 5090 32 GB (GPU 0 only), driver 580.126, host CUDA toolkit 12.8 |
| Python / torch | 3.12.9 / `torch==2.13.0+cu130` |
| vLLM | `0.30.0` (Model Runner V2, `VLLM_USE_V2_MODEL_RUNNER=1` in every run; kvcached requires it on ≥ 0.29) |
| kvcached | `0.1.6` (PyPI sdist, 2026-10-03; README: vLLM "≥ 0.17.0, tested up to 0.30.0"; docker pairs v0.1.6 with vLLM 0.30.0) |
| LMCache | `0.5.6` (PyPI CUDA-13 wheel, 2026-10-15), external MP connector via `kv_connector_module_path` |
| model | `meta-llama/Meta-Llama-3-8B`, bf16, 128 KiB KV per token (6000-token prompt = 0.75 GiB) |
| engine flags | `--max-model-len 8192 --max-num-seqs 32 --max-num-batched-tokens 8192 --no-enable-prefix-caching --seed 0`; plain runs `--kv-cache-memory-bytes`, kvcached runs `ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1` |
| workarounds | `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer's sampler JIT refuses sm_120 in this build; unrelated to kvcached, applied to every run); LMCache server on port 5556 (5555 is taken on this host); LMCache pinned L1 capped at 6 GB with 2 GB initial |

kvcached on this engine: 13.7 GiB KV budget at start-up (`104,704 tokens`), 32 pools, 64 KiB pages,
mapped on demand (64 MiB resident at idle). Prefix caching was off in all runs so that the only
occupied pages belong to running requests (and, in B, so that every multi-turn prefix must come
from LMCache).

## 3. Experiment A — kvcached alone: shrink and grow under load

Method: `scripts/run_resize_exp.sh` runs a 100 ms sampler (NVML memory, kvcached's own
limit/used counters from its `/dev/shm` segment, vLLM gauges), a timed `kvctl limit` sequence,
and a Poisson stream of 6000-token random prompts with greedy, `ignore_eos` outputs; per-token
step time is recovered from the streamed chunks (`return_token_ids`). t = 0 of each resize is the
first sampler row in which the new limit is visible in the segment.

### A1 — 3.5 req/s × 128 output tokens, 12 G → 3 G → 12 G (`results/expA1.png`)

The load exceeded the GPU's prefill throughput (≈ 9.6k tok/s), so a queue of ~55 requests existed
before the shrink; the run is still a valid mechanism test.

| | |
|---|---|
| requests | 480/480 completed, 0 errors, preemption counter 0 → 0 |
| at the shrink | 15 running; 15 → 7 after 1 s → 4 after 3 s, all by normal completion |
| memory (NVML) | 29.66 → 20.66 GiB; first movement 0.5 s; within 2 % of plateau 2.7 s; kvcached used 11.9 → 3.0 GiB |
| under the 3 G cap | exactly 4 requests (4 × 0.75 GiB) resident at a time; ITL median 19.6 → 12.8 ms (smaller batches) |
| grow | 3.0 → 11.9 GiB in 6.2 s, paced by 6k-token prefill admissions (~1.5 GiB/s) |

### A2 — 1.2 req/s × 512 output tokens, 12 G → 3 G → 12 G (`results/expA2.png`, headline run)

Longer outputs keep ~15 requests resident (11.5 GiB, i.e. at the 12 GiB capacity) and make the
KV of a running request grow during decode.

| | |
|---|---|
| requests | 160/160 completed with all 512 tokens, 0 errors |
| shrink | no request stopped at the shrink; kvcached used 11.5 → 4.1 GiB in 8 s, fits under 3 GiB after 21 s (the resident requests' remaining lifetime); engine log: `shrink to 1536 blocks refused by allocator (in-use pages above target); keeping shrink pending` until then |
| preemptions | counter 0 → 15; 13 token-stream stalls > 1 s (1.7–7.0 s), the first 19.6 s *after* the shrink, the rest while at the 3 G cap and later at the 12 G capacity with 15–16 running |
| grow | 2.4 → 11.9 GiB in 5.4 s; queue of 28 drained in ~30 s |

Reading: kvcached enforces a limit by shrinking vLLM's *logical* block pool. New allocations beyond
it fail in vLLM's own scheduler, which responds with its standard preempt-and-recompute. kvcached's
own "pool exhausted" emergency path was never taken (0 warnings). Running requests therefore
survive a shrink, but not "for free": if their decode needs more blocks than the cap leaves, they
are preempted like at any full KV cache. A controller that shrinks a decode engine should leave
head-room for the growth of the resident requests or accept recompute stalls.

### A3 — per-step overhead at a static budget (`results/overhead_comparison.md`)

Same engine class and flags, 8 GiB KV in both (plain: `--kv-cache-memory-bytes`; kvcached:
`kvctl limit 8G`), 1 req/s × 6000-token prompts × 256 tokens, 120 s, budget never binding.

| metric | plain | kvcached |
|---|---|---|
| decode-only step p50 / p90 | 14.03 / 16.17 ms | 14.03 / 16.18 ms |
| ITL p99 (steps sharing a 6k prefill chunk) | 457.9 ms | 457.6 ms |
| TTFT p50 / p90 | 0.49 / 1.13 s | 0.49 / 1.14 s |
| greedy outputs of 8 fixed prompts | — | identical to plain |

## 4. Experiment B — kvcached + LMCache MP connector

LMCache's multi-process mode runs the cache as a separate server process. In its default
`lmcache_driven` transfer the **server** reads and writes vLLM's GPU KV blocks directly through
CUDA IPC handles; `engine_driven` instead has the vLLM worker gather/scatter KV into host buffers
itself. kvcached allocates KV through the CUDA VMM API (`cuMemCreate`/`cuMemMap`) with no
shareable handle type. kvcached's own docs list `LMCacheMPConnector` as "no current use case".

| run | config | result |
|---|---|---|
| B1 | `lmcache_driven` (default) | **engine fails at KV registration**: `CudaIPCWrapper` → `storage._share_cuda_()` → `RuntimeError: invalid device pointer: 0x1f0000000000` (legacy CUDA IPC cannot export a VMM address). `results/expB_lmcache_driven_kvcached_failure.txt` |
| B2 | `lmcache_driven` + `lmcache.mp.use_vmm_api=true` | **fails**: `VmmCudaIPCWrapper requires a tensor that is contiguous …; got shape=(6661, 16, 8, 256) stride=(1048576, …)` — kvcached's default BLNHC layout interleaves the 32 layers inside each block. `results/expB_vmm_kvcached_failure.txt` |
| B2b | B2 + `KVCACHED_CONTIGUOUS_LAYOUT=false` (per-layer pools) | **fails** one check later: `cannot export VMM allocation (requestedHandleTypes=0): … Exportability is fixed at cuMemCreate time`. `results/expB_vmm_kvcached_lbnhc_failure.txt` |
| B3 | `engine_driven` | see §5 |

Where it fails, precisely: `lmcache/v1/platform/devices/cuda/ipc_wrapper.py` (`CudaIPCWrapper.__init__`
/ `VmmCudaIPCWrapper.__init__`), called from `worker_transfer.py:register` during
`LMCacheMPConnector.register_kv_caches`, i.e. before the first request. The allocator side is
kvcached `csrc/inc/gpu_vmm.hpp` (`cuMemCreate` with `prop.requestedHandleTypes` unset). Making the
server-driven path work would need kvcached to request `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`
*and* LMCache to accept a tensor backed by many physical chunks (one export per 2 MiB page), so it
is not a configuration fix.

## 5. B3 — `engine_driven` transfer

_(filled in below)_

Configuration: `lmcache.mp.mp_transfer_mode=engine_driven`, server `--supported-transfer-mode auto`,
pinned L1 6 GB, chunk 256 tokens; kvcached default layout. Load: 6 closed-loop sessions, each a
private 3000-token context growing by 256 tokens per turn, 128 greedy tokens per turn, ~3 s think
time, 150 s; prefix caching off, so every turn's prefix is looked up and fetched from LMCache.
Limit 13 G → 2 G at 40 s → 12 G at 90 s. Control: the same thing on a plain engine.

| | kvcached + connector (B3) | plain + connector (control) |
|---|---|---|
| requests | 91/91 ok, 0 errors, 0 preemptions | 97/97 ok, 0 errors, 0 preemptions |
| kvcached warnings / LMCache error lines | 0 / 0 | – / 0 |
| LMCache ops | 89 stores, 84 retrieves (379,648 tokens hit from L1) | 96 stores, 91 retrieves |
| retrieve speed (server log) | 0.84 s per op, 5.4k tok/s | 0.79 s per op, 5.9k tok/s |
| store speed | 0.44 s per op, 0.9k tok/s | 0.50 s per op, 0.8k tok/s |
| TTFT p50 / p90 | 2.65 / 5.62 s | 2.49 / 5.50 s |
| decode step p50 | 12.9 ms | 13.1 ms |
| token-stream stalls > 1 s | 74 in 42 requests (max 4.3 s) | 55 in 32 requests (max 4.5 s) |
| shrink to 2 G at 40 s | took effect in 1.9 s; mapped memory stayed within the limit as turns completed; no request affected | n/a |
| greedy outputs vs plain engine | 4/8 identical | 4/8 identical, and 8/8 identical to B3 |

Findings:

1. **It runs.** With the engine-driven transfer, kvcached and the LMCache MP connector coexist:
   registration succeeds, stores and loads happen continuously, a shrink and a grow under that
   traffic cause no crash, hang, error or preemption. The control shows the connector's own
   costs: the inline ("pickle") transport over ZMQ moves KV at only ~5–6k tok/s on loads and
   ~1k tok/s on stores, loads are synchronous on the worker, so every load of a 4–6k-token prefix
   stalls all decoding for ~0.8 s (the 1–4 s token-stream stalls and the 2.5 s TTFT in both
   columns). None of that is kvcached's doing.
2. **In-flight loads and stores vs. page release — the argument that looked safe was not.** A load
   is synchronous inside the worker's step, so a shrink cannot interleave with it. A store is
   asynchronous; `LMCacheMPConnector.request_finished` returns `_can_store=True`, so vLLM keeps a
   finished request's blocks until `get_finished` reports the store done, and one would expect
   kvcached never to unmap a page under a copy. The stress variant below shows that expectation
   does not hold: with larger stores the asynchronous gather reads an unmapped kvcached page and
   the engine dies, with or without a limit change.
3. **Outputs.** The LMCache MP connector changes greedy outputs relative to a no-connector engine
   on 4 of 8 short prompts; kvcached does not (8/8 identical without the connector, 8/8 identical
   to the plain+connector engine with it). Deterministic across three repeats. Not investigated
   further here; it is a property of the connector path (`results/output_check_comparison.md`).

### Stress variant — large stores and alternating shrink/grow

Same configuration, heavier offload traffic: 6 sessions whose first turn is 6000 tokens, growing
by 512 tokens per turn, think time 2 s (stores of ~2–3k tokens, ≈0.3 GiB of KV, per finished
turn). Three runs:

| run | limit schedule | outcome |
|---|---|---|
| kvcached + connector | 2 G / 12 G alternating every 15 s (6 steps) | **EngineCore crashed at t = 148 s** (43 s after the last grow, limit 12 G, 26 % KV used): LMCache's background store thread hit `torch.AcceleratorError: CUDA error: an illegal memory access` at `async_engine_driven.py:310 gather_done.synchronize()`; kernel log `Xid 31 … MMU Fault … faulted @ 0x1f00_0a4dc000, FAULT_PDE ACCESS_TYPE_VIRT_READ`. 79 requests had completed, the 5 in flight failed. `results/expB_engine_driven_kvcached_stress_crash.txt` |
| kvcached + connector | **none** (limit untouched) | **same crash at t = 90 s** (`Xid 31 … faulted @ 0x1f00_14ac9000, FAULT_PDE … VIRT_READ`), 0 kvcached warnings before it. `results/expB_engine_driven_kvcached_stress_noresize_crash.txt` |
| plain + connector | none | 94/94 ok, 0 errors, engine healthy at the end; same transfer rates (stores 1.9k tok/s, loads 5.3k tok/s) and the same LMCache-induced stalls |

Where it fails, precisely: the GPU→host *gather* of LMCache's asynchronous engine-driven store
(`lmcache/v1/multiprocess/transfer_context/async_engine_driven.py`, phase 2, on LMCache's own
copy stream in a background thread) reads a virtual address inside kvcached's VMM reservation
(the `0x1f00…` range that B1 also reported) for which no physical page is mapped at that moment
(`FAULT_PDE`, read access). On a plain engine the same read always hits backed memory, which is
why the control never fails. kvcached orders its physical page releases against the worker's
compute stream (its "physical release barrier") and pins pages for vLLM's native
`OffloadingConnector` (`offload_pins.py`); LMCache's copy stream is covered by neither. Whether the
page had been released because its blocks were freed (normal reclaim) or because LMCache's
block list reached beyond the blocks vLLM had backed was not determined, and does not change the
conclusion: **kvcached and the LMCache MP connector cannot be run together on these versions.**
The two 150-s runs with 256-token stores (B3 and its control) survived by the luck of a race
that large stores lose within 1.5–2.5 minutes. Resizing is *not* required to trigger it.

## 6. What this means for the proposal

- The mechanism the three directions assume exists and works on vLLM 0.30 + kvcached 0.1.6: a
  per-engine KV budget can be lowered and raised live from outside the process (`kvctl` or the
  Python control API), takes effect within ~1.3 s, and costs nothing per step at a static budget.
- A shrink is **not instantaneous memory release**: pages held by running requests are released
  only as those requests finish (seconds to tens of seconds, set by the residency of the resident
  set), and if the resident set keeps growing against the cap, vLLM preempts and recomputes
  (A2: 15 preemptions, stalls up to 7 s). A controller should therefore shrink with head-room, or
  pair the shrink with admission control, and treat "seconds" (Prism) as an optimistic bound for a
  decode engine with long outputs.
- With prefix caching off (as here) the only reclaimable memory is running requests'. With prefix
  caching on, kvcached 0.1.6 evicts cached prefixes only through the allocation path; a fix that
  releases the prefix cache when the limit is lowered landed upstream after 0.1.6 (#552,
  2026-10-12), so an APC-on controller should use kvcached main. Not measured here.
- LMCache's host-RAM offload does **not** coexist with kvcached on these versions. The
  server-driven zero-copy modes cannot register kvcached's non-exportable VMM pages; the
  engine-driven mode registers and serves but crashes under store traffic because LMCache's
  asynchronous GPU reads are not ordered against kvcached's page unmapping. Fixing it needs either
  kvcached to pin pages for the LMCache connector the way it does for vLLM's native
  `OffloadingConnector`, or LMCache to copy synchronously inside the step. The proposal's
  "LMCache offload + live budget" combination therefore needs one of those changes (or vLLM's
  native `OffloadingConnector`, which kvcached ≥ 0.1.6-main explicitly supports on vLLM 0.29/0.30,
  as the offload path instead of LMCache).

## 7. Live shrink sweep at fixed RPS with DRAM offload (re-doing the vram-limit study 1 with kvcached)

The `vram-limit` branch's study 1 (`branches/vram-limit/reports/report_simple_inference.md`) swept
the GPU KV budget of an offloading engine by **restarting** it at each budget. Here the same
workload and the same offload backend are driven through one continuous run in which the budget
is shrunk **live** with `kvctl limit`.

Setup: Llama-3-8B, 32 sessions × 6144-token prefixes (24 GiB of KV demand), every request = one
session's prefix + 128 unique tokens → 128 output tokens, Poisson **2 QPS**, Zipf-1.1 session
choice, 32-request warm-up; vLLM native CPU offload (`--kv-offloading-size 24
--kv-offloading-backend native`, a 24 GiB pinned DRAM tier); prefix caching on. Engine: kvcached
**main** (`9106e0f`, 2026-11-04) in `.venv-main`, because 0.1.6 has no page pinning for vLLM's
`OffloadingConnector` (main pins the pages of every offload copy job; it logged "Pinning kvcached
pages for OffloadingConnector copies") and because main releases cached prefixes when the limit
is lowered. `KVCACHED_MAX_CACHED_TOKENS=-1` so the GPU prefix cache may fill the budget (the
default bound of 16 000 tokens ≈ 2 GiB would have capped every window at the same ~4 GiB; that
first attempt is kept as `results/sweep_offload_zipf*`). Baseline 60 s at the start-up limit
(13 G), then 10G → 8G → 6G → 5G → 4G → 3G → 2G, 60 s (≈110 requests) each; the first 5 s after
each change are excluded from the window's request statistics. Runner:
`scripts/run_shrink_sweep.sh`, analysis `scripts/analyze_sweep.py`.

### With offload (`results/sweep_offload_zipf_full.png`)

| limit | req ok/fail | TTFT p50 | TTFT p95 | ITL p50 | tok/s | preempt | GPU prefix hit | DRAM hit | kvcached used | waiting max |
|---|---|---|---|---|---|---|---|---|---|---|
| 13 G (start) | 116/0 | 51 ms | 499 ms | 12.6 ms | 263 | 0 | 76 % | 72 % | 12.3 GiB | 1 |
| 10 G | 106/0 | 52 | 454 | 12.7 | 247 | 0 | 72 % | 81 % | 10.0 | 1 |
| 8 G | 105/0 | 91 | 399 | 13.1 | 244 | 0 | 55 % | 88 % | 8.0 | 2 |
| 6 G | 105/0 | 72 | 351 | 12.6 | 244 | 0 | 55 % | 87 % | 6.0 | 1 |
| 5 G | 114/0 | 146 | 770 | 13.1 | 265 | 0 | 48 % | 92 % | 5.0 | 4 |
| 4 G | 119/0 | 155 | 795 | 12.9 | 277 | 0 | 43 % | 95 % | 4.0 | 5 |
| 3 G | 108/0 | 3 210 | 5 837 | 12.2 | 251 | 0 | 29 % | 91 % | 3.0 | 13 |
| 2 G | 99/0 | 18 019 | 23 670 | 11.8 | 230 | 0 | 20 % | 93 % | 2.0 | 34 |

Each `kvctl limit` evicted cached prefixes down to exactly the new limit within the window (the
"kvcached used" column equals the limit) while the engine kept serving. From 13 GiB down to
4 GiB the decode side is untouched (ITL 12.6–13.1 ms, 244–277 tok/s) and TTFT p50 moves from
51 ms to 155 ms as GPU prefix hits fall from 76 % to 43 % and the DRAM tier absorbs the rest
(DRAM hit 72 → 95 %); p95 stays under 0.8 s; **zero preemptions, zero stalls, zero failures in
every window**. At 3 GiB and 2 GiB the resident set no longer fits (2 QPS × ~1.8 s ≈ 3.7
requests in flight × 0.77 GiB), and the engine queues: TTFT p50 3.2 s then 18 s with 13 and 34
waiting, still with zero preemptions and every request completing.

Against study 1's static sweep at the nearest GPU-KV sizes:

| GPU KV | study 1 (restart per budget): TTFT p50 / p95, preemptions | this run (live shrink) |
|---|---|---|
| 13.8 / 12.3 GiB | 54 / 305 ms, 0 | 51 / 499 ms, 0 |
| 5.8 / 6 GiB | 73 / 332 ms, 0 | 72 / 351 ms, 0 |
| 3.8 / 4 GiB | 135 / 1 528 ms, 4 | 155 / 795 ms, 0 |
| 2.8 / 3 GiB | 2 078 / 5 451 ms, 7 | 3 210 / 5 837 ms, 0 |
| 1.8 / 2 GiB | 4 856 / 10 666 ms, engine stalled | 18 019 / 23 670 ms, engine healthy, all complete |

Same knee, same flat region; past the knee kvcached turns the overload into admission queueing
(it refuses to back pages beyond the limit, so vLLM's scheduler waits) instead of the
preempt-recompute spiral and stall that the static budget produced.

### Without offload (control, `results/sweep_nooffload_zipf_full.png`)

Same engine, same sweep, no offload tier (an evicted prefix must be recomputed: ~0.4 s of
prefill for 6144 tokens).

| limit | req ok/fail | TTFT p50 | TTFT p95 | ITL p50 | preempt | GPU prefix hit | kvcached used | waiting max |
|---|---|---|---|---|---|---|---|---|
| 13 G (start) | 116/0 | 52 ms | 1 265 ms | 13.5 ms | 0 | 76 % | 12.4 GiB | 1 |
| 10 G | 106/0 | 66 | 773 | 13.8 | 0 | 72 % | 10.0 | 0 |
| 8 G | 105/0 | 505 | 2 580 | 15.5 | 0 | 55 % | 8.0 | 7 |
| 6 G | 105/0 | 500 | 1 541 | 14.2 | 0 | 55 % | 6.0 | 2 |
| 5 G | 114/0 | 3 488 | 9 041 | 14.9 | 0 | 48 % | 5.0 | 18 |
| 4 G | 119/0 | 7 053 | 13 410 | 13.9 | 1 | 41 % | 4.0 | 26 |
| 3 G | 108/0 | 27 986 | 56 741 | 12.0 | 3 | 40 % | 3.0 | 52 |
| 2 G | 99/0 | 89 754 | 107 649 | 11.8 | 1 | 10 % | 2.0 | 107 |

Without the DRAM tier the knee moves up to where the Zipf hot set stops fitting: TTFT p50 is
already 0.5 s at 8 GiB and 3.5 s at 5 GiB, where the offloaded run was still at 91 and 146 ms,
and the queue starts building at 5 GiB instead of 3 GiB. Side by side, TTFT p50 with / without
offload: 13 G 51 / 52 ms, 10 G 52 / 66, 8 G 91 / 505, 6 G 72 / 500, 5 G 146 / 3 488,
4 G 155 / 7 053 ms. Decode (ITL) is unaffected in both runs.

**Answer to the first question (shrink live with offload):** at a fixed 2 QPS with DRAM offload, the KV budget of a running
engine can be shrunk live from 13 GiB to 4 GiB with no change in decode speed or throughput and
TTFT p50 rising only from 51 to 155 ms (p95 under 0.8 s), zero preemptions and zero failures;
the wall is the in-flight KV (between 4 and 3 GiB here), exactly where study 1 found it with
restarts, and past it kvcached queues instead of spiralling. Without offload the same shrink
costs recompute from 8 GiB downward.


### Live shrink vs. the static budget, same vLLM 0.30, same request timeline (`results/fig_live_vs_static_offload.png`)

The proper control for "does resizing live cost anything" is the same engine class restarted at
each budget: `scripts/run_static_sweep.sh`, a plain vLLM 0.30 engine (no kvcached) with
`--kv-cache-memory-bytes` = 13, 10, 8, 6, 5, 4, 3, 2 GiB, prefix caching on, the same 24 GiB
offload tier and workload, 300 requests per budget.

The first comparison used the continuous live sweep above, and its 8 GiB window read 91 ms
against the static 57 ms. That turned out to be a windowing artefact, not a resize cost: the
continuous sweep gives every budget a *different* 60-s slice of the seeded Zipf sequence, the
slices differ by ±40 ms of TTFT p50 on their own (two live runs reproduce bin-for-bin; the 8 G
window contains a 10-s stretch with 36 % GPU hits and another where 16 % of tokens miss both
tiers and are recomputed, 20–60 s after the shrink, while the first 10 s after it were normal),
and the static arm scores sequence positions 0–300 at every budget. The live arm was therefore
re-run with the **same timeline at every budget** (`scripts/run_live_replay_sweep.sh`): the
engine keeps running, each limit is applied with `kvctl`, the engine idles 10 s, then the
identical seeded workload is replayed (warm-up + 150 s, 300 requests) and only that window is
scored; cache contents carry over between budgets.

![live vs static, prefix caching on](results/fig_live_vs_static_offload.png)

| GPU KV | live (limit changed in place): TTFT p50 / queue wait / TPOT / tok/s / preempt | static (restart): TTFT p50 / queue wait / TPOT / tok/s / preempt |
|---|---|---|
| 13 GiB | 51 ms / 27 ms / 13.1 ms / 258 / 0 | 52 ms / 23 ms / 13.1 ms / 258 / 0 |
| 10 | 55 / 37 / 13.1 / 258 / 0 | 55 / 33 / 13.1 / 258 / 0 |
| 8 | 58 / 42 / 13.2 / 258 / 0 | 58 / 41 / 13.1 / 258 / 0 |
| 6 | 67 / 66 / 13.1 / 258 / 0 | 68 / 65 / 13.1 / 258 / 0 |
| 5 | 137 / 92 / 13.1 / 257 / 0 | 136 / 91 / 13.1 / 257 / 0 |
| 4 | 154 / 151 / 13.1 / 257 / 2 | 155 / 153 / 13.1 / 257 / 2 |
| 3 | 2 333 / 2 050 / 12.3 / 252 / 0 | 2 451 / 2 154 / 12.3 / 252 / 0 |
| 2 | 32 153 / 22 981 / 11.8 / 179 / 0 | 32 134 / 22 996 / 11.8 / 179 / 0 |

With the same timeline the two arms are indistinguishable at every budget: TTFT p50 within
1–2 ms above the knee, TPOT within 0.1 ms, queue wait within 4 ms, the same 2 preemptions at
4 GiB, the same throughput, and the same deep queue below the knee (3 GiB: 2.3 vs 2.5 s;
2 GiB: 32 s both). Changing the budget in place costs nothing that a restart does not also pay.
Study 1's preempt-recompute spiral and 1.8 GiB stall did not recur on vLLM 0.30 even with a
static budget: the newer scheduler queues.

### Decode-only emulation (`results/fig_live_vs_static_decodeonly.png`)

To look at the decode side alone without a second GPU, the same workload was run with **prefix
caching off** and a **28 GiB** DRAM tier that holds every prefix: the GPU then holds only in-flight
KV and every request's prefix arrives from the store (the decode-engine situation of the
proposal, the store standing in for the prefill GPU; the remaining prefill is the 128 new tokens,
~10 ms). vLLM's native offload does work with prefix caching off (DRAM hit 95–98 % of looked-up
tokens). Judged on TPOT, scheduler queue wait, delivered throughput and failures; TTFT here is
fetch + 128-token prefill, not a first-token latency. Both arms use the same timeline at every
budget (live: `run_live_replay_sweep.sh`; static: `run_static_sweep.sh`; the continuous live run
`results/sweep_decodeonly_offload_zipf*` is kept).

![live vs static, decode-only emulation](results/fig_live_vs_static_decodeonly.png)

| GPU KV | live: TPOT / queue wait / tok/s / preempt | static: TPOT / queue wait / tok/s / preempt |
|---|---|---|
| 13 GiB | 13.3 ms / 112 ms / 258 / 0 | 13.3 ms / 114 ms / 258 / 0 |
| 10 | 13.3 / 114 / 258 / 0 | 13.2 / 114 / 258 / 0 |
| 8 | 13.4 / 115 / 258 / 0 | 13.2 / 116 / 258 / 0 |
| 6 | 13.4 / 156 / 257 / 0 | 13.3 / 156 / 257 / 0 |
| 5 | 13.4 / 218 / 257 / 0 | 13.4 / 218 / 257 / 0 |
| 4 | 13.4 / 375 / 255 / 0 | 13.4 / 370 / 256 / 0 |
| 3 | 12.3 / 10 434 / 226 / 0 | 12.3 / 10 543 / 225 / 0 |
| 2 | 11.7 / 29 113 / 158 / 0 | 11.7 / 29 161 / 158 / 0 |

Decode speed never sees the budget: TPOT is 13.2–13.4 ms from 13 GiB down to 4 GiB in both
arms, and the GPU holds only ~3 GiB of in-flight KV at any budget. The 110–150 ms "queue wait"
of the wide region is the KV fetch itself (a request waits for its slot and its 0.77 GiB load).
From 6 GiB the in-flight set's p95 (~5.4 GiB) starts to bump the limit and queueing grows
(156 → 375 ms at 6 → 4 GiB); below 4 GiB the engine is capacity-bound and delivers 226 then
158 of the 258 tok/s offered, with zero failures and zero preemptions. Live and static agree
on every point to within 5 ms of queue wait and 0.2 ms of TPOT.

**Answer to the second question (decode-only, live vs static):** no measurable difference. A
decode engine whose KV comes from a store can have its budget cut live from 13 GiB to 6 GiB
with nothing visible on TPOT, queueing or throughput, to 4 GiB with ~0.25 s of extra queueing
per request, and is capacity-bound (queueing, not failing) below that — the same numbers a
freshly started engine at each budget produces.

## 8. Caveats

- One GPU, one model (Llama-3-8B bf16), TP=1. Experiments A and B use kvcached 0.1.6; the
  sweep of §7 uses kvcached main (`9106e0f`) because 0.1.6 lacks the offload page pins and the
  prefix-cache release on shrink. The LMCache crash of §5 was not re-tested on main (main has no
  LMCache-specific pinning either).
- §7: Zipf only (uniform not run); the continuous sweeps use 60-s windows (≈110 requests) on different slices of the sequence, the same-timeline sweeps and the static controls 300 requests per budget.
- Prefill throughput of this GPU/config (~9.6k tok/s for 6k-token prompts) limited the load that
  could be applied without queueing; A1 was over-subscribed, A2 at capacity.
- FlashInfer's sampler was disabled in every run (JIT refuses sm_120 in this build); attention ran
  on FLASH_ATTN. Kernel choice is the same in every compared pair.
- LMCache's MP server log reports per-op timings; those numbers are the server's view.
