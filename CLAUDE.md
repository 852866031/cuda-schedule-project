# Working in this repo

Nine completed studies of VRAM limits and GPU sharing on a 2× RTX 5090 (32 GB, PCIe
Gen4 ×8, no P2P — GeForce) workstation with 32 cores and 60 GiB RAM. Plans (`PLAN*.md`)
are pre-registration documents: never rewrite their predictions, only annotate outcomes.
Run-level decisions go in a decision log (`DECISIONS_*.md`): each experiment's prediction
written before it runs, its outcome after. Results live in `reports/`, figures in
`figures/`, all data under `output/` (summaries, raw JSONs, per-second GPU and host
telemetry in `output/gpumon/`, engine logs in `output/logs/`).

## Environments — non-negotiable details

- `.venv-matched` runs vLLM **0.15.1 (pinned)**; `.venv` is for everything else.
- Every vLLM invocation needs `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`
  (conda's libstdc++ is too old). Launch scripts set it.
- **lmcache 0.4.4 is the only release that pairs with this vLLM** (older = torch-ABI
  mismatch, 0.5.x = adapter break; its compiled ops never load against torch 2.9.1 —
  a source build deadlocks at engine init; the Python fallback is what all measurements
  used). pip drags `transformers` to 5.x when installing it — re-pin `<5` after.
- `PYTHONHASHSEED=0` is mandatory for any cross-process LMCache use.
- **More than one vLLM engine on the box: set `OMP_NUM_THREADS` ≤ cores ÷ engines for
  every engine** (4 was used; 2/4/8 behave identically). Each engine's torch/OpenMP pool
  defaults to all 32 cores (~135 threads per EngineCore). Uncapped, 4 engines with
  DRAM-offloaded KV spin-wait on all 32 cores, the GPU idles and requests time out
  (4 of 4 runs); 8 engines with resident KV storm transiently (2 of 4). The cap costs
  nothing at N=1. Export it before the driver — the launch scripts inherit it.
- No ptrace, perf or sudo on this box (`ptrace_scope=1`, `perf_event_paranoid=4`): stack
  sampling is unavailable; `scripts/inf_multi_coloc/thread_sampler.py` reads per-thread
  `/proc` counters instead.

## Safety — this machine has been frozen and rebooted by pinned memory

Pinned (page-locked) host memory is unswappable; overcommitting it does not OOM-kill, it
hard-freezes the box. The P2P connector defaults to **32 GiB pinned per instance**.
Rules: never launch vLLM stacks except through the launch scripts (they budget-check),
keep `scripts/common/mem_guard.sh` running (they start it), and remember the pool
allocator rounds sizes up to powers of two. The stall/heartbeat watchdogs exist because
every failure mode of the KV-transfer path is a silent hang, not an error. Run strictly
one stack set at a time. Host RAM is the ceiling for offload: budget ~5 GB of RSS per
engine (API server + EngineCore incl. 1 GiB pinned L1) plus the store's working set; the
box was swap-backed at 6 × 0.5B with 3.4 GiB stores.

## Layout

- `scripts/inference/simple/` — colocated study (also `--backend lmcache`).
- `scripts/inference/split_simple/` — hand-built P2P split + `p2p_patch/` (four
  connector fixes, loaded via PYTHONPATH sitecustomize) + `disagg_sweep.py`, the
  budget-sweep driver for BOTH split stacks (`--stack p2p|lmcache`).
- `scripts/inference/split_lmcache/` — shared-LMCache split (server wrapper adds
  SO_REUSEADDR; launch with `--warmup-qps 0.5` or the L1 staging pool overflows).
- `scripts/inf_ft_coloc/` — colocate-finetuning study (split-LMCache stack + a LoRA
  trainer on GPU1; `coloc_sweep.py`, MPS/gate/CUPTI arms in `orion_gate/`).
- `scripts/inf_inf_coloc/` — colocate-a-second-model study (self-contained copies +
  `coloc2_*.sh` launch both stacks with one teardown; `coloc2_sweep.py` runs the 8B in
  process and the Qwen tenant as a subprocess). Env knobs: `TASKSET_ISOLATE`,
  `--qwen-decode-skew`; MPS is run by exporting `CUDA_MPS_PIPE_DIRECTORY` before the
  driver (start `nvidia-cuda-mps-control -d` first, `echo quit | …` to tear down).
- `scripts/inf_multi_coloc/` — N homogeneous models on GPU1 (self-contained copies).
  `multi_sweep.py` brings engines up one at a time via `multi_launch.sh`, owns the MPS
  daemon (`--arm mps`), runs all requested cells against one launch
  (`--cells dfits doff ffits foff`), sweeps load with `--qps a b c`, and rebuilds
  `output/summary_mc_<cohort>*.csv` from raw on every run. `--cohort small|medium` =
  Qwen2.5-0.5B (util 0.08) | Qwen2.5-3B (util 0.22, `max-num-seqs 16`, sized with
  `probe_kv.sh`). `host_monitor.py` / `thread_sampler.py` are started per launch.
  `mc_scheduler.py` + the `pfill` cell (external cross-model scheduler) exist but were
  never run. Queue scripts (`queue_*.sh`) record the exact order of what was run.
- `scripts/inf_pf_coloc/` — a 3B prefill on the 8B split's decode GPU (`pfc_sweep.py`, one
  launch per cell; `pfc_launch.sh big|small`). The 3B store is capped (`LMC_STORE_CAP_GB`,
  the stock store never evicts). `async_save_patch/` (`--sm-async-save`) returns token #1
  before LMCache's GPU→host copy; the 3B router then waits for its marker file before the
  decode leg (the stock split races: ~27% partial KV hits).
- `scripts/inf_pf_prio/` — 8B-decode vs 3B-prefill priority (P2 only; `pfp_sweep.py --arm
  fair|mps8b|gate8b|gate8bt|mps3b|gate3b|eager8b`). MPS priority = `CUDA_MPS_CLIENT_PRIORITY=1`
  on the low-priority engine; gating = `site/hp_signal.py` (busy window) + `orion_gate/
  cupti_gate.so` (forces eager). `--sm-quant fp8` loads bf16 first (7–9 GiB peak): start
  it before the 8B (`--sm-first`). The 3B store falls back to the next port if :8301 will
  not bind (seen 3×, cause unknown).
- `scripts/plots/` — one script per figure family; regenerate after data changes and
  **always view the PNG before finishing** (label collisions are the recurring bug).

## Measurement conventions

- The reference workload is identical across all studies (32 sessions × 6144-token fixed
  prefixes + 128 unique suffix tokens, 128 out, zipf-1.1, open-loop 2 QPS, seeded and
  fully deterministic). Don't invent variants.
- Split-system TTFT is only honest with the router's `--forward-first-token` and no
  admission cap (`--max-inflight 999`); the capped/discarding router variants were setup
  faults — do not report their numbers.
- Config names must never collide across arms/runs (name suffixes like `_lmcache`,
  `_fwd_ng`); raw files are overwritten by name, and original study data has been lost
  that way once.
- p50 TTFT run-to-run variance on flat regions is ±15 ms; throughput dips are the
  trustworthy knee signals.
- **Telemetry file names must not collide either.** Per-launch traces
  (`output/gpumon/<tag>{,_host,_threads}.csv`) were overwritten once by later launches
  with the same N/arm/suffix; `multi_sweep.py` now appends `_L2`, `_L3`, … and records
  `telemetry_tag` in each raw record. Engine logs are archived per launch
  (`output/logs/mc_archive/<tag>/`) because `mc_engine_<i>.log` is reused.
- **Decode-only TTFT is N/A.** With no prefill it is queue + KV-reload time, not a
  time-to-first-token. Judge decode-only cells on TPOT, throughput, failed requests and
  the engine's queue-wait metric.
- **Capacity = scheduler queue wait** (`vllm:request_queue_time_seconds`, mean < 50 ms and
  ≤ 1% failed), found by sweeping per-model QPS at fixed N. Not delivered/offered: with a
  fixed request count the window's drain tail reads 92–94% at high QPS with zero
  queueing. Not SM-active either: it fell while throughput doubled.
- A cell is `ok` only with zero failed requests; a lone client-side broken pipe is
  recorded, not treated as a collapse (plots ring a cell only above 1% failed).
- `--gpu-memory-utilization` fixes the footprint; within it the KV grant trades against
  `--max-num-seqs` (3B at util 0.22: 64 → 0.59 GiB, 16 → 0.94 GiB), and that cap then
  bounds per-engine capacity. Size both together, by probe, with the GPU verified empty
  between probes.
