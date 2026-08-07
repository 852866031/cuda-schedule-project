# src/observer/ — passive CUPTI observer (`libobserver.so`)

Answers "when did each operation ACTUALLY run on the GPU" without touching
the colocator: no shared queues, no locks, no streams — it only talks to
CUPTI. Driven from Python (`run_demo.py --observer on`): `obs_init()` before
CUDA work, `obs_dump(dir)` at the end of each mode.

## What it records (CUPTI Activity API)

| Activity kind | Gives us | File |
|---|---|---|
| `CONCURRENT_KERNEL` | per-kernel GPU `start`/`end` (ns), stream, grid/block, mangled name, correlation id | `obs_kernels.csv` |
| `MEMCPY` | same for DMA copies (bytes, direction) | `obs_memcpys.csv` |
| `RUNTIME` | one record per CUDA runtime call: calling thread's Linux TID (`cuptiSetThreadIdType(SYSTEM)`), correlation id shared with the GPU activity it caused | `obs_runtime.csv` |
| — clock samples | `cuptiGetTimestamp()` bracketed by `CLOCK_MONOTONIC_RAW` at init + dump | `obs_calib.csv` |

`CONCURRENT_KERNEL` (not `KERNEL`) matters: the plain KERNEL activity kind
serializes kernel execution and would destroy the concurrency we're measuring.

## How records join back to the colocator (in analysis/analyze.py)

The scheduler issues ops strictly one at a time from one thread, so its k-th
`cudaLaunchKernel` RUNTIME record (in start-time order, filtered by the
scheduler's TID) *is* the k-th `KernelLaunch` row of the issue log. That
gives each issue-log row a correlation id → GPU activity record → GPU
start/end. Same for `cudaMemcpyAsync`. RUNTIME TIDs also power the leak
check: a kernel whose runtime record came from a *client* TID bypassed the
colocator.

## Live feed (opt-in, used by the scheduler)

`obs_live_enable(period_ms)` starts an observer-owned flusher thread calling
`cuptiActivityFlushAll(0)` on the period (`cuptiActivityFlushPeriod` proved
ineffective on this CUPTI: partially-filled buffers were never delivered, so
live counters stayed 0 — found via a standalone probe);
`obs_live_track(stream)` registers a stream (via `cuptiGetStreamIdEx`);
`obs_live_get(slot, …)` polls per-stream counters — completed kernels/copies
and summed measured kernel busy ns — updated by CUPTI's own worker thread as
records arrive (lag ≤ flush period, tail records occasionally later). This
keeps the observer passive: the scheduler only reads atomics; nothing touches
its queues or streams. Note CUPTI reports kernels **at completion** — the
feed can never say what is executing this instant.

RUNTIME records are filtered to the two cbids analysis correlates on
(cudaLaunchKernel, cudaMemcpyAsync): the scheduler's event/stream polling
would otherwise generate millions of useless records per run.

## Non-interference

- Buffers (8 MB) are handed to CUPTI asynchronously; parsing happens in
  CUPTI's own worker thread (unregistered → its CUDA calls pass through);
  everything is drained only at `obs_dump` (`cuptiActivityFlushAll`).
- No I/O, no colocator locks during the run; rows accumulate in memory.
- CUPTI adds small in-driver per-launch overhead that applies equally to all
  clients; measured impact on the demo is below run-to-run noise
  (`--observer off` vs `on`: same p50/p95 within noise).

## Clock domains

CUPTI timestamps and the colocator's `CLOCK_MONOTONIC_RAW` are different
clocks. `obs_calib.csv` holds side-by-side samples from init and dump;
analysis maps CUPTI→mono with the mean offset and *checks* the drift across
the run (assertion tolerance scales with it; measured drift ≈ 0.01 ms over a
demo run).
