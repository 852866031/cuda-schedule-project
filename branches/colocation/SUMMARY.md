# Colocator — Summary of What Was Built and What Was Tested

Date: 2026-07-27 · Machine: 2× RTX 5090 (sm_120, 170 SMs), driver 580.126, CUDA 12.8,
PyTorch 2.11 cu128, conda env `colocator` · Reference: Orion (EuroSys '24), `orion/`
(read-only). Design doc: [PROPOSAL.md](PROPOSAL.md) · Run guide: [README.md](README.md)

---

## 1. What was built

A minimal Orion-style GPU kernel colocator plus an independent GPU-side observer.
Everything lives under `colocator/`; three C++ shared libraries, a Python frontend,
two demo workloads, and an analysis pipeline.

### 1.1 Interception layer — `src/intercept/` → `build/libcolocator.so`

- `LD_PRELOAD` symbol interposition of the CUDA runtime API (same mechanism as
  Orion): `cudaLaunchKernel`, `cudaMalloc`, `cudaFree`, `cudaMemcpy`,
  `cudaMemcpyAsync`, `cudaMemset`, `cudaMemsetAsync`, `cudaStreamSynchronize`
  (+ `cudaLaunchKernelExC` counted/warned). Real functions resolved via
  `dlsym(RTLD_NEXT)`.
- Two paths per interposer, decided by the **calling thread's TID**:
  - *registered client thread* (managed mode): the call is not executed — its
    arguments are packed into a `FuncRecord` **on the caller's stack**, pushed to
    that client's mutex-guarded queue, and the caller spins until the scheduler
    flips the record's state (`ISSUED` for async ops, `DONE` for sync ops).
    Orion-style lockstep: queue depth ≤ 1 per client. 30 s watchdog aborts loudly
    if the scheduler dies.
  - *any other thread* (Python main thread, CUPTI workers, the scheduler itself):
    passthrough + per-thread/per-kind counters, exit-time summary. This is also
    how the scheduler's own replayed calls reach the real runtime without
    recursion.
- `cudaStreamSynchronize` is **redirected**: it completes when the client's
  *colocator* stream drains (the stream the client names is meaningless once its
  work is re-routed) — this is what makes `.cpu()` / `.item()` correct.
- Client registration is explicit from Python (`col_register_client(idx)`) before
  the thread touches CUDA.

### 1.2 Scheduler core — `src/sched/` → `build/libsched.so`

- One **CUDA stream per client**, created with `cudaStreamCreateWithPriority`,
  priority configurable per client from the CLI (default 0/none; on this device
  the range is 0…−5).
- Busy-wait loop (dedicated thread, GIL released via ctypes): peek every queue
  head → ask the **Policy** which client to serve → `execute()` replays the
  record through the real CUDA call **with the stream argument substituted** by
  that client's colocator stream → pop.
- **Pluggable policy interface** (`src/sched/policy.h`):
  `pick_next(heads, SchedState) -> client | -1`. Shipped policy: **FCFS**
  (earliest intercept timestamp across queue heads). An Orion/REEF-style policy
  is a new subclass; the mechanism doesn't change.
- Sync-semantics ops complete by **polling** `cudaStreamQuery` from the loop
  (`pending_sync` slot per client) instead of blocking in
  `cudaStreamSynchronize` — a blocking sync was measured to stall the other
  client's queue for a full 21 ms kernel.
- In-memory **issue log** (no hot-path I/O): per op `client, op_id, kind,
  t_intercept, t_issue, t_returned, func, grid/block, bytes, stream` → CSV at
  teardown. Accounting counters (`submitted` vs `issued`) per client.
- Shared state between the two libraries is resolved by the dynamic linker at
  load time (libcolocator is preloaded into global scope) — no dlsym wiring
  needed, simpler than Orion's.

### 1.3 Observer — `src/observer/` → `build/libobserver.so`

- **Passive by construction**: talks only to CUPTI; never touches queues, locks,
  or streams; 8 MB activity buffers drained only at `obs_dump()`.
- Records: `CONCURRENT_KERNEL` (GPU start/end ns per kernel — the variant that
  does *not* serialize execution), `MEMCPY` (DMA copies), `RUNTIME` (one record
  per CUDA API call with the caller's Linux TID via
  `cuptiSetThreadIdType(SYSTEM)` and a correlation id shared with the GPU
  activity it caused), plus clock-calibration samples
  (`cuptiGetTimestamp` ↔ `CLOCK_MONOTONIC_RAW`).
- Join logic (in analysis): the scheduler issues ops strictly one-at-a-time from
  one thread ⇒ its k-th `cudaLaunchKernel`/`cudaMemcpyAsync` RUNTIME record (by
  start time) is the k-th kernel/memcpy row of the issue log ⇒ correlation id ⇒
  GPU record. Result: **four timestamps per op** on one clock:
  `t_intercept ≤ t_issue ≤ t_gpu_start < t_gpu_end`.

### 1.4 Workloads & demo — `ext/`, `demo/`

- `ext/colocator_kernels`: tiled shared-memory fp32 SGEMM as a torch CUDA
  extension. Exists because `torch.matmul` → cuBLAS → driver-API launches that
  would **bypass** `cudaLaunchKernel` interception (the reason Orion interposes
  cuBLAS); extension kernels launch through `cudaLaunchKernel` like ATen native
  ops, so v1 interception covers 100% of the demo's kernels.
- Two clients, each doing the full representative cycle per iteration
  (H2D copy → compute kernels → async D2H into a **pinned** buffer → stream
  sync):
  - `latency`: 4×(256×1024×1024 SGEMM → relu) + row-sum; 9 kernels + 2 copies
    per iter; ~0.46 ms/iter solo.
  - `throughput`: 64 MiB weight-shard H2D + one 4096³ SGEMM (saturates all
    170 SMs) + row-sum; 2 kernels + 2 copies per iter; ~21 ms/iter solo.
- `demo/run_demo.py` modes: `seq` (solo baselines), `streams` (plain
  `torch.cuda.Stream`s, no colocator), `colocated` (through the colocator;
  re-execs itself to set `LD_PRELOAD`; each client in its own torch-stream
  context so it gets its own caching-allocator pool — prevents cross-client
  block reuse hazards).
- `analysis/analyze.py` (trace join, self-checks, metrics) and
  `analysis/plot_timeline.py` (per-client GPU Gantt + latency CDFs).

### 1.5 Documentation

Run-from-scratch [README.md](README.md) with phase-status table and results;
per-directory READMEs (`src/` with mermaid data-flow chart, `src/intercept`,
`src/sched`, `src/observer`, `src/common`, `ext/`, `demo/`, `analysis/`);
[PROPOSAL.md](PROPOSAL.md) with the Orion code study and the phased plan.

---

## 2. What was tested (and the result)

Every phase gated on runnable verification (PROPOSAL.md §7); all pass on the
target machine.

### 2.1 Functional correctness

| Test | How | Result |
|---|---|---|
| Environment | `tools/check_env.py` (torch+CUDA, GPUs, nvcc, CUPTI, g++) | pass |
| Interposer transparency | `demo/smoke_test.py` with and without `LD_PRELOAD`, results vs CPU | both `OK`; 25 kernels + 10 copies + 5 syncs counted |
| Interception coverage | workloads under passthrough interposer; counts vs analytic inventory | exact match (e.g. latency ×10 iters = 90 kernels); **zero** `cudaLaunchKernelExC` |
| Workload numerics | one full iteration vs CPU float64 reference, `torch.allclose` | pass for both clients, in **every** mode (solo, streams, colocated, priority) |
| Colocated single client | latency client alone through colocator | check passes; 299 submitted == 299 issued; p50 identical to bare solo (0.46 ms → no measurable overhead) |
| Colocated two clients | both clients concurrently | checks pass; 419/419 and 176/176 submitted==issued; clean exit, watchdog silent |
| Observer self-check | `analyze.py --self-check` | 515/515 ops joined with all four timestamps; ordering holds; per-client GPU start order monotonic; **0 leaked kernels** (none launched by client threads directly); clock drift 0.02–0.06 ms |
| External validation | `nsys profile` of a colocated run vs observer | kernel instance counts match exactly (170 sgemm / 136 elementwise / 68 reduce / 1 init); durations agree |
| Observer perturbation | `--observer on` vs `off` | p50/p95 within run-to-run noise |

### 2.2 Evaluation: concurrency and interference (30 iters/client, FCFS)

| config | latency p50 / p95 | throughput | A/B kernel overlap |
|---|---|---|---|
| solo (`seq`) | 0.46 / 0.47 ms | 48.3 it/s | — |
| plain two streams | 20.8 / 21.7 ms | 47.4 | ~0 |
| colocated, prio 0,0 | 20.8 / 21.7 ms | 47.6 | 42% of latency client's busy time |
| colocated, prio **−5**,0 | **0.98 / 5.2 ms** | 46.9 (−2%) | **89%** |

Findings the four-timestamp trace makes provable:

1. **Two streams alone give no concurrency against an SM-saturating kernel**:
   at equal priority the latency client's delay is entirely hardware-side
   (`launch→gpu_start` p50 ≈ 20 ms behind the SGEMM's 16 K blocks) while its
   software queue delay is 0.2 µs — a 45× p50 degradation vs solo.
2. **Configurable stream priority restores latency at ~2% throughput cost**
   (p50 back to ~1 ms), with real SM sharing: 89% of the latency client's
   GPU-busy time overlaps the throughput client's kernels.
3. **Interference is measurable per kernel class**: while actually overlapping,
   the latency client's elementwise kernels slow 1.5→3.1 µs and its SGEMMs
   90→112 µs (cross-mode GPU-duration table).
4. **Colocator overhead is negligible** for these workloads: software queue
   delay p95 ≤ 1 µs after the fixes below.
5. Two scheduler-side hazards were found *by the measurements* and fixed:
   blocking sync ops stalled the whole loop (→ `cudaStreamQuery` polling), and
   pageable-D2H (`.cpu()`) makes `cudaMemcpyAsync` silently synchronous,
   stalling the scheduler and coupling both clients (→ pinned result buffers +
   redirected stream sync).

Artifacts: `runs/full/` and `runs/prio/` (results.json, issue_log.csv,
obs_*.csv, trace.json, `timeline_colocated.png`, `latency_cdf.png`),
`runs/nsys_check.nsys-rep`.

---

## 3. Known limitations / natural next steps

- Only the v1 CUDA-runtime op set is managed; cuBLAS/cuDNN entry points are not
  interposed (workloads avoid them by design; stock `torch.matmul` would leak —
  and the observer's leak check would flag it). Orion-style cuBLAS interposers
  are the Phase 6 stretch.
- Inference-only clients (no autograd-thread adoption), single GPU, lockstep
  queue depth 1 (`COLOCATOR_QUEUE_DEPTH>1` unimplemented).
- FCFS is intentionally dumb — the `Policy` interface plus this baseline data
  is the starting point for REEF/Orion-style policies.
