# CUPTI Auto-Profiler

An injection-based CUPTI library that automatically identifies the hottest GPU kernel in a running CUDA application and collects hardware performance counters for it — without modifying the target binary.

The profiler operates in a continuous **trace → profile → trace** cycle:

1. **Trace** for a configurable window (default 10 s), building a per-kernel hotspot table.
2. **Select** the kernel with the highest cumulative GPU time.
3. **Profile** the next launch of that kernel using CUPTI's range profiling API with hardware counter collection.
4. **Write** the results to a JSON file, then return to step 1.

## Architecture

The library is compiled as a shared object (`libauto_profiler.so`) and loaded via CUDA's injection mechanism. It combines three CUPTI subsystems:

| Subsystem | Purpose | When active |
|-----------|---------|-------------|
| **Activity API** | Records kernel launch timestamps to build the hotspot table | Tracing mode |
| **Callback API** | Intercepts `cuLaunchKernel` calls to detect the target kernel | Always |
| **Profiler API** | Collects hardware performance counters (AutoRange + KernelReplay) | Profiling mode |

### State machine

```
                   ┌──────────────────────────────────────────────┐
                   │                                              │
                   ▼                                              │
              ┌─────────┐    timer    ┌──────────────────────┐    │
              │ TRACING │───expires──▶│ TRANSITION_TO_       │    │
              │         │             │ PROFILING             │    │
              └─────────┘             └──────────┬───────────┘    │
                   ▲                             │                │
                   │                    target kernel launches    │
                   │                             │                │
                   │                             ▼                │
                   │                  ┌──────────────────────┐    │
                   │                  │ PROFILING_ACTIVE      │    │
                   │                  │ (kernel replayed for  │    │
                   │                  │  multi-pass counters) │    │
                   │                  └──────────┬───────────┘    │
                   │                             │                │
                   │                    kernel exits              │
                   │                             │                │
                   │                             ▼                │
                   │                  ┌──────────────────────┐    │
                   └──────────────────│ PROFILING_DONE        │────┘
                     evaluate +       │ (evaluate metrics,    │
                     write JSON       │  write results)       │
                                      └──────────────────────┘
```

Transitions happen on two threads:

- **State machine thread** — manages the timer, hotspot analysis, and mode transitions.
- **Application thread** (via CUPTI callback) — begins and ends the profiling session inside the `cuLaunchKernel` call, ensuring the correct CUDA context is active.

A `thread_local` flag and `compare_exchange_strong` on the mode atomic prevent race conditions when multiple application threads launch kernels concurrently.

## File structure

```
profiler/
├── profiler_common.h      Shared types, macros, config, global state
├── nvpw_metrics.h         NVPW / CUPTI Profiler API helper declarations
├── nvpw_metrics.cpp        └─ implementations
├── tracing.h              Activity-based tracing declarations
├── tracing.cpp             └─ implementations
├── auto_profiler.cpp      Entry point, callback handler, state machine
├── Makefile               Builds libauto_profiler.so
└── README.md              This file
```

### profiler_common.h

The shared header included by every translation unit. Contains:

- **Error macros** — `CUPTI_CALL` (fatal), `CUPTI_TRY` / `NVPW_TRY` (non-fatal, returns `false` so callers can fall back gracefully).
- **Configuration constants** — buffer size (32 KB), default trace duration (10 s), profiling timeout (30 s), default metric list.
- **Mode enum** — the five states of the profiler state machine.
- **Data structures** — `KernelStats` (per-kernel running totals), `Row` (flat snapshot for I/O), `ProfilerCtxData` (holds all NVPW/CUPTI binary blobs needed for a profiling session).
- **Global state declarations** — `extern` declarations for the kernel stats map, mode atomic, condition variable, profiler context, etc.
- **Inline utilities** — `demangleName()` (C++ symbol demangling), `getOutdir()`, `setupOutputPaths()`.

### nvpw_metrics.h / nvpw_metrics.cpp

All NVPW (Perfworks) and CUPTI Profiler API setup. This is the most complex module because hardware counter collection requires multiple binary blobs to be generated before a profiling session can start:

1. **`getRawMetricRequests()`** — Resolves high-level metric names (e.g. `sm__cycles_active.avg`) into the raw hardware counter names they depend on, using the NVPW MetricsEvaluator.

2. **`getConfigImage()`** — Generates the *config image*, a binary blob that tells CUPTI which hardware counters to program and how to schedule them across passes. Uses `NVPW_CUDA_RawMetricsConfig_*` functions.

3. **`getCounterDataPrefixImage()`** — Generates the *counter data prefix*, a template that sizes the results buffer. Uses `NVPW_CUDA_CounterDataBuilder_*` functions.

4. **`createCounterDataImage()`** — Allocates the *counter data image* (where CUPTI writes raw counter values) and its scratch buffer. Uses `cuptiProfilerCounterDataImage*` functions. Called once.

5. **`reinitCounterDataImage()`** — Clears stale counter data between profiling cycles. Reuses the same buffer.

6. **`evaluateMetrics()`** — Decodes raw counter values into human-readable metric doubles using the CUPTI Profiler Host API (`cuptiProfilerHostEvaluateToGpuValues`).

7. **`initializeProfilerContext()`** — One-time setup: initializes CUPTI Profiler + NVPW, discovers the chip name, checks device support, queries counter availability, then calls (2)–(4) above.

8. **`beginProfilingSession()` / `endProfilingSession()`** — Start and stop a CUPTI profiling session with AutoRange + KernelReplay mode. Must be called on a thread with an active CUDA context.

### tracing.h / tracing.cpp

The tracing-mode half of the profiler, using the CUPTI Activity API (same mechanism as the standalone tracer):

- **`bufferRequested()` / `bufferCompleted()`** — CUPTI buffer lifecycle callbacks. Allocate 32 KB heap buffers, iterate packed activity records, dispatch `CONCURRENT_KERNEL` records to the stats table.
- **`writeGlobalCsv()`** — Snapshots the kernel stats map under lock and writes `kernel_hotspots_global.csv`.
- **`findHottestKernel()`** — Scans the stats map and returns the kernel with the highest `total_ns`.
- **`writeProfilingJson()`** — Writes one profiling cycle's results (target kernel, trace-phase stats, metric values) to `profile_cycle_N.json`.

### auto_profiler.cpp

The main entry point and orchestration logic:

- **Global state definitions** — the single copy of all `extern` variables declared in `profiler_common.h`.
- **`callbackHandler()`** — CUPTI callback registered for two domains:
  - `CUPTI_CB_DOMAIN_RESOURCE` — captures the first CUDA context created by the application.
  - `CUPTI_CB_DOMAIN_DRIVER_API` (`cuLaunchKernel`) — in `TRANSITION_TO_PROFILING` mode, checks if the launched kernel matches the target. On match: reinitializes counter data, begins a profiling session (ENTER callback), ends the session (EXIT callback), then signals the state machine thread.
- **`stateMachineLoop()`** — background thread that cycles through six phases:
  1. Sleep for the trace duration.
  2. Flush activity buffers and find the hottest kernel.
  3. One-time profiler context initialization (if first cycle).
  4. Disable activity tracing, set target kernel, enter `TRANSITION_TO_PROFILING`.
  5. Wait (with timeout) for the callback to complete profiling.
  6. Evaluate metrics, write JSON, clear stats, re-enable tracing.
- **`finalizeProfiler()`** — `atexit` handler that shuts down the state machine thread, flushes remaining records, and writes a final CSV.
- **`InitializeInjection()`** — called by CUDA runtime via `dlsym`. Parses env vars, registers activity + callback APIs, starts the state machine thread.

## Running pipeline (step by step)

Here is what happens from `make llm-profile` to output files:

### 1. Build

```
make -C profiler
  g++ -c auto_profiler.cpp → auto_profiler.o
  g++ -c nvpw_metrics.cpp  → nvpw_metrics.o
  g++ -c tracing.cpp       → tracing.o
  g++ *.o → libauto_profiler.so  (linked against -lcupti -lnvperf_host -lnvperf_target)
```

### 2. Injection

The LLM app is launched with:
```bash
CUDA_INJECTION64_PATH=profiler/libauto_profiler.so python run_llm.py
```

Before any CUDA API call reaches the application, the CUDA runtime:
1. `dlopen()`s `libauto_profiler.so`
2. Calls `InitializeInjection()` via `dlsym`

### 3. Initialization (InitializeInjection)

- Reads `CUPTI_PROFILER_TRACE_S` (trace window) and `INJECTION_METRICS` (metric list) from environment.
- Registers CUPTI activity callbacks and enables `CONCURRENT_KERNEL` activity recording.
- Subscribes to CUPTI callbacks for `cuLaunchKernel` and context creation.
- Spawns the state machine background thread.
- Registers `finalizeProfiler()` with `atexit`.
- Returns control to the application.

### 4. Tracing phase

The application runs normally. As kernels execute on the GPU:

- CUPTI fills activity buffers with `CUpti_ActivityKernel9` records.
- `bufferCompleted()` processes each record: demangles the kernel name, computes `duration = end - start`, and increments `g_kernel_stats[name].count` and `.total_ns`.
- Meanwhile, the state machine thread is sleeping for the trace duration.

### 5. Hotspot selection

After the trace window expires:

- `cuptiActivityFlushAll(1)` drains any pending records.
- `writeGlobalCsv()` writes a snapshot of all observed kernels.
- `findHottestKernel()` picks the kernel with the highest `total_ns`.

### 6. Profiler context setup (first cycle only)

On the first transition to profiling mode:

- `cuptiProfilerInitialize()` + `NVPW_InitializeHost()` — global initialization.
- `cuptiDeviceGetChipName()` — discovers the NVPW chip name.
- `cuptiProfilerDeviceSupported()` — verifies range profiling is supported.
- `cuptiProfilerGetCounterAvailability()` — queries which HW counters exist.
- `getConfigImage()` — builds the config image via NVPW.
- `getCounterDataPrefixImage()` — builds the counter data prefix via NVPW.
- `createCounterDataImage()` — allocates and initializes the results buffer.

If any step fails, the profiler falls back to tracing-only mode permanently.

### 7. Profiling phase

Activity tracing is disabled (`cuptiActivityDisable`). The mode is set to `TRANSITION_TO_PROFILING`. The application continues running — all kernel launches pass through the callback handler, which checks each kernel's demangled name against the target.

When the application launches the target kernel:

- **ENTER callback**: `compare_exchange_strong` atomically claims the transition (so only one thread wins if the same kernel is launched concurrently). The counter data image is re-initialized to clear stale data. `beginProfilingSession()` calls `cuptiProfilerBeginSession` (AutoRange + KernelReplay), `SetConfig`, `EnableProfiling`. At this point CUPTI has programmed the GPU's hardware performance counters and is ready to collect.

- **The kernel executes with replay** — this is where the profiling phase fundamentally differs from normal execution. See the detailed explanation below.

- **EXIT callback**: `endProfilingSession()` calls `DisableProfiling`, `UnsetConfig`, `EndSession`. The mode is set to `PROFILING_DONE` and the condition variable is signaled.

#### What happens to the profiled kernel (AutoRange + KernelReplay)

This is the core of what makes profiling different from normal execution. Understanding it requires knowing how GPU hardware performance counters work.

**Normal execution (no profiler)**

When an application calls `cuLaunchKernel`, the CUDA driver submits the kernel to the GPU command queue. The GPU schedules it, the kernel runs once, produces its output, and the driver returns. The kernel's execution time is whatever the GPU hardware takes.

```
App calls cuLaunchKernel(myKernel, ...)
  └─► GPU runs myKernel once
       └─► writes output to device memory
  └─► cuLaunchKernel returns
```

**With the profiler (AutoRange + KernelReplay mode)**

GPUs have a limited number of hardware performance counter registers. A modern NVIDIA GPU might have ~16 counter registers per SM, but a single metric like `sm__cycles_active.avg` might require reading 2-3 raw counters, and different metrics often need counters from different groups that cannot be collected simultaneously. When you request 5 metrics, they might decompose into 10+ raw counters that require 2-3 separate "passes" through the kernel.

CUPTI handles this transparently using **Kernel Replay**:

```
App calls cuLaunchKernel(myKernel, ...)
  │
  ├─► CUPTI intercepts the launch (ENTER callback)
  │     └─► profiler enables hardware counter collection
  │
  ├─► CUPTI saves the kernel's input state
  │     • copies all input buffers (global memory the kernel reads)
  │     • records grid dims, block dims, shared memory, kernel arguments
  │
  ├─► Pass 1: GPU runs myKernel
  │     • hardware counters group A are programmed
  │     • kernel executes, counters collect data for group A
  │     • CUPTI reads counter values into the counter data image
  │     • output memory is discarded (rolled back)
  │     • input state is restored from the saved copy
  │
  ├─► Pass 2: GPU runs myKernel again
  │     • hardware counters group B are programmed
  │     • kernel executes identically (same inputs)
  │     • CUPTI reads counter values for group B
  │     • output memory is discarded again
  │     • input state is restored again
  │
  ├─► ... (repeat for as many passes as needed)
  │
  ├─► Final pass: GPU runs myKernel one last time
  │     • last counter group is collected
  │     • this time the output is kept (not rolled back)
  │     • the application sees the correct final result
  │
  ├─► CUPTI signals completion (EXIT callback)
  │     └─► profiler disables hardware counter collection
  │
  └─► cuLaunchKernel returns
       • from the application's perspective, nothing unusual happened
       • the kernel produced the correct output
       • but it took N× longer because it was replayed N times
```

Key points about what the replay does:

1. **The kernel is launched multiple times** but the application sees it as a single launch. The `cuLaunchKernel` call blocks for longer than usual because the kernel is being replayed internally.

2. **Input state is saved and restored** between passes. CUPTI snapshots the GPU memory regions the kernel reads before the first pass, and restores them before each subsequent pass. This ensures every pass sees identical inputs and produces identical execution behavior (same branch paths, same memory access patterns), so the counters from different passes are consistent with each other.

3. **Output is discarded until the final pass.** Intermediate passes might write garbage to output buffers (since the kernel runs to completion each time), but CUPTI rolls back those writes. Only the final pass's output is kept, so the application gets correct results.

4. **The application is unaware.** From the app's perspective, the kernel ran once and produced the right output. The only observable difference is that `cuLaunchKernel` took longer to return (wall-clock time increases by roughly N× for N passes, plus overhead for state save/restore).

5. **Other kernels are not affected.** Only kernels launched while profiling is enabled get replayed. Since we enable profiling in the ENTER callback and disable it in the EXIT callback of the same `cuLaunchKernel` call, only the target kernel is replayed. Other kernels launched by other threads during this window may also be replayed, but we only evaluate metrics for the target kernel's range.

#### Overhead comparison: tracing vs profiling vs normal

| | Normal execution | Tracing mode | Profiling mode |
|---|---|---|---|
| **Mechanism** | None | CUPTI Activity API | CUPTI Profiler API |
| **What is recorded** | Nothing | Kernel name + GPU timestamps | Hardware performance counter values |
| **Kernel behavior** | Runs once | Runs once (timestamps added) | Runs N times (replayed per pass) |
| **Performance impact** | Baseline | Low (~1-5% overhead) | High for the profiled kernel only (N× slowdown) |
| **GPU state** | Unmodified | Unmodified | Input memory saved/restored between passes |
| **Output correctness** | Correct | Correct | Correct (final pass output kept) |
| **Scope** | N/A | All kernels | Single target kernel per cycle |

#### Why multi-pass replay is necessary

A natural question is: why can't the GPU just collect all counters in a single pass?

The answer is hardware constraints. GPU streaming multiprocessors (SMs) have a fixed number of performance monitoring registers. These registers are multiplexed: each register can count one specific hardware event at a time (e.g., "FMA instructions executed" or "L2 cache misses"). Different metrics require different events, and many events conflict — they need the same physical counter register.

NVPW's config image generation (in `getConfigImage()`) figures out the minimum number of passes needed to collect all requested raw counters without conflicts, and encodes the counter programming schedule. CUPTI's KernelReplay mode then executes that schedule automatically.

For example, with 5 default metrics:
- `sm__cycles_elapsed.avg` and `sm__cycles_active.avg` might share a pass (non-conflicting counters)
- `sm__warps_active.avg` might need its own pass (conflicts with the above)
- `dram__bytes_read.sum` and `dram__bytes_write.sum` might share a third pass

This would result in 2-3 replay passes. The exact number depends on the GPU architecture and the specific metrics requested.

### 8. Evaluation and output

The state machine thread wakes up (signaled by the EXIT callback via the condition variable).

At this point, `g_profiler_data.counterDataImage` contains raw hardware counter values collected across all replay passes. These are raw register reads — not yet the human-readable metrics the user requested. The evaluation step decodes them:

- `evaluateMetrics()` initializes a CUPTI Profiler Host API object for the chip, then calls `cuptiProfilerHostEvaluateToGpuValues()`. This function takes the raw counter data image and the metric names, and computes the final metric values by:
  - Looking up which raw counters each metric depends on
  - Applying the metric's formula (e.g., `sm__cycles_active.avg` = sum of active cycles across all SMs / number of SMs)
  - Returning one `double` per metric

- `writeProfilingJson()` writes the results to `output/profile_cycle_N.json`.
- A summary is printed to stderr showing each metric name and value.

### 9. Reset and repeat

- Kernel stats are cleared.
- Activity tracing is re-enabled.
- Mode returns to `TRACING`.
- The cycle repeats from step 4.

### 10. Finalization

When the application exits, `finalizeProfiler()`:
- Sets mode to `SHUTDOWN` and joins the state machine thread.
- Synchronizes the GPU and flushes remaining activity records.
- Writes a final CSV snapshot.

## Output files

| File | Description |
|------|-------------|
| `kernel_hotspots_global.csv` | Hotspot table at each trace-phase snapshot |
| `profile_cycle_1.json` | Profiling results for cycle 1 |
| `profile_cycle_2.json` | Profiling results for cycle 2 |
| ... | One JSON per profiling cycle |

CSV columns: `kernel_name`, `launch_count`, `total_duration_ms`, `avg_duration_us`

JSON structure:
```json
{
  "cycle": 1,
  "target_kernel": "void cutlass::Kernel2<...>(...)",
  "trace_duration_s": 10,
  "total_launches": 1500,
  "total_gpu_time_ms": 4500.0,
  "avg_duration_us": 3000.0,
  "metrics": {
    "sm__cycles_elapsed.avg": 123456.0,
    "sm__cycles_active.avg": 98765.0,
    "sm__warps_active.avg": 42.5,
    "dram__bytes_read.sum": 1048576.0,
    "dram__bytes_write.sum": 524288.0
  }
}
```

## Understanding the profiled metrics

### NVPW metric naming convention

NVIDIA Perfworks metric names follow a structured format:

```
<unit>__<quantity>.<rollup>
```

- **unit** — the hardware unit being measured (e.g. `sm` = streaming multiprocessor, `dram` = device memory, `lts` = L2 cache).
- **quantity** — what is being counted (e.g. `cycles_active`, `bytes_read`, `warps_active`).
- **rollup** — how per-unit values are aggregated: `.avg` (average across all units), `.sum` (total across all units), `.min`, `.max`, `.per_cycle_active`, etc.

For example, `sm__cycles_active.avg` means: "the number of cycles during which the SM was doing useful work, averaged across all SMs on the GPU."

### Default metrics explained

These are the five metrics collected by default. They give a first-order picture of whether a kernel is compute-bound, memory-bound, or underutilizing the GPU.

#### `sm__cycles_elapsed.avg`

**What it is:** The total number of GPU clock cycles that elapsed from the start to the end of the kernel, averaged across all SMs.

**What it tells you:** This is the wall-clock duration of the kernel in GPU cycles. Multiply by the GPU clock period to get time in nanoseconds. It includes all time — active computation, stalls, idle gaps, everything.

**How to use it:** This is the denominator for utilization calculations. Comparing it against `sm__cycles_active.avg` tells you what fraction of elapsed time the SMs were actually busy.

#### `sm__cycles_active.avg`

**What it is:** The number of GPU clock cycles during which the SM had at least one active warp (i.e., was doing useful work), averaged across all SMs.

**What it tells you:** How much of the kernel's execution time the SMs were actually doing something, as opposed to sitting idle waiting for memory, synchronization, or other stalls.

**How to use it:** Compute the **SM active ratio**:

```
active_ratio = sm__cycles_active.avg / sm__cycles_elapsed.avg
```

- Close to 1.0 → the SMs are busy for the entire kernel duration. Good utilization.
- Much less than 1.0 → the SMs are frequently idle. The kernel is likely bottlenecked on memory, synchronization barriers, or has insufficient parallelism.

#### `sm__warps_active.avg`

**What it is:** The average number of warps (groups of 32 threads) that are resident and eligible for execution on an SM, averaged across all SMs and all cycles.

**What it tells you:** This is a direct measure of **occupancy** — how well the kernel keeps the SM's warp schedulers fed with work. Modern GPUs can have 32–64 concurrent warps per SM; this metric tells you how many you are actually using.

**How to use it:**

- Divide by the GPU's maximum warps-per-SM to get the occupancy percentage. For example, if max is 48 and `sm__warps_active.avg` is 24, occupancy is 50%.
- Low occupancy means the warp schedulers have fewer warps to choose from, which reduces the GPU's ability to hide memory latency through warp switching. This is often caused by:
  - High register usage per thread (limits concurrent warps)
  - High shared memory usage per block
  - Small grid sizes (not enough blocks to fill the GPU)

#### `dram__bytes_read.sum`

**What it is:** The total number of bytes read from device memory (DRAM / HBM) across the entire GPU during the kernel execution.

**What it tells you:** How much data the kernel fetched from global memory. This is the actual bytes transferred on the memory bus, which may be higher than what the kernel logically requested due to cache line granularity (memory transactions are 32-byte or 128-byte sectors).

**How to use it:** Compute the **memory read throughput**:

```
read_throughput_GB_s = dram__bytes_read.sum / (kernel_duration_ns) * 1e9 / 1e9
```

Compare this against the GPU's peak memory bandwidth. For example:
- A100: ~2 TB/s peak HBM bandwidth
- RTX 4090: ~1 TB/s peak GDDR6X bandwidth
- If your read throughput is a large fraction of peak, the kernel is **memory-read-bound**.

#### `dram__bytes_write.sum`

**What it is:** The total number of bytes written to device memory across the entire GPU during the kernel execution.

**What it tells you:** Same as `dram__bytes_read.sum` but for writes. Includes both store instructions and write-back from caches.

**How to use it:** Same throughput calculation as reads. Add reads + writes for total memory bandwidth utilization:

```
total_bandwidth = (dram__bytes_read.sum + dram__bytes_write.sum) / kernel_duration_ns * 1e9
```

### Putting the metrics together

The five default metrics answer the key performance question for any GPU kernel: **what is the bottleneck?**

| Observation | Diagnosis |
|-------------|-----------|
| High `cycles_active / cycles_elapsed`, low DRAM bytes | **Compute-bound** — the SMs are busy doing math, memory is not the bottleneck |
| Low `cycles_active / cycles_elapsed`, high DRAM bytes | **Memory-bound** — the SMs are frequently stalled waiting for data from DRAM |
| Low `cycles_active / cycles_elapsed`, low DRAM bytes | **Latency-bound** — stalled on something else (synchronization, L2 misses not turning into DRAM traffic, small grid) |
| Low `warps_active` | **Low occupancy** — not enough concurrent warps to hide latency; consider reducing register/shared memory usage or increasing grid size |
| High `warps_active`, low `cycles_active` | **Memory-latency-bound** — many warps are resident but all are stalled on memory; the memory subsystem can't keep up |

### Customizing metrics

Set `INJECTION_METRICS` to collect different counters:

```bash
INJECTION_METRICS="sm__cycles_active.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed" \
CUDA_INJECTION64_PATH=... ./your_app
```

Some useful metrics for deeper analysis:

| Metric | What it measures |
|--------|-----------------|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM bandwidth utilization as % of peak |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | L2 cache throughput as % of peak |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Overall SM utilization as % of peak |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | FMA pipe utilization |
| `sm__inst_executed.avg.per_cycle_active` | Instructions per cycle (IPC) |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | L1 cache sectors read for global loads |
| `smsp__sass_thread_inst_executed_op_dadd_pred_on.avg` | Double-precision add instructions |

The full list of available metrics for your GPU can be queried with `ncu --query-metrics`.

## Build

```bash
make                # builds libauto_profiler.so
make clean          # removes .o and .so files
make print-config   # shows CUDA paths
```

Requires CUDA 12.4+ with CUPTI, NVPW (nvperf_host, nvperf_target). Edit `CUDA_HOME` in the Makefile if your CUDA install is not at `/usr/local/cuda`.

## Usage

### With the LLM workload (from the parent directory)

```bash
cd cupti/
make llm-profile
```

This builds the profiler and runs the LLM app with injection. `sudo` is required because CUPTI's profiling API needs elevated privileges for hardware counter access.

### With any CUDA application

```bash
sudo CUDA_INJECTION64_PATH=$(pwd)/libauto_profiler.so \
     CUPTI_TRACE_OUTDIR=output \
     LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
     ./your_cuda_app
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUPTI_PROFILER_TRACE_S` | `10` | Seconds to trace before selecting the hottest kernel |
| `CUPTI_TRACE_OUTDIR` | `output` | Directory for CSV and JSON output files |
| `INJECTION_METRICS` | See below | Comma/semicolon-separated list of NVPW metric names |

Default metrics:
- `sm__cycles_elapsed.avg`
- `sm__cycles_active.avg`
- `sm__warps_active.avg`
- `dram__bytes_read.sum`
- `dram__bytes_write.sum`

## Implementation notes

- **Thread safety**: A single `std::mutex` protects the kernel stats map. Mode transitions use `std::atomic<Mode>` with `compare_exchange_strong` to prevent races in multi-threaded applications.
- **Graceful fallback**: If profiler initialization fails (unsupported GPU, missing privileges, NVPW errors), the library continues in tracing-only mode rather than crashing the target application.
- **Activity/Profiler coexistence**: CUPTI activity recording is disabled during profiling to avoid interference with hardware counter collection, then re-enabled afterward.
- **Counter data reuse**: The config image, counter data prefix, and scratch buffers are created once and reused. Only the counter data image itself is re-initialized between cycles.
- **Profiling timeout**: If the target kernel does not launch within 30 seconds of entering `TRANSITION_TO_PROFILING`, the profiler reverts to tracing.
