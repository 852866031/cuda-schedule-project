# CUPTI Auto-Profiler

An injection-based CUPTI library that automatically identifies the hottest GPU kernel in a running CUDA application and collects hardware performance counters for it — without modifying the target binary.

The profiler operates in a continuous **trace → profile → trace** cycle:

1. **Trace** for a configurable window (default 10 s), building a per-kernel hotspot table.
2. **Select** the kernel with the highest cumulative GPU time.
3. **Profile** the next launch of that kernel using CUPTI's range profiling API with hardware counter collection.
4. **Write** the results to a JSON file, then return to step 1.

## Architecture

The library is compiled as a shared object (`libauto_profiler.so`) and loaded via CUDA's injection mechanism (`CUDA_INJECTION64_PATH`). Before the target application's first CUDA call, the CUDA runtime `dlopen`s the library and calls its `InitializeInjection()` entry point. From that point on, the profiler runs invisibly alongside the application.

It combines three CUPTI subsystems:

| Subsystem | Purpose | When active |
|-----------|---------|-------------|
| **Activity API** | Records kernel launch timestamps to build the hotspot table | Tracing mode |
| **Callback API** | Intercepts `cuLaunchKernel` calls to detect the target kernel and capture the CUDA context | Always |
| **Profiler API** | Collects hardware performance counters (AutoRange + KernelReplay) | Profiling mode |

Two threads coordinate the work:

- A **state machine thread** manages the trace timer, hotspot analysis, profiler setup, and result evaluation.
- The **application's own threads** (via CUPTI callbacks) handle the profiling session start/stop, since these calls require an active CUDA context.

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

## How it works

### The trace → profile cycle

Once the library is injected, it runs through these phases repeatedly:

**Phase 1 — Trace.** The CUPTI Activity API records every kernel launch on the GPU. For each kernel, the profiler accumulates a running total of launch count and GPU execution time (nanoseconds). This runs for a configurable window (default 10 seconds) with very low overhead (~1-5%).

**Phase 2 — Select.** The state machine thread wakes up, flushes pending CUPTI records, and scans the stats table. The kernel with the highest cumulative GPU time is chosen as the profiling target. A hotspot CSV is written to disk.

**Phase 3 — Profile.** Activity tracing is paused. The profiler arms itself and waits for the next launch of the target kernel. When the application calls `cuLaunchKernel` for that kernel, the CUPTI Profiler API takes over:

- CUPTI saves the kernel's GPU input state (global memory, arguments, grid configuration).
- The kernel is **replayed multiple times**, once per hardware counter pass. Each pass programs a different group of performance counter registers and collects their values. Between passes, CUPTI restores the saved input state so every pass sees identical data.
- On the final pass, the output is kept. From the application's perspective, the kernel ran once and produced correct results — it just took longer.

This replay mechanism is necessary because GPUs have a limited number of counter registers that cannot all be read simultaneously. CUPTI's KernelReplay mode handles the scheduling transparently.

**Phase 4 — Evaluate.** The raw counter values are decoded into the requested high-level metrics (e.g. "average active cycles per SM") using the CUPTI Profiler Host API. Results are written to a JSON file and printed to stderr.

**Phase 5 — Reset.** Kernel stats are cleared, activity tracing is re-enabled, and the cycle begins again from Phase 1.

On process exit, an `atexit` handler stops the state machine thread, flushes any remaining activity records, and writes a final CSV snapshot.

### Normal execution vs profiled execution

During tracing, the application runs at near-normal speed — CUPTI only adds timestamps to kernel records. During profiling, only the single target kernel is affected:

| | Normal | Tracing mode | Profiling mode |
|---|---|---|---|
| **Kernel behavior** | Runs once | Runs once (timestamps added) | Replayed N times (once per counter pass) |
| **Performance impact** | Baseline | ~1-5% overhead | N x slowdown for the profiled kernel only |
| **GPU memory state** | Unmodified | Unmodified | Input saved/restored between passes |
| **Output correctness** | Correct | Correct | Correct (final pass output kept) |
| **Scope** | — | All kernels | Single target kernel per cycle |

The profiled kernel's `cuLaunchKernel` call blocks for longer than usual, but all other kernels and the rest of the application are unaffected. The application sees correct output because only the final replay pass's writes are preserved.

### Why multi-pass replay is necessary

GPU streaming multiprocessors (SMs) have a fixed number of performance monitoring registers. Each register can count one hardware event at a time (e.g. "FMA instructions executed" or "L2 cache misses"). Different metrics need different events, and many events conflict — they require the same physical register.

When you request 5 metrics, they decompose into 10+ raw counters that may need 2-3 separate passes. NVPW figures out the minimum pass schedule and CUPTI's KernelReplay executes it automatically.

## Output files

| File | Description |
|------|-------------|
| `kernel_hotspots_global.csv` | Hotspot table snapshot from each tracing phase |
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

These are the five metrics collected by default. Together they answer the first-order question for any GPU kernel: **what is the bottleneck?**

#### `sm__cycles_elapsed.avg`

**What it is:** Total GPU clock cycles from kernel start to end, averaged across all SMs.

**What it tells you:** The wall-clock duration of the kernel in GPU cycles. This is the denominator for utilization calculations — compare it against `sm__cycles_active.avg` to see what fraction of elapsed time the SMs were actually busy.

#### `sm__cycles_active.avg`

**What it is:** GPU clock cycles during which the SM had at least one active warp, averaged across all SMs.

**What it tells you:** How much of the kernel's time the SMs spent doing useful work vs sitting idle.

**How to use it:** Compute the SM active ratio:

```
active_ratio = sm__cycles_active.avg / sm__cycles_elapsed.avg
```

- Close to 1.0 → SMs are busy the entire time. Good utilization.
- Much less than 1.0 → SMs are frequently idle (memory stalls, sync barriers, insufficient parallelism).

#### `sm__warps_active.avg`

**What it is:** Average number of resident warps (groups of 32 threads) per SM, averaged across all SMs and cycles.

**What it tells you:** This measures **occupancy** — how well the kernel fills the SM's warp schedulers. Modern GPUs support 32–64 concurrent warps per SM.

**How to use it:** Divide by max warps-per-SM for occupancy percentage. Low occupancy (often caused by high register/shared memory usage or small grids) limits the GPU's ability to hide memory latency through warp switching.

#### `dram__bytes_read.sum`

**What it is:** Total bytes read from device memory (DRAM/HBM) across the entire GPU.

**What it tells you:** How much data the kernel fetched from global memory. Compare against peak memory bandwidth to assess memory-boundedness:

```
read_throughput_GB_s = dram__bytes_read.sum / kernel_duration_ns
```

#### `dram__bytes_write.sum`

**What it is:** Total bytes written to device memory across the entire GPU.

**How to use it:** Add reads + writes for total memory bandwidth utilization:

```
total_bandwidth_GB_s = (dram__bytes_read.sum + dram__bytes_write.sum) / kernel_duration_ns
```

### Putting the metrics together

| Observation | Diagnosis |
|-------------|-----------|
| High `active / elapsed`, low DRAM bytes | **Compute-bound** — SMs are busy doing math, memory is not the bottleneck |
| Low `active / elapsed`, high DRAM bytes | **Memory-bound** — SMs are stalled waiting for data from DRAM |
| Low `active / elapsed`, low DRAM bytes | **Latency-bound** — stalled on synchronization, small grid, or L2 misses not reaching DRAM |
| Low `warps_active` | **Low occupancy** — not enough concurrent warps to hide latency |
| High `warps_active`, low `active` | **Memory-latency-bound** — many warps are resident but all stalled on memory |

## Build

```bash
make                # builds libauto_profiler.so
make clean          # removes .o and .so files
make print-config   # shows CUDA paths
```

Requires CUDA 12.4+ with CUPTI, NVPW (nvperf_host, nvperf_target). Edit `CUDA_HOME` in the Makefile if your CUDA install is not at `/usr/local/cuda`.

From the parent `cupti/` directory, additional make targets are available:

```bash
make profiler       # build libauto_profiler.so
make llm-profile    # build + run LLM app with profiler injected (requires sudo)
make plot           # generate all plots (hotspots, overhead, profiling cycles)
```

### Usage

#### With the LLM workload

```bash
cd cupti/
make llm-profile
```

This builds the profiler and runs the LLM app with injection. `sudo` is required because CUPTI's profiling API needs elevated privileges for hardware counter access.

#### With any CUDA application

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

Default metrics: `sm__cycles_elapsed.avg`, `sm__cycles_active.avg`, `sm__warps_active.avg`, `dram__bytes_read.sum`, `dram__bytes_write.sum`

## Customizing metrics

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

---

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

2. **`getConfigImage()`** — Generates the *config image*, a binary blob that tells CUPTI which hardware counters to program and how to schedule them across passes.

3. **`getCounterDataPrefixImage()`** — Generates the *counter data prefix*, a template that sizes the results buffer.

4. **`createCounterDataImage()`** — Allocates the *counter data image* (where CUPTI writes raw counter values) and its scratch buffer. Called once.

5. **`reinitCounterDataImage()`** — Clears stale counter data between profiling cycles. Reuses the same buffer.

6. **`evaluateMetrics()`** — Decodes raw counter values into human-readable metric doubles using the CUPTI Profiler Host API.

7. **`initializeProfilerContext()`** — One-time setup: initializes CUPTI Profiler + NVPW, discovers the chip name, checks device support, queries counter availability, then calls (2)–(4) above.

8. **`beginProfilingSession()` / `endProfilingSession()`** — Start and stop a CUPTI profiling session with AutoRange + KernelReplay mode.

### tracing.h / tracing.cpp

The tracing-mode half of the profiler, using the CUPTI Activity API (same mechanism as the standalone tracer):

- **`bufferRequested()` / `bufferCompleted()`** — CUPTI buffer lifecycle callbacks. Allocate 32 KB heap buffers, iterate packed activity records, dispatch `CONCURRENT_KERNEL` records to the stats table.
- **`writeGlobalCsv()`** — Snapshots the kernel stats map under lock and writes `kernel_hotspots_global.csv`.
- **`findHottestKernel()`** — Scans the stats map and returns the kernel with the highest `total_ns`.
- **`writeProfilingJson()`** — Writes one profiling cycle's results to `profile_cycle_N.json`.

### auto_profiler.cpp

The main entry point and orchestration logic:

- **Global state definitions** — the single copy of all `extern` variables declared in `profiler_common.h`.
- **`callbackHandler()`** — CUPTI callback registered for context creation (captures the CUDA context) and kernel launches (manages profiling around the target kernel using `compare_exchange_strong` for thread safety).
- **`stateMachineLoop()`** — background thread that cycles through: sleep → flush → hotspot analysis → profiler setup → wait for callback → evaluate → write → reset.
- **`finalizeProfiler()`** — `atexit` handler that stops the state machine, flushes records, and writes final output.
- **`InitializeInjection()`** — called by CUDA runtime via `dlsym`. Parses env vars, registers activity + callback APIs, starts the state machine thread.

## Implementation notes

- **Thread safety**: A single `std::mutex` protects the kernel stats map. Mode transitions use `std::atomic<Mode>` with `compare_exchange_strong` and a `thread_local` flag to prevent races when multiple application threads launch kernels concurrently.
- **Graceful fallback**: If profiler initialization fails (unsupported GPU, missing privileges, NVPW errors), the library continues in tracing-only mode rather than crashing the target application.
- **Activity/Profiler coexistence**: CUPTI activity recording is disabled during profiling to avoid interference with hardware counter collection, then re-enabled afterward.
- **Counter data reuse**: The config image, counter data prefix, and scratch buffers are created once and reused. Only the counter data image itself is re-initialized between cycles.
- **Profiling timeout**: If the target kernel does not launch within 30 seconds of entering `TRANSITION_TO_PROFILING`, the profiler reverts to tracing.
