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

Activity tracing is disabled. The mode is set to `TRANSITION_TO_PROFILING`.

When the application launches the target kernel:

- **ENTER callback**: `compare_exchange_strong` atomically claims the transition. Counter data image is re-initialized. `beginProfilingSession()` calls `cuptiProfilerBeginSession` (AutoRange + KernelReplay), `SetConfig`, `EnableProfiling`.
- **The kernel executes**: CUPTI automatically replays it as many times as needed to collect all requested counters across multiple passes. This happens transparently within the single `cuLaunchKernel` call.
- **EXIT callback**: `endProfilingSession()` calls `DisableProfiling`, `UnsetConfig`, `EndSession`. The mode is set to `PROFILING_DONE` and the condition variable is signaled.

### 8. Evaluation and output

The state machine thread wakes up and:

- Calls `evaluateMetrics()` which uses the CUPTI Host API to decode raw counter data into metric doubles.
- Writes `profile_cycle_N.json` with the kernel name, trace stats, and metric values.
- Prints a summary to stderr.

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
