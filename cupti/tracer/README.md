# CUPTI Activity Tracer

An injection-based CUPTI tracer that attaches to an unmodified CUDA process at runtime. It records kernel launch frequency and duration, maintaining both a global all-time hotspot table and a rolling 5-second hotspot table. CSV outputs are written to disk automatically.

## How it works

The tracer is compiled as a shared library (`libactivity_tracer.so`). When you set `CUDA_INJECTION64_PATH` to its path, the CUDA runtime calls the library's `InitializeInjection()` entry point before any CUDA code runs. The tracer then:

1. Registers CUPTI activity callbacks for `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL`
2. For each completed kernel record, accumulates launch counts and total execution time (nanoseconds) into a global map and a 5-second sliding-window deque
3. Runs a background thread that writes the recent-5s CSV to disk every 5 seconds
4. On process exit, flushes remaining CUPTI buffers, stops the writer thread, and writes final CSV files

C++ symbol names are automatically demangled via `abi::__cxa_demangle`.

## Output

| File | Description |
|------|-------------|
| `kernel_hotspots_global.csv` | All-time stats: every kernel seen since process start |
| `kernel_hotspots_recent_5s.csv` | Rolling stats: kernels active in the most recent 5 seconds |

Both files use the columns: `kernel_name`, `launch_count`, `total_duration_ms`, `avg_duration_us`.

Top-10 kernels by duration and by launch count are also printed to stderr at exit.

## Build

```bash
make              # builds libactivity_tracer.so and example_workload
make run-example  # runs example_workload with the tracer injected
```

Requires CUDA and CUPTI. Edit `CUDA_HOME` in the Makefile if your CUDA install is not at `/usr/local/cuda`.

## Injecting into your own process

```bash
export CUDA_INJECTION64_PATH=/path/to/libactivity_tracer.so
export CUPTI_TRACE_OUTDIR=/path/to/output/dir
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
./your_cuda_program
```

## Example workload

`example_workload.cu` launches two kernels in a loop for ~3 seconds:

- `dense_fma_kernel` — compute-bound FMA loop
- `bad_gather_kernel` — memory-bound kernel with irregular gather pattern (poor cache locality)

It serves as a quick sanity check that the tracer is working correctly.

## Implementation notes

- Buffer size per CUPTI allocation: 32 KB
- Sliding window duration: 5 seconds (5 × 10⁹ ns)
- Background CSV write interval: 5 seconds
- Thread safety: a single `std::mutex` protects all shared statistics
- `g_stop_writer` and `g_initialized` are `std::atomic<bool>` for clean shutdown
