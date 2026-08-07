# analysis/ — trace assembly, verification, metrics, plots

## Files

- `analyze.py` — joins the colocated mode's three data sources (scheduler
  issue log ↔ CUPTI runtime records ↔ CUPTI GPU activity) into one row per
  op with **four timestamps** (all CLOCK_MONOTONIC_RAW ns):

  | field | meaning | source |
  |---|---|---|
  | `t_intercept_ns` | client called the CUDA API | interposer |
  | `t_issue_ns` | scheduler launched it on the colocator stream | scheduler |
  | `t_gpu_start_ns` | execution actually began on the GPU | CUPTI |
  | `t_gpu_end_ns` | execution finished | CUPTI |

  Writes `<run>/colocated/trace.json`, runs `--self-check` (default on):
  every op matched, timestamps ordered, per-client GPU order monotonic, zero
  kernels launched directly by client threads. Prints stage-delay stats
  (queue delay, launch→gpu-start), kernel durations by class, per-client GPU
  busy time and A/B overlap, and a cross-mode kernel-duration table
  (interference) when seq/streams dirs carry observer data.

  For a **single-client seq run** with observer data (e.g. `llmdecode`
  solo), `analyze.py` instead writes an **observer-only** trace into
  `<run>/seq/trace.json`: every kernel/memcpy CUPTI saw in the mode window
  (including cuBLAS launches the interposer can never see), `t_issue_ns` =
  host launch time joined by **correlation id** where a RUNTIME record
  exists, `t_intercept_ns` null. Caveat: "has a RUNTIME record" ≠
  "interceptable" — cuBLAS statically links the CUDA runtime, so its
  launches emit RUNTIME records yet bypass the LD_PRELOAD PLT; ground truth
  for interceptability is the interposer's own passthrough counter.

- `kernel_class()` also maps the LLM-decode kernel families: `gemm`
  (cuBLAS `nvjet`/`cutlass`/`gemv`/`gemm`), `attention` (flash/fmha), plus
  softmax→reduce and norm/cat/index/gather→elementwise. Extend it when a
  replay page renders "?" chips.

- `plot_timeline.py` — renders the trace: per-client GPU Gantt lanes
  (kernels + memcpys) and latency CDFs across modes. PNG output into the run
  directory.

## Join logic in one sentence

The scheduler launches ops one at a time from one thread, so its k-th
`cudaLaunchKernel`/`cudaMemcpyAsync` CUPTI RUNTIME record (start-time order,
scheduler TID) corresponds to the k-th kernel/memcpy row of the issue log;
the RUNTIME record's correlation id then keys the GPU activity record. See
`../src/observer/README.md`.

## Metric definitions

- **queue delay** = `t_issue − t_intercept` (time waiting in the colocator's
  software queue; scheduler overhead + policy decisions).
- **launch→gpu-start** = `t_gpu_start − t_issue` (hardware-side queueing:
  stream depth + SM availability — this is where cross-client interference
  shows up).
- **duration** = `t_gpu_end − t_gpu_start`; compared across modes (solo vs
  shared) it measures SM/memory contention while actually overlapping.
- **overlap** = intersection time of the two clients' merged kernel-busy
  interval sets; reported vs busy-union and vs the smaller client's busy time.
