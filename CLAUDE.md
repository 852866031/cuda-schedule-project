# CLAUDE.md — colocation project

Orientation for Claude sessions working in this repo. Deeper detail lives in
the per-directory READMEs (every directory has one — keep them updated in the
same change as code, with mermaid diagrams for non-trivial data flow).

## What this project is

**Colocator**: a working prototype of the core mechanism of Orion
(EuroSys'24) — transparently intercept the CUDA calls of *unmodified* PyTorch
programs, buffer them in per-client software queues, and re-issue them on
colocator-owned CUDA streams under a pluggable scheduling policy — plus a
passive CUPTI **observer** that measures what actually ran on the GPU, an
**analysis** pipeline, and a browser **replay visualizer**. The purpose is
*experimenting with kernel colocation*: what is actually resident on the GPU,
for how long, and how two workloads interfere. Raw performance is explicitly
not the goal.

- `orion/` is the reference implementation — **read-only, never modify or
  import it**.
- Design + phased plan: [PROPOSAL.md](PROPOSAL.md). Results/summary of the
  original build: [README.md](README.md), [SUMMARY.md](SUMMARY.md).
- Machine: 2× RTX 5090 (sm_120, 170 SMs), driver 580, CUDA 12.8 at
  `/usr/local/cuda`, Nsight Systems 2024.6.
- Python: conda env **`colocator`** at `/mnt/storage/conda/envs/colocator`
  (torch 2.11 cu128). The user creates conda envs themselves — provide
  commands, don't run `conda create`. In scripts/tests use
  `/mnt/storage/conda/envs/colocator/bin/python` directly.

## Architecture (all under `colocator/`)

Three C++ shared libs (`make -C colocator` → `build/`):

1. **`src/intercept/` → libcolocator.so** — LD_PRELOAD interposition of:
   `cudaLaunchKernel`, `cudaMalloc/Free`, `cudaMemcpy(Async)`,
   `cudaMemset(Async)`, `cudaStreamSynchronize`, `cudaEventRecord` **and
   `cudaEventRecordWithFlags`** (torch's actual entry point), plus
   `cudaLaunchKernelExC` (counted only), plus — Orion-style **cuBLAS API
   interception** — `cublasGemmEx` and `cublasSgemm_v2` (torch 2.11's
   linear-layer entry points; cuBLAS's internal launches are invisible to
   LD_PRELOAD, so the whole library call is queued and the scheduler replays
   it after `cublasSetStream_v2(handle, colocator stream)`; real symbols
   resolve via `col_resolve` — dlopen NOLOAD, since RTLD_NEXT fails under
   python's RTLD_LOCAL loading; `cublasSetStream_v2` itself is deliberately
   NOT interposed). Registered client threads (by TID)
   get their calls packed into a `FuncRecord` **on the caller's stack**,
   pushed to a per-client mutex-guarded queue, and spin until the scheduler
   flips the record state (`ISSUED` for async ops, `DONE` for sync ops) —
   lockstep, queue depth ≤ 1 per client. All other threads (Python main,
   CUPTI workers, **the scheduler itself**) pass through to the real runtime
   — that passthrough is how the scheduler's replayed calls avoid recursion.
   `cudaStreamSynchronize` and event records are *redirected* to the client's
   colocator stream (the stream the client names is meaningless once its work
   is re-routed).
2. **`src/sched/` → libsched.so** — one `cudaStreamCreateWithPriority` stream
   per client (priorities configurable); with `COLOCATOR_SM_LIMIT=<pct>`
   (run_demo `--sm-limit`) also a second per-client stream inside a
   **green-context SM partition** (~pct of the SMs; 50% → 88/170 on the
   5090) — `execute()` routes cuBLAS ops there, event-chaining every stream
   switch so each client's two streams stay linear (program order and the
   per-client completion-event ring remain exact). Busy-wait loop: poll pending syncs
   (`cudaStreamQuery`, never block the loop) → advance the per-op
   completion-event cursor (Orion-style: exact in-flight counts, published as
   `SchedState.in_flight`, rate-limited ~30 µs) → peek queue heads →
   `Policy::pick_next` → `execute()` replays the record with the stream
   substituted → pop. Policies in `policy.h`: **fcfs (default — "throttle
   none")** and **throttle** (`COLOCATOR_POLICY=throttle`,
   `COLOCATOR_THROTTLE=N`, default cap 2: hold a client's head once it has ≥N
   unfinished stream ops). Issue log (t_intercept/t_issue per op) → CSV.
   Optional live GPU feed from the observer via `dlsym` (`obs_live_*`).
   Shared globals resolve at load time (libcolocator is preloaded into global
   scope) — no dlsym wiring for queues, unlike Orion.
3. **`src/observer/` → libobserver.so** — passive CUPTI: CONCURRENT_KERNEL
   (the non-serializing kind), MEMCPY, RUNTIME activity (filtered to the
   launch/memcpyAsync cbids), Linux TIDs via `cuptiSetThreadIdType(SYSTEM)`,
   clock calibration `cuptiGetTimestamp` ↔ CLOCK_MONOTONIC_RAW, CSV dump at
   teardown. Live feed: per-stream completion counters updated by CUPTI's
   worker; **periodic delivery needs the observer's own flusher thread**
   calling `cuptiActivityFlushAll(0)` (`cuptiActivityFlushPeriod` does NOT
   deliver partially-filled buffers on this CUPTI). One CUPTI client per
   process — the observer owns it; anything else must go through its API.

Supporting parts:

- **`ext/`** — `colocator_kernels` torch extension: tiled SGEMM (`matmul.cu`)
  + the six probe kernels ported from
  `~/Documents/Projects/gpu-interfere/profiler` (`probes.cu`: k_sleep,
  k_copy, k_copy_tb, k_fma32, k_fma64). Exists because `torch.matmul` →
  cuBLAS → driver-API launches that would bypass interception; extension
  kernels go through `cudaLaunchKernel`. Build:
  `cd colocator/ext && python setup.py build_ext --inplace`.
- **`demo/workload_defs/`** — **one workload per file**, exporting class
  `WORKLOAD`; `demo/workloads.py` (outside the dir) auto-discovers them.
  9 workloads: `latency`, `throughput` (application-style, H2D→compute→D2H,
  pinned result buffers — pageable D2H would silently block the scheduler),
  probes `sleep dram l2 l1 fma fp64` (one block/SM, each saturating one
  resource), and `llmdecode` (real Llama-3-8B bf16 decode: 3 timed forward
  passes generating tokens 65–67 over a 64→66-entry KV cache; the 63-token
  prefill runs once per machine and its state persists in
  `~/.cache/colocator/` — later runs load it and execute pure decode, no
  prefill kernels on the timeline; cache snapshot restored at every
  `drain()`, greedy argmax fed back GPU-side; decode GPU time is 85% cuBLAS
  GEMV —
  weight-streaming at ~90% of spec bandwidth, ~10 ms/pass). `llmdecode` is
  **colocatable via the cuBLAS API interception** (validated: colocated
  `check()` deterministic, submitted == issued, zero leaked kernels).
  Workloads whose kernels can't be captured must set `colocatable = False`
  (enforced by run_demo/run_pairs). `llmdecode.setup()` device-syncs after
  the weight load (transformers loads from passthrough worker threads) and
  stamps `marks["load_done"]`, which export.py uses to trim the ~1.3 s
  upload from replay timelines. Execution is a **depth-2 pipeline**: `run_iter()` admits an
  iteration and waits only for the one from 2 admissions ago via
  `torch.cuda.Event` (works colocated because event records are redirected);
  `drain()` at loop ends/barriers; iteration counts auto-calibrate to
  **~500 ms** (block-average measurement — per-sample min is wrong under
  jitter) unless forced. Each workload has `describe()` → HTML (summary +
  kernel geometry/sizes) recorded into results.json.
- **`demo/run_demo.py`** — one run; modes `seq | streams | colocated`; writes
  `results.json` (incl. per-client desc, iters, barrier_ns) and
  `issue_log.csv`; `--observer on` for CUPTI; re-execs itself to set
  LD_PRELOAD; each colocated client runs in its own `torch.cuda.Stream`
  context (own allocator pool — correctness requirement).
- **`demo/run_pairs.py`** — the matrix driver: one solo `seq` pass over all
  workloads (shared baseline at `runs/solo/`, fixes iteration counts
  deterministically), then a fresh process per unordered pair **including
  self-pairs** (36 runs for 8 workloads) into
  `runs/<throttle_cfg>/<a>__<b>/`, where cfg = `throttle_none` (default) or
  `throttle_<N>` (`--policy throttle --throttle N`). `--skip-solo` reuses the
  baseline across configs. Same iters across configs ⇒ **identical submitted
  work per pair across throttle settings** (verified).
- **`analysis/analyze.py`** — joins issue log ↔ CUPTI records ↔ GPU
  activity into `trace.json` with four timestamps per op. Kernel join walks
  GPU kernel records in **correlation order** (== the serial scheduler's
  call order regardless of launch API): plain launch rows take exactly one
  kernel; cuBLAS rows take 1+ (bounded by host launch-time anchors where
  RUNTIME records exist — cuBLAS driver-API kernels have none — plus a
  reserve so later rows never starve; clock offset corrected by projection
  onto the hard [t_issue, t_returned] windows, never toward guessed
  positions). Memcpys stay a strict 1:1 order join.
  (`t_intercept ≤ t_issue ≤ t_gpu_start < t_gpu_end`, one clock domain);
  `--self-check` asserts the join, ordering, per-client stream monotonicity,
  and zero leaked kernels (kernel launched by a client TID directly =
  interception bypass). `kernel_class()` maps mangled names → classes
  (sgemm/elementwise/reduce + sleep/streamcopy/blockcopy/fma32/fma64 +
  gemm/attention for LLM kernels) — extend it when adding kernels or chips
  render "?". For **single-client seq runs** with observer data, analyze.py
  instead writes an observer-only `seq/trace.json` (host launch times joined
  by correlation id where RUNTIME records exist; note cuBLAS's statically
  linked runtime emits RUNTIME records yet still bypasses LD_PRELOAD — the
  interposer's passthrough counter is the interceptability ground truth).

## The visualizer (`colocator/visualizer/`)

Replays a colocated run in the browser from `trace.json` + `results.json`.
Self-contained pages (no network, CSP-safe), dark GUI modeled on
`~/Documents/Projects/gpu-interfere/visualizer`.

- **`export_all.py`** — scans `runs/` (both `runs/<pair>/` legacy and
  `runs/<throttle_cfg>/<pair>/`, plus `runs/<name>/seq/` single-client solo
  runs → `<name>.solo.html` observer-only pages that hide the
  pipeline/provenance/concurrency panels), auto-runs analyze.py when a run
  has observer CSVs but no trace yet, writes `replays/<pair>.<cfg>.html` +
  `index.html`, and embeds a navigation map in every page. Overhead baseline:
  `runs/solo/seq/results.json`. `export.py` = single run.
- **Page anatomy** (top to bottom): compare bar (**workload A vs B selectors
  + throttle-config selector**; navigation across all exported runs; missing
  combos show the run_pairs command) → two **description boxes** (each
  client's `describe()` HTML, prio, iters) → the **pipeline animation**
  (Apps → Interceptor·queues → Scheduler (policy, round counter, decision
  ticker) → Streams issued/awaiting-GPU → Running-on-GPU with CONCURRENT
  badge → Done), with a **provenance strip** underneath stating which
  component measured what (INTERCEPTOR t_intercept / SCHEDULER t_issue /
  OBSERVER t_gpu_start/end) → controls (play, speed = trace-time per
  wall-second, skip-idle, ⇥ barrier jump) → **GPU timeline** Gantt (64 px
  label gutter, lane separators, green concurrency bands, dimmed
  setup·check·warmup region ending at the dashed barrier line, scrubber
  aligned above) → **Concurrency periods** (clickable start–end chips + total
  and %-of-busy sums; intersection of the two clients' merged kernel-busy
  intervals) → **Overhead** (per client: alone vs colocated per-iteration
  wall time, log-y when needed, p50s + ratio) → provenance legend. Chip
  color = kernel class, outline = client; tooltips show all four timestamps
  with source badges.
- **Serving**: `bash colocator/visualizer/serve.sh` (stdlib `server.py` on
  :8000 serving `replays/`, `/api/runs`, friendly page when replays are
  empty). Mac: `visualizer/mac/colocator-replay.sh` (edit `REMOTE=`) starts
  the remote server, tunnels 8000, opens the browser — same pattern as
  gpu-interfere. Pages are read per request: re-export → refresh, no restart.
- Template JS changes can be smoke-tested headlessly under node with DOM
  stubs (pattern used throughout the transcript logs: eval the page script
  with stubbed `document`, call `render()`/`drawGantt()`).

## Typical commands

```bash
# build everything C++ / extension
make -C colocator && cd colocator/ext && python setup.py build_ext --inplace

# full matrices (solo baseline + 36 pairs each) + replays
python colocator/demo/run_pairs.py                                    # throttle_none
python colocator/demo/run_pairs.py --skip-solo --policy throttle --throttle 2
python colocator/visualizer/export_all.py

# one run / one workload
python colocator/demo/run_demo.py --mode colocated --clients l2,dram --observer on --out runs/throttle_none/l2__dram
python colocator/demo/workloads.py --client fma --check

# offline analysis / plots for a run dir
python colocator/analysis/analyze.py runs/throttle_none/dram__l2
```

## Hard-won invariants and gotchas (do not relearn these)

- **FuncRecord lifetime**: records live on the client's stack; the state
  store must be the scheduler's *last* touch of a record (use-after-free
  otherwise). Sync-parked records stay alive until `poll_pending_syncs`.
- **Never block the scheduler loop**: sync ops complete via `cudaStreamQuery`
  polling; a blocking sync stalls *every* client. Same reason pageable-D2H is
  banned in workloads: `cudaMemcpyAsync` to pageable memory silently blocks
  the calling thread — which is the scheduler.
- **torch records events via `cudaEventRecordWithFlags`** — interposing only
  `cudaEventRecord` silently breaks event-based pipelining (event lands on
  the unused torch stream, completes instantly, pipeline runs away to the
  hardware launch-queue limit ~1022 grids/stream).
- **CUPTI**: use CONCURRENT_KERNEL (plain KERNEL serializes execution);
  activity records arrive only at kernel *completion*; partially-filled
  buffers are only delivered by an explicit `cuptiActivityFlushAll(0)`
  (hence the observer's flusher thread); RUNTIME records must be filtered by
  cbid or scheduler polling floods millions of junk records.
- **Allocator safety colocated**: each client in its own torch-stream context
  ⇒ own caching-allocator pool; without it, cross-client block reuse races.
- **Scheduler CUDA calls** go through the interposers' passthrough path
  (unregistered TID) — by design, don't "fix" the recursion.
- **Calibration** must measure a block of pipelined admissions (average), not
  per-sample min — admission times jitter between ~0 and a full kernel.
- **Coverage guarantee**: demo kernels all go through `cudaLaunchKernel`
  (custom SGEMM instead of cuBLAS); the analyze self-check's leak detector
  flags any kernel that bypassed interception. Keep it at zero.
- **Throttle vs pipeline**: the throttle caps *scheduler-side stream ops*;
  the workload pipeline caps *iterations* (multi-op iterations get gated
  mid-iteration under small caps — expected behavior, not a bug).
- `--priorities=-5,0` needs the `=` form (argparse eats a leading `-`).
- Iteration wall times under pipelining measure steady-state *admission*
  (≈ per-iter GPU time), not one iteration's end-to-end latency.

## State of the repo (as of 2026-08-04)

All phases of PROPOSAL.md are implemented and verified, plus everything
above, **including cuBLAS API interception** (llmdecode runs colocated:
solo seq p50 8.6 ms/pass → managed alone 8.9 ms → paired with `l2`
~10–17 ms (3-sample p50 jitters run-to-run against l2's 16 ms kernels) with
35.5 ms measured GPU overlap = 38% of its kernel-busy time; under throttle
the pair runs llmdecode at ~50 ms/pass (cap 2) and ~25 ms (cap 4) — the cap
empties the ~2,600-op hardware backlog that hides decode's per-launch
latency (launch→gpu-start p50 4.2 ms → ~0), the extreme case of the
throttle-vs-pipeline gotcha; the excess over the uncapped baseline scales
≈ 1/cap (stall count per pass ≈ 1,300 ops / cap), while l2 and the overlap
are unaffected in all configs). Current data:
`runs/solo/` + full 36-pair matrices under `runs/throttle_none/` and
`runs/throttle_2/` (identical submitted work per pair across configs; zero
check failures), plus `runs/llmdecode_solo/` (seq + observer, load-trimmed
replay), single-client managed runs `runs/<cfg>/llmdecode/` for cfg =
throttle_none/2/4/8 (alone: 8.9 ms/pass uncapped → 27.0/18.5/15.3 ms at
caps 2/4/8 ≈ ~1300/cap stalls × ~30 µs completion-detection latency each;
reachable via the compare bar's "(alone)" option; NOTE the throttle policy's
`adds_stream_work` must include the cuBLAS OpKinds — omitting them let
GEMMs bypass the cap, caught via >cap pending chips in the replay) and
`runs/<cfg>/l2__llmdecode/` for the same three cfgs, plus
`runs/smlimit_50/{llmdecode, l2__llmdecode}` (green-context SM limit:
llmdecode alone pays only ~5% for GEMVs on 88/170 SMs — bandwidth-bound
GEMV saturates DRAM from half the SMs); 82 replay pages +
index in `replays/`,
replay server typically on :8000. `transformers` 5.14.1 is installed in the
colocator env; Meta-Llama-3-8B lives in the HF cache. NOTE: run_pairs'
default matrix now includes llmdecode (colocatable) — a full matrix loads
the model once per pair and forces ~50 timed iterations from the 500 ms
budget; use `--workloads` to exclude it if that's unwanted. Headline findings so
far: two streams alone give no concurrency against an SM-saturating kernel;
stream priority (−5) restores a latency client at ~2% throughput cost;
self-pairs reproduce the resource-interference ladder (fma+fma 1.8×,
dram+dram 2.0×, l2+l2 1.8× slowdowns with near-total overlap; sleep+sleep
~free).
