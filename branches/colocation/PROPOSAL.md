# Colocator: A Minimal Orion-Style Kernel Interception & Scheduling Prototype

**Status:** Proposal
**Reference system:** Orion (EuroSys '24), source in `orion/` (read-only reference, not reused)
**Target machine:** 2× RTX 5090 (sm_120), CUDA 12.8, PyTorch 2.8 (cu128), Linux

---

## 1. Background: How Orion Does It (Summary of Code Study)

Orion achieves "intercept PyTorch kernel launches and re-submit them under its own policy"
with three cooperating pieces, all living **inside one process**:

### 1.1 Interception via `LD_PRELOAD` symbol interposition

- `src/cuda_capture/libinttemp.so` re-defines the *exact* public symbols of the CUDA
  runtime / cuDNN / cuBLAS C APIs: `cudaLaunchKernel`, `cudaMalloc`, `cudaMemcpy(Async)`,
  `cudaMemset(Async)`, `cudnnConvolutionForward`, `cudnnBackendExecute`,
  `cublasSgemm_v2`, `cublasLtMatmul`, etc. (`intercept_temp.cpp`, `intercept_cudnn.cpp`,
  `intercept_cublas.cpp`).
- The process is started with `LD_PRELOAD=libinttemp.so ...`, so the dynamic linker binds
  PyTorch's calls to these interposers instead of the real libraries.
- Each interposer looks up the *real* function lazily with `dlsym(RTLD_NEXT, "...")` and
  keeps it in a function pointer (e.g. `kernel_func` for the real `cudaLaunchKernel`).
- Interposed `cudaLaunchKernel` does **not** launch anything. It packs all launch
  arguments (`func`, `gridDim`, `blockDim`, `args`, `sharedMem`, `stream`) into a
  `kernel_record`, wraps it in a tagged-union `func_record`, and pushes it into a
  per-client software queue (`kqueues[idx]`) under a per-client mutex
  (`intercept_temp.cpp:427`).

### 1.2 Client identification & synchronization

- Clients are Python `threading.Thread`s in the same process (`benchmarking/launch_jobs.py`).
  The scheduler is told each client thread's Linux TID; `get_idx()` maps the calling
  thread's `gettid()` to a client index (`utils_interc.cpp`). Extra TID slots exist because
  PyTorch runs the backward pass on a separate autograd thread.
- After pushing a record, the client thread **spins until its queue is empty**
  (`block()`), i.e. until the scheduler has consumed the op. This keeps client and
  scheduler in lockstep (bounded queue, in-order), while still preserving *async* launch
  semantics (the scheduler pops right after issuing the kernel, not after it completes).
  Synchronous ops (`cudaMalloc`, sync `cudaMemcpy`) block a second time until executed.

### 1.3 Scheduler: poll queues, replay onto its own streams

- The scheduler is a separate C++ shared lib (`scheduler_eval.so`), driven from Python via
  `ctypes` (`scheduler_frontend.py`). At `setup()` it `dlopen`s the interception lib with
  `RTLD_GLOBAL` and wires up the shared globals (queues, mutexes, TID table, control flags)
  through `dlsym` — this is how the two libraries share state.
- It creates **one CUDA stream per client** with `cudaStreamCreateWithPriority`
  (high-priority client gets the highest priority; `utils_sched.cpp:121`) and a large pool
  of per-stream `cudaEvent`s.
- A busy-wait loop (`busy_wait_profile`, `scheduler_eval.cpp:233`) peeks the front record
  of every client queue each iteration and applies the policy. To execute a record it calls
  `schedule_kernel()` (`utils_sched.cpp:240`): a `switch` on record type that calls the
  *real* function pointer with the saved arguments, but **substitutes the stream** with the
  scheduler-owned stream for that client (for cuDNN/cuBLAS it first calls
  `cudnnSetStream`/`cublasSetStream`). After each kernel it records a `cudaEvent` so the
  policy can later query completion (`cudaEventQuery`) to know what is still in flight.
- Policies implemented on top of this identical mechanism: Orion (profile-guided SM/duration
  aware colocation), REEF, and sequential. The policy is just the decision of *which queue
  head to pop next and onto which stream* — exactly the extension point we want.
- Scheduling decisions use per-kernel **offline profiles** (CSV files with SM usage +
  duration per kernel, collected with NCU/NSys beforehand). Our prototype's FCFS policy
  does not need profiles.

**Key takeaway:** the mechanism (interpose → record → queue → replay on scheduler-owned
stream) is completely separable from the policy (Orion/REEF/FCFS). Our prototype clones the
mechanism, keeps the policy trivial and pluggable, and adds an independent observer.

---

## 2. Goals & Non-Goals

### Goals
1. **Interception**: transparently capture GPU operations issued by *unmodified* PyTorch
   code from two concurrent clients.
2. **Colocator**: per-client software queues + a scheduler thread that replays operations
   onto **two colocator-owned CUDA streams** (no priority by default; priority
   *configurable* per stream).
3. **Pluggable policy**: ship a dummy **FCFS** policy behind a minimal policy interface
   (`pick_next(queue_heads, state) -> client_id | none`), so Orion/REEF-style logic can be
   dropped in later.
4. **Observer**: independently record, for every intercepted operation:
   - `t_intercept` — when the client called the CUDA API,
   - `t_issue` — when the colocator actually launched it on a stream,
   - `t_gpu_start` / `t_gpu_end` — when it actually ran on the GPU,
   and correlate all four per kernel — **without perturbing the colocator's behavior**.
5. **Demo + evaluation**: two PyTorch programs running concurrently through the colocator;
   quantify launch delay, achieved overlap, and interference.

### Non-Goals (v1)
- No profile-guided scheduling, no SM-occupancy modeling (Orion's policy).
- No multi-process clients (Orion is single-process; we keep that). No MPS/MIG.
- No full cuDNN/cuBLAS record-replay coverage (see §5 risk R1 — we start with
  `cudaLaunchKernel` + memory ops, which covers ATen's native kernels).
- Not a performance-optimized system; it is an instrumented testbed.

---

## 3. Architecture

```
 ┌────────────────────────── one Python process ──────────────────────────┐
 │                                                                        │
 │  Client A thread          Client B thread         Scheduler thread     │
 │  (PyTorch model A)        (PyTorch model B)       (C++, busy-poll)     │
 │        │ cudaLaunchKernel      │                        │              │
 │        ▼                       ▼                        │              │
 │  ┌───────────────── libcolocator.so (LD_PRELOAD) ────────────────┐     │
 │  │ interpose: cudaLaunchKernel / cudaMalloc / cudaMemcpy(Async)  │     │
 │  │            / cudaMemset(Async) / cudaFree                     │     │
 │  │ get_idx(): TID → client index (incl. autograd threads)        │     │
 │  │ stamp t_intercept, tag op_id  →  push func_record             │     │
 │  └──────────┬──────────────────────────┬────────────────────────┘      │
 │             ▼                          ▼                               │
 │      Queue[A] + mutex           Queue[B] + mutex                       │
 │             └──────────┬───────────────┘                               │
 │                        ▼                                               │
 │            Policy.pick_next()  (FCFS v1, pluggable)                    │
 │                        │  stamp t_issue, launch via real fn ptr        │
 │             ┌──────────┴──────────┐                                    │
 │             ▼                     ▼                                    │
 │        Stream A (prio pA)    Stream B (prio pB)     ← configurable     │
 │                                                                        │
 │  ┌──────────────────── Observer (CUPTI, passive) ────────────────┐     │
 │  │ Activity API: CONCURRENT_KERNEL + RUNTIME records             │     │
 │  │ correlation_id ↔ op_id map;  async buffer flush at teardown   │     │
 │  └───────────────────────────────────────────────────────────────┘     │
 └────────────────────────────────────────────────────────────────────────┘
                          │ post-run
                          ▼
            trace.json / trace.csv  →  analysis/plot_timeline.py
```

### 3.1 Component: interception library (`libcolocator.so`)

C++ shared library, `LD_PRELOAD`-ed. Mirrors Orion's design:

- Interposes (v1 set): `cudaLaunchKernel`, `cudaMalloc`, `cudaFree`, `cudaMemcpy`,
  `cudaMemcpyAsync`, `cudaMemset`, `cudaMemsetAsync`, `cudaStreamSynchronize`.
  Everything else passes through untouched (not interposed at all).
- `cudaStreamSynchronize` from a client thread is redirected: drain that client's queue,
  then synchronize the client's **colocator** stream (the stream the client passed is
  meaningless once we re-route kernels). This is what makes `.cpu()` / `.item()`
  retrieval correct. `cudaDeviceSynchronize` is device-wide and passes through safely.
- Real functions resolved once via `dlsym(RTLD_NEXT, ...)`.
- `get_idx()`: TID → client index using a registration table filled by the frontend
  (client threads call a small registration hook via ctypes before running the model;
  unknown TIDs that appear later — PyTorch autograd worker threads — are adopted into
  their client's slot, as in Orion). Threads not belonging to any client (e.g. the
  scheduler itself, CUPTI worker) get index −1 → direct passthrough. **This passthrough
  is also what makes the observer non-interfering: the observer never touches the
  interposition path.**
- Each queued `func_record` additionally carries: monotonically increasing `op_id`
  (per client), `t_intercept` (`clock_gettime(CLOCK_MONOTONIC_RAW)` — comparable with
  CUPTI timestamps via a one-time offset calibration), and the client's original stream
  (kept for reference, ignored for execution).
- Client blocking semantics: Orion-style **block-until-consumed** (spin until own queue
  empty) as the default; a `COLOCATOR_QUEUE_DEPTH=N` escape hatch to allow deeper
  pipelining later. Sync ops (`cudaMalloc`, sync `cudaMemcpy`) additionally wait for
  execution completion (scheduler sets a per-op done flag).

### 3.2 Component: colocator core (scheduler)

C++ shared library (`libsched.so`) driven from Python via ctypes (like Orion's
`scheduler_frontend.py`):

- `setup(num_clients, tids, config)`: dlopen `libcolocator.so` (`RTLD_GLOBAL`), wire
  globals via `dlsym`, create **one non-blocking stream per client** with
  `cudaStreamCreateWithPriority(stream, cudaStreamNonBlocking, prio[i])` where `prio[i]`
  comes from config (default 0 = no priority for both).
- Busy-wait loop: peek head of each queue → `policy.pick_next()` → `execute(record)`:
  - stamp `t_issue`, call real function with saved args on the **colocator stream** of
    that client, then pop the queue.
  - sync semantics: intercepted *sync* `cudaMemcpy` is executed as `cudaMemcpyAsync` on
    that client's colocator stream + `cudaStreamSynchronize` on it (avoids legacy-stream
    interaction with our non-blocking streams — cleaner than Orion, which replays the
    blocking `cudaMemcpy` directly).
- **Policy interface** (compile-time simple, one virtual class):
  ```cpp
  struct Policy {
    // heads[i] is nullptr if queue i is empty; return client to launch, or -1
    virtual int pick_next(const std::vector<func_record*>& heads,
                          const SchedState& s) = 0;
  };
  ```
  v1 ships `FcfsPolicy`: pick the head with the smallest `t_intercept` (global FCFS
  across clients). `SchedState` exposes per-client in-flight counts / last events so
  future policies (REEF/Orion-style) have what they need.
- Issue log: fixed-size preallocated ring of `{client, op_id, t_intercept, t_issue,
  kind, grid, block}` — dumped to file at teardown (no I/O on the hot path).

### 3.3 Component: observer (CUPTI, non-interfering)

A separate shared library (`libobserver.so`) that only talks to CUPTI — it never touches
queues, streams, or the scheduler:

- **CUPTI Activity API** with `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` (gives per-kernel
  GPU `start`/`end` ns, `streamId`, `correlationId`, grid/block, name) and
  `CUPTI_ACTIVITY_KIND_RUNTIME` (gives the driver-side timestamp of each
  `cudaLaunchKernel` call with the same `correlationId`).
- Correlation to colocator records: the runtime-API activity record for a launch is
  produced on the *scheduler thread* at issue time; since the scheduler issues kernels
  one at a time, `(thread = scheduler, sequence order)` maps 1:1 onto the issue log.
  Belt-and-suspenders: the scheduler also stores
  `cuptiGetTimestamp()` at issue, letting us match by nearest-timestamp.
- Buffers are large (8 MB) and handed to CUPTI asynchronously; **flush only at
  teardown** (`cuptiActivityFlushAll`) so the measurement path does no I/O and takes no
  colocator locks. CUPTI's own overhead (~sub-µs per kernel, in-driver) applies equally
  to all clients and is reported.
- Clock domain: `cuptiGetTimestamp()` sampled side-by-side with
  `CLOCK_MONOTONIC_RAW` at startup → single offset used to place `t_intercept`/`t_issue`
  and GPU timestamps on one timeline.
- Toggle: `COLOCATOR_OBSERVER=0` disables it entirely, letting us A/B the observer's own
  perturbation.
- Output: `trace.json` (one record per op: client, op_id, kernel name, all four
  timestamps, stream, grid/block) + summary CSV.

Cross-validation: one demo run under Nsight Systems (`nsys profile`) to sanity-check the
CUPTI timeline (kernel overlap visible in the nsys GUI should match our computed overlap).

### 3.4 Frontend & demo

`python/colocator.py`:
- loads `libsched.so` via ctypes, spawns N client threads (each runs a user-supplied
  `fn(barrier)`), registers TIDs, starts scheduler thread, handles warmup barriers and
  teardown — closely modeled on `launch_jobs.py` + `scheduler_frontend.py`.

`demo/run_demo.py` — two clients, run in three modes for comparison:
1. **baseline-seq**: run A then B, no colocator (LD_PRELOAD unset).
2. **baseline-streams**: A and B threads with plain `torch.cuda.Stream`s, no colocator —
   what "just use two streams" gives you.
3. **colocated**: through the colocator (FCFS, both streams priority 0), observer on.

**Demo workloads** (two clients; each does the full H2D → compute → D2H cycle with
representative kernels). PyTorch's `torch.matmul` goes to cuBLAS, whose kernels are
launched internally via the driver API and would *bypass* our `cudaLaunchKernel`
interposer (this is exactly why Orion interposes cuBLAS entry points). To keep v1
interception minimal **and** still have real matmuls, the workloads use our own tiled
shared-memory SGEMM CUDA kernel, built as a small torch C++/CUDA extension
(`ext/colocator_kernels`) — extension kernels launch through `cudaLaunchKernel`, so the
colocator sees them. Elementwise/reduction ops (`relu`, `add`, `sum`) use ATen native
kernels, which also go through `cudaLaunchKernel`.

- **Client A — "latency" client**: preloads a stack of medium weight matrices to GPU
  (H2D); per iteration: copy a small input batch H2D (pinned), run a chain of
  `matmul → relu` layers (many short/medium kernels), reduce, copy the result back D2H.
  Fixed iteration count; per-iteration latency is the metric.
- **Client B — "throughput" client**: per iteration: copy a *large* weight shard H2D
  (hundreds of MB), one big square matmul (long, SM-saturating kernel), row-sum
  reduction, copy output D2H. Few long kernels + big memcpys; achieved throughput is
  the metric.

Correctness for both is checked against a CPU float64 reference with `torch.allclose`
(loose fp32 tolerance). `demo/README.md` documents each workload: purpose, tensor
shapes, expected kernel inventory (which kernels, roughly how many per iteration), and
what "correct" means — kept up to date as workloads evolve.

### 3.5 Repository layout

```
colocation/                    # repo root
  README.md                    # how to run everything from scratch (kept current)
  PROPOSAL.md                  # this document
  orion/                       # reference only — never modified, never imported
  colocator/
    Makefile                   # builds all three shared libs into build/
    build/                     # .so outputs (gitignored)
    tools/check_env.py         # Phase 0 environment verifier
    src/
      README.md                # module map + mermaid flowchart of the data flow
      common/                  # + README: shared types
        records.h              # func_record, op tagging, shared globals decl
      intercept/               # + README: how interposition works, symbol list
        intercept.cpp          # libcolocator.so
      sched/                   # + README: loop, execute(), policy interface
        scheduler.cpp          # libsched.so
        policy.h               # Policy interface + FcfsPolicy
      observer/                # + README: CUPTI usage, correlation, clock calib
        observer.cpp           # libobserver.so
    ext/                       # + README: custom CUDA kernels for workloads
      colocator_kernels/       # tiled SGEMM torch extension (setup.py build)
    python/colocator.py        # ctypes frontend, thread launcher
    demo/
      README.md                # what each workload does, kernel inventory
      workloads.py             # client A (latency) / client B (throughput)
      smoke_test.py            # Phase 1 passthrough test
      run_demo.py              # modes: seq | streams | colocated | all
    analysis/
      analyze.py               # metrics + self-checks from trace.json
      plot_timeline.py         # per-stream Gantt + latency CDFs
```

Every directory under `colocator/` carries a `README.md` explaining what its files do
and how they relate (mermaid flowcharts for anything with non-trivial data flow); these
are updated in the same commit as the code they describe.

---

## 4. Evaluation Plan

From `trace.json`, per kernel: **queue delay** = `t_issue − t_intercept`,
**launch-to-start** = `t_gpu_start − t_issue`, **duration** = `t_gpu_end − t_gpu_start`.

Questions the demo must answer:
1. **Does interception work transparently?** Unmodified PyTorch code runs to completion
   with correct results (compare model outputs vs. baseline, `torch.allclose`).
2. **Do kernels actually run concurrently?** Compute wall-clock overlap: fraction of time
   where kernels from A and B are simultaneously resident
   (interval-union over `[t_gpu_start, t_gpu_end)` per client). Compare
   baseline-streams vs. colocated; visualize as a two-lane Gantt chart.
3. **Interference:** per-client kernel duration and end-to-end iteration time, solo vs.
   colocated (same kernel, same shapes → duration inflation = SM/memory contention).
   Also A's p50/p95/p99 iteration latency with and without B.
4. **Colocator overhead:** queue delay distribution + iteration-time delta of
   "colocated with idle B" vs. baseline solo; observer on/off delta.

---

## 5. Risks & Mitigations

- **R1 — cuBLAS/cuDNN kernels bypass `cudaLaunchKernel`** (launched internally via the
  driver API). Orion interposes those library entry points; v1 avoids the problem by
  using our own SGEMM extension kernel for matmuls and ATen native kernels otherwise,
  and *asserts* coverage: the observer sees every kernel on the GPU, so any kernel with
  no matching colocator record ⇒ leaked op ⇒ loud warning in `analyze.py --self-check`.
  A later phase can add `cublasSgemm/cublasLtMatmul` interposers Orion-style to support
  stock `torch.matmul`.
- **R2 — PyTorch sync/event APIs on the original stream** (`cudaEventRecord`,
  `cudaStreamSynchronize` on stream 0) could become no-ops w.r.t. our streams.
  Mitigated in v1: `cudaStreamSynchronize` is interposed and redirected to the client's
  colocator stream (§3.1), and `torch.cuda.synchronize()` (device-wide) is safe. If the
  caching allocator's cross-stream events show up, we interpose `cudaEventRecord` to
  redirect onto the client's colocator stream (small, known fix; Orion has analogous
  handling).
- **R3 — spin-wait burns CPU** (client spins + scheduler polls). Same as Orion; pin
  threads to cores (Orion sets affinity in `get_idx`) and accept it — this is a testbed.
- **R4 — clock-domain skew** between `CLOCK_MONOTONIC_RAW` and CUPTI timestamps.
  Calibrate offset at start and end of run; report drift; all cross-client comparisons
  use CUPTI GPU timestamps only, so skew affects only queue-delay attribution.
- **R5 — blocking `block()` deadlock** if the scheduler dies. Watchdog: client spin loops
  time out after N seconds and fall back to direct passthrough with a loud error.

---

## 6. Environment & Setup

Requirements (host): NVIDIA driver ≥ 570, CUDA toolkit 12.8 at `/usr/local/cuda`
(provides `nvcc` and CUPTI headers/libs under `extras/CUPTI` or `targets/x86_64-linux`),
`g++`, `make`, conda. Verified target: 2× RTX 5090 (sm_120), driver 580.126, nvcc 12.8.

Python environment (user creates the conda env; all Python deps live in it):

```bash
conda create -n colocator python=3.12 -y
```

```bash
conda activate colocator && pip install torch --index-url https://download.pytorch.org/whl/cu128 && pip install numpy pandas matplotlib ninja
```

The C++ side (`libcolocator.so`, `libsched.so`, `libobserver.so`) is built with the
system toolchain via `make` and does not depend on the conda env; the torch extension
(`ext/`) is built inside the conda env with `nvcc`.

---

## 7. Implementation Plan — Phases with Runnable Verification

Rules that apply to **every** phase:
- The phase is only "done" when its **verify** commands pass on the target machine.
- The root `README.md` is updated so the repo remains runnable *from scratch* by
  following it top-to-bottom; per-directory READMEs are updated in the same change as
  the code they describe.
- `orion/` is never touched.

### Phase 0 — Environment & skeleton
**Build:** repo skeleton (`colocator/` tree, Makefile stubs, root README),
`tools/check_env.py`.

**Run & verify:**
```bash
conda activate colocator && python colocator/tools/check_env.py
```
Prints and asserts: torch version + CUDA available, GPU name/SM count, `nvcc` version,
CUPTI header + library found, g++ present. Exit code 0 = phase passed.

### Phase 1 — Interception proof-of-life (passthrough mode)
**Build:** `libcolocator.so` with all v1 interposers, a `COLOCATOR_MODE=passthrough`
mode that logs each intercepted call (client TID, op kind, grid/block) and immediately
calls the real function — no queues, no scheduler. Exit-time summary report.

**Run & verify:**
```bash
make -C colocator intercept
```
```bash
conda activate colocator && COLOCATOR_MODE=passthrough LD_PRELOAD=$PWD/colocator/build/libcolocator.so python colocator/demo/smoke_test.py
```
`smoke_test.py` runs simple ATen elementwise ops and checks results (`torch.allclose`
vs CPU). Verify: (a) script prints `OK`, (b) the exit summary shows >0 intercepted
`cudaLaunchKernel` and memcpy calls, (c) running *without* `LD_PRELOAD` also prints
`OK` (interposer changes nothing functionally).

### Phase 2 — Workloads + custom SGEMM extension (standalone)
**Build:** `ext/colocator_kernels` (tiled SGEMM), `demo/workloads.py` (client A/B),
`demo/README.md`.

**Run & verify:**
```bash
conda activate colocator && python -m colocator.ext.build   # or: cd colocator/ext && python setup.py build_ext --inplace
```
```bash
conda activate colocator && python colocator/demo/workloads.py --client latency --check && python colocator/demo/workloads.py --client throughput --check
```
`--check` compares against CPU float64 reference (`allclose`, fp32 tolerance). Then
re-run both under the Phase 1 passthrough interposer and verify the summary shows the
expected kernel inventory (SGEMM + elementwise + memcpys) and **no unexplained gap**
between ops issued by torch and ops intercepted.

### Phase 3 — Colocator core (queues + FCFS + two streams)
**Build:** `libsched.so` (setup, busy-wait loop, `execute()`, FCFS policy, issue log),
`python/colocator.py` frontend (re-execs itself with `LD_PRELOAD` set so users just run
one command), `run_demo.py` modes `seq|streams|colocated`.

**Run & verify:**
```bash
make -C colocator
```
```bash
conda activate colocator && python colocator/demo/run_demo.py --mode colocated --clients latency            # single client through colocator
```
```bash
conda activate colocator && python colocator/demo/run_demo.py --mode colocated --clients latency,throughput # both clients
```
Verify: (a) both clients' `--check` outputs still pass (results identical to
standalone), (b) scheduler exit stats show ops-consumed == ops-submitted per client,
(c) process exits cleanly (no deadlock; watchdog silent), (d) `--mode seq` and
`--mode streams` also run (baselines working).

### Phase 4 — Observer (CUPTI)
**Build:** `libobserver.so`, correlation + clock calibration, `trace.json` writer,
`analysis/analyze.py --self-check`.

**Run & verify:**
```bash
make -C colocator observer
```
```bash
conda activate colocator && python colocator/demo/run_demo.py --mode colocated --observer on --out runs/p4 && python colocator/analysis/analyze.py runs/p4 --self-check
```
`--self-check` asserts: every colocator op has all four timestamps
(`t_intercept ≤ t_issue ≤ t_gpu_start < t_gpu_end`), zero leaked GPU kernels (no CUPTI
kernel record without a colocator record), monotonic per-client GPU ordering. Also run
with `--observer off` and verify wall-clock delta is small (report the number).

### Phase 5 — Demo & evaluation
**Build:** `analyze.py` metrics (§4), `plot_timeline.py` (two-lane Gantt + latency
CDFs), results writeup section in root README.

**Run & verify:**
```bash
conda activate colocator && python colocator/demo/run_demo.py --mode all --out runs/full && python colocator/analysis/analyze.py runs/full && python colocator/analysis/plot_timeline.py runs/full
```
Verify: summary table covers all three modes (per-client iteration p50/p95/p99, kernel
duration inflation, queue delay, overlap fraction); overlap fraction > 0 in `streams`
and `colocated` modes; PNG plots produced. Cross-validate once with Nsight Systems:
```bash
nsys profile -o runs/nsys_check python colocator/demo/run_demo.py --mode colocated --observer off
```
and confirm the nsys timeline's A/B overlap qualitatively matches our CUPTI-derived
Gantt.

### Phase 6 (optional / stretch)
cuBLAS interposers (stock `torch.matmul` support), stream-priority experiments
(`--stream-priority A=high,B=low`), a second policy (e.g. REEF-lite) to prove the
policy interface, deeper client queues (`COLOCATOR_QUEUE_DEPTH>1`).

---

## 8. Documentation Plan

| File | Contents | Updated when |
|---|---|---|
| `README.md` (root) | prerequisites, conda env creation, build, run-from-scratch walkthrough, current status | every phase |
| `colocator/src/README.md` | module map, mermaid flowchart: client call → interposer → queue → policy → stream → GPU, and observer tap points | any src change |
| `colocator/src/{intercept,sched,observer,common}/README.md` | what each file does, file relations, key invariants | any change in that dir |
| `colocator/ext/README.md` | SGEMM kernel design, why not cuBLAS, build notes | ext changes |
| `colocator/demo/README.md` | per-workload: purpose, shapes, kernel inventory, correctness criterion; per-mode semantics | workload/demo changes |
| `colocator/analysis/README.md` | metric definitions, trace.json schema | analysis changes |
