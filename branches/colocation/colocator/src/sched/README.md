# src/sched/ — scheduler core (`libsched.so`)

## Files

- `policy.h` — the `Policy` interface (`pick_next(heads, state) -> client or
  -1`, where -1 = issue nothing this round) and the implemented policies.
  This is the extension point: an Orion/REEF-style policy is a new subclass
  consulting `SchedState` without touching the mechanism. Selected via
  `COLOCATOR_POLICY` (run_demo: `--policy`):
  - **`fcfs`** (**default** — "throttle none") — earliest `t_intercept`
    across queue heads; never holds anything; uses none of the execution
    state.
  - **`throttle`** (`ThrottledFcfsPolicy`, cap via
    `COLOCATOR_THROTTLE` / `--throttle`, default 2) — FCFS, but a client whose stream already has
    `cap` or more unfinished ops (from `SchedState.in_flight`, the event
    cursor) gets its head **held** until completions arrive, so in-flight
    never exceeds `cap`. Ops that add no stream work (malloc/free/streamSync)
    are always eligible. Measured (30-iter demo): max in-flight exactly 3
    (vs 11 under fcfs) with no latency/throughput cost at `prio -5,0`; at
    equal priority it changes nothing — the latency client's problem there
    is the SGEMM monopolizing SMs *after* issue, which no in-flight cap on
    the issuing side can fix.
- `scheduler.cpp` — everything else:
  - `col_setup(n, priorities)` — creates one `cudaStreamCreateWithPriority`
    non-blocking stream per client (0 = default; more negative = higher,
    clamped to the device range) and resets state. With
    **`COLOCATOR_SM_LIMIT=<pct>`** (run_demo `--sm-limit`), it additionally
    carves a **green-context** SM partition (`cuDevSmResourceSplitByCount` →
    `cuGreenCtxCreate`, e.g. 50% → 88 of 170 SMs on the 5090) and gives every
    client a second stream inside it. `execute()` then routes cuBLAS ops to
    the capped stream (`route_stream`), inserting a `cudaEventRecord` +
    `cudaStreamWaitEvent` pair at every stream *switch* so the client's two
    streams stay **linear** — program order is preserved exactly, which also
    keeps the single per-client completion-event ring valid (completions
    arrive in issue order). Order-only ops (event records, redirected stream
    syncs) follow the last-used stream; drains/frees cover both streams.
    Verified on sm_120: runtime-API and cuBLAS launches on green-context
    streams execute correctly with the SM census capped (88 vs 170).
  - `col_run()` — busy-wait loop: poll pending syncs → peek every client
    queue head → `policy->pick_next` → `execute()` → pop. Called from a
    dedicated Python thread; ctypes releases the GIL.
  - `execute()` — replays a `FuncRecord` via the real CUDA call with the
    stream argument **substituted** by the client's colocator stream; stamps
    `t_issue`/`t_returned` into the issue log; advances `rec->state`, which
    unblocks the spinning client.
  - `col_stop` / `col_issued_count` / `col_dump_issue_log(path)` (CSV).

## Sync-op handling (why poll, not block)

Sync-semantics records (`cudaStreamSynchronize`, sync `cudaMemcpy`/`Memset`)
are NOT completed by calling `cudaStreamSynchronize` inline — that would
stall the scheduler loop and every other client for the whole drain
(measured: one client's 21 ms sync inflated the other's p95 from 0.5 ms to
43 ms). Instead the async part is issued, the record parks in
`pending_sync[client]`, and the loop polls `cudaStreamQuery` each iteration,
flipping the record to `DONE` when the stream drains. At most one pending
sync per client can exist because that client is blocked on it.
`cudaMalloc`/`cudaFree` stay host-synchronous (rare).

## Live queue + GPU state (Phase 6a)

Two live inputs let policies know more than "drained yes/no":

1. **Exact in-flight counts (Orion-style events).** `execute()` records one
   `cudaEvent` (from a pre-allocated 1024-deep ring per client) after every op
   it enqueues on a stream. Streams complete in order, so the loop advances a
   cursor with non-blocking `cudaEventQuery` (rate-limited to ~30 µs — each
   query is a runtime call and, with the observer on, a CUPTI record) and
   always knows **exactly how many issued ops each client still has pending**.
   Published to policies as `SchedState.in_flight[client]`; exposed to ctypes
   as `col_in_flight()` / `col_max_in_flight()`. Measured: max in-flight = 11
   for the latency client (9 kernels + 2 copies = one lockstep iteration).
2. **Measured GPU truth (observer live feed, optional).** If libobserver.so is
   loaded, `col_setup` connects via `dlsym` (`obs_live_enable/track/get`) and
   can poll, per colocator stream: CUPTI-measured completed kernel/copy counts
   and summed *real* kernel busy time. CUPTI only reports a kernel after it
   completes, so this is a completion feed with lag ≤ the 5 ms flush period
   (occasionally a couple of tail records arrive later) — use it to calibrate
   duration models online, not as an is-it-running-now oracle. Without the
   observer, the scheduler runs events-only.

What remains unknowable live: which blocks are on the SMs *right now* — events
bound it ("issued, not complete"), CUPTI confirms it just after the fact.

## Scheduler's own CUDA calls

The scheduler thread is not a registered client, so its replayed calls go
through libcolocator's interposers on the passthrough path — no recursion,
and they show up in the exit summary as passthrough counts.

## Issue log columns

`client, op_id, kind, t_intercept_ns, t_issue_ns, t_returned_ns, func,
grid_x..z, block_x..z, bytes, stream` — timestamps are CLOCK_MONOTONIC_RAW;
the Phase 4 observer joins GPU-side timestamps onto these rows.
