# Colocation VRAM-adjust prototype — design

Date: 2026-11-05. Successor of `branches/vram-limit/reports/report_colocation_ft.md` §2 ("the
view"): the decode node's VRAM has an elastic region that follows the request rate, and a
preemptable fine-tuning tenant lives in whatever that region leaves free. The old report could
only propose the resize (vLLM 0.15 had no runtime budget API); `kvcached_validation/REPORT.md`
established that kvcached main on vLLM 0.30 resizes a running engine's KV budget live at no
measurable cost, with vLLM's native DRAM offload. This document designs the loop around it.

## 1. Goal

One GPU (GPU 0, RTX 5090, 31.35 GiB usable) serves a Llama-3-8B decode node and, beside it, a
Qwen2.5-0.5B LoRA fine-tune. A VRAM scheduler watches the request rate; when it rises the decode
node's KV budget grows and the tenant shrinks (in steps, down to nothing); when it falls the
budget shrinks and the tenant grows back. Decode must keep its SLO-relevant metrics (TPOT,
scheduler queueing, throughput, zero failures) at the decode-alone level; the tenant should make
as much progress as the leftover VRAM and idle GPU windows allow, and never lose more than one
snapshot interval of work.

## 2. Platform (all validated earlier on this branch)

| piece | choice | why |
|---|---|---|
| decode node | vLLM 0.30.0 + kvcached main (`9106e0f`) + `--kv-offloading-size 28 --kv-offloading-backend native`, **prefix caching off** | the "decode-only emulation" of `kvcached_validation/REPORT.md` §7: GPU holds only in-flight KV, every prefix comes from the DRAM store (stands in for the prefill GPU); live budget change = static restart at every budget |
| budget knob | `kvctl limit <ipc> <bytes>` (visible in ~1.3 s; pages released as requests finish; new admissions wait rather than preempt) | measured in Experiment A |
| tenant | Qwen2.5-0.5B, LoRA r=16 on q/k/v/o, bf16, batch 8×512 tokens per optimizer step, AdamW + cosine schedule, synthetic deterministic batches | the trainer of the colocation-ft study, in a 1-step-per-budget-check loop |
| compute sharing | MPS daemon on, idle-window gate: engine publishes its busy window to `/dev/shm`, the trainer issues each transformer block's kernels only while decode is idle, with a 1-block CUDA-event fence (gate v1.1, the frontier scheme of the colocation-ft study: TTFT at 1×, TPOT +41 %, no knob) | best inference protection measured; MPS lets kernels co-reside instead of time-slicing |
| reference workload | 32 sessions × 6144-token prefixes + 128 new tokens → 128 out, seeded, now with a **piecewise-constant RPS schedule** | the same request stream family as every earlier study |

Memory arithmetic that drives the VRAM scheduler (measured): engine fixed footprint 17.7 GiB
(weights 15.0 + runtime 2.7, NVML at idle with kvcached); KV per request = 6272 tokens × 128 KiB
= 0.766 GiB; mean residence R ≈ 1.9 s at 2 QPS (retrieve 0.77 GiB + 128-token prefill + 128
decode steps). In-flight count N = λ·R; its Poisson p95 ≈ N + 2√N + 1.

| RPS λ | N | p95 | KV need (p95 × 0.766) | lendable = 31.35 − 17.7 − 1.0 safety − KV |
|---|---|---|---|---|
| 0.5 | 1.0 | 4 | 3.1 GiB | 9.6 GiB |
| 1 | 1.9 | 5.7 | 4.4 | 8.3 |
| 2 | 3.8 | 8.7 | 6.7 | 6.0 |
| 3 | 5.7 | 11.5 | 8.8 | 3.9 |
| 4 | 7.6 | 14.1 | 10.8 | 1.9 |

## 3. Components

```
                 /metrics (1 s)            kvctl limit (bytes)
  ┌──────────┐ ───────────────► ┌────────────┐ ───────────────► ┌──────────────────────┐
  │ loadgen  │  requests        │ VRAM scheduler │                  │ vLLM + kvcached      │
  │ RPS      │ ───────────────► │ rate est.  │  /dev/shm/       │ in-flight KV elastic │
  │ schedule │                  │ KV policy  │  coloc_ft_budget │ busy-window page     │
  └──────────┘                  │ FT budget  │ ───────┐         └──────────┬───────────┘
                                └────────────┘        ▼                    │ busy flag
                                                 ┌──────────────┐ ◄────────┘
                                                 │ FT worker    │  gate per block
                                                 │ tier 0..4    │  snapshot → DRAM
                                                 └──────────────┘
```

### 3.1 Busy-signal publisher (`hp_patch/sitecustomize.py`, ported to vLLM 0.30)

Same mechanism as the old study: a `sitecustomize` loaded via `PYTHONPATH` into the engine
wraps the model runner's step. On 0.30 with Model Runner V2 and async scheduling the GPU work
of one step spans `GPUModelRunner.execute_model` (launch) to `sample_tokens` (which syncs on
the sampled ids), so the page's busy flag is raised at `execute_model` entry and cleared at
`sample_tokens` exit. Page layout unchanged (`/dev/shm/coloc_hp_busy`: magic, heartbeat_ns,
busy, last_step_end_ns, steps). The offload connector's loads run inside the step on 0.30
(they are synchronous in the worker), so the window covers them — the gap the old study
identified as the residual TPOT cost.

### 3.2 Request-state tracking without touching vLLM: a router (`router.py`)

Nothing of the scheduler lives inside vLLM. A thin HTTP router (aiohttp, streaming
pass-through, ~1 ms) sits in front of the engine; every request passes through it, so the
VRAM scheduler knows, in its own process, exactly what the engine holds:

- on arrival: prompt length (token ids are counted directly; text prompts are tokenized with
  the model's tokenizer) and `max_tokens` → the request's final block count
  ⌈(prompt + max_tokens) / 16⌉ and its admission need ⌈prompt / 16⌉;
- while streaming: tokens generated so far (counted from the SSE chunks' `token_ids`), so the
  blocks it holds now are ⌈(prompt + generated) / 16⌉ and the growth left is known;
- on first token: the request has been admitted (it moved from *waiting* to *running*);
  before that it is waiting (or being loaded/prefilled, which for the VRAM scheduler counts as
  admitted-and-growing, and is cross-checked against vLLM's `num_requests_running` gauge);
- on completion / error: removed.

The router publishes `{running, waiting, blocks_in_use, blocks_running_final,
blocks_waiting_admit, blocks_waiting_final, ts}` to the VRAM scheduler (shared in-process, it is
the same program) — the same fields the KV-slots model needs (a request is admitted when its
prompt's blocks are free and takes one more block per 16 generated tokens), computed from
what the client sent rather than from engine internals. vLLM's `/metrics` gauges
(`num_requests_running/waiting`, `kv_cache_usage_perc`) and kvcached's segment (mapped bytes)
are read alongside as the physical cross-check. This works for any workload, not only the
reference one; for the reference workload every request is 400 blocks final, so the router's
numbers reduce to (running + waiting) × 400.

What still runs inside the engine process is **observation only**, loaded via `PYTHONPATH`
`sitecustomize` without modifying any vLLM file: the busy-window page the gate needs (§3.1)
and, optionally, the connector transfer tracer (§3.6). Neither makes a scheduling decision;
both can be switched off (scheme D runs without them).

### 3.3 VRAM scheduler (`controller.py`) — inference first, engine-state driven

Tick every **100 ms** (the router's view is updated per streamed chunk; kvcached applies a
limit within ~100 ms of its segment changing; `kvctl` itself takes 1.3 s, so the VRAM scheduler
calls kvcached's Python API `set_instance_memory_limit` directly, ~ms).

1. **Demand** `D = blocks_running_final + blocks_waiting_final` × 2 MiB (block bytes for
   Llama-3-8B), from the router's request table — the whole admitted and queued set at its final size, so nothing already in
   the engine can ever be blocked by the tenant. Plus a **look-ahead** for arrivals during the
   tenant's release time: `A = λ̂ × T_react × final_blocks_per_request`, with λ̂ the arrivals
   over the last 5 s (from Δ`request_success_total` + Δ(running+waiting)) and `T_react` = the
   measured time for the tenant to drop one tier (calibrated, ~0.5–2 s; starts at 2 s).
2. **Limit**: `need = D + A + 0.5 GiB`, rounded up to 0.5 GiB, clamped to
   [2 GiB, 12.65 GiB]. **Up** (need > limit): act *now*: publish the smaller tenant budget,
   and raise the limit in the same tick — the engine does not wait for the tenant; if the
   tenant has not released yet kvcached's growth fails to map pages for at most one tier's
   release time and vLLM queues the admission (never fails), which is the designed order of
   priorities. **Down** (need < limit − 1 GiB): only after it has held for 5 s, and the new
   limit is still ≥ `blocks_in_use` + the next queued request, so no running request is
   touched. 
3. **Tenant budget** = 31.35 − 17.7 − 1.0 − limit (GiB), published as `{seq, ts, bytes, limit,
   D, A, lambda}` to `/dev/shm/coloc_ft_budget`; on the way down it is raised only once
   kvcached's used bytes are under the new limit.
4. One CSV row per tick: all of the above plus kvcached used/limit, running/waiting, queue
   wait, the tenant's acknowledged tier and footprint.

The two numbers to report about the VRAM scheduler itself: reaction latency (RPS step → limit
raised → tenant released → kvcached used rises) and how often the engine queued *because of*
the tenant (admissions delayed while `blocks_in_use + waiting_admit > limit` and the tenant
still held memory).

### 3.4 Fine-tune worker (`ft_worker.py`)

One optimizer step per loop iteration; before each step it reads the budget file and chooses
the highest tier whose calibrated peak fits (calibration: the worker measures the peak of
each tier on its first steps and writes `tiers.json`; defaults from a probe run). The torch
allocator cap (`set_per_process_memory_fraction`) is re-applied to the budget on every change
so an estimation error OOMs the tenant, never the engine; an OOM drops one tier and retries.

| tier | mechanism | state on GPU | expected peak (to be calibrated) |
|---|---|---|---|
| 0 | full batch 8×512, forward + backward, no offload | base weights 1.0 GiB bf16, LoRA + Adam ≈ 0.1, activations ≈ 4–5 (logits 8×512×152k fp32 = 2.4 GiB alone) | ~6–7 GiB |
| 1 | **micro-batching**: the same 8×512 tokens as k ∈ {2, 4, 8} micro-batches with gradient accumulation (same optimizer step); k chosen so activations fit | weights + LoRA + activations/k | ~3.5 (k=2), ~2.2 (k=4), ~1.6 (k=8) |
| 2 | **activation offload**: `torch.autograd.graph.saved_tensors_hooks` pack → pinned DRAM on a copy stream, unpack → GPU at backward; on top of k=8 | weights + LoRA + one layer's activations | ~1.3 GiB |
| 3 | **weight offload**: frozen base weights of the last n layers streamed from pinned DRAM (the `LayerStreamer`/`OffloadManager` of the finetune study, depth-1 prefetch, gradient checkpointing) on top of tier 2; n grows with the deficit | LoRA + Adam + ≤ 2 staged layers | ~0.6 GiB (n = 20 of 24) |
| 4 | **pause**: snapshot, free everything (`model.cpu()`, empty cache); wake when the budget allows tier 3 | CUDA context only (~0.4 GiB) | — |

Tier 1 has a second trigger, the "collision" rule the task asks for: when the engine's busy
duty cycle over the last second exceeds 60 % (a large inference batch is running most of the
time), the worker splits the batch into k = 4 micro-batches even if memory allows tier 0, so
that each gated block of work stays short and the gate's 1-block fence keeps the backlog
small; with micro-batches the trainer also yields between them.

**Snapshots.** LoRA weights + optimizer state + LR-scheduler state + step counter + RNG state
are copied to pinned DRAM every N = 10 steps and before every tier change and on SIGTERM
(`torch.save` of the state dict into `/dev/shm/coloc_ft_snapshot.pt`, ~30 MB, < 50 ms). On
start the worker resumes from the snapshot if present. A kill therefore costs at most 10
steps. (Base weights are read-only and reloaded from the HF cache.)

**Gate.** As in the old trainer: forward-pre and backward-pre hooks on every decoder layer
spin on the busy flag (0.2 ms polls), with the 1-block event fence and the heartbeat-staleness
escape. The worker also records how much of each step was spent gated.

**Trace** (`ft.csv`, one row per optimizer step): ts, step, tier, k, n_offload, dt, tok/s,
loss, peak GiB, gated ms, budget bytes.

### 3.5 Load generator

`kvcached_validation/scripts/loadgen.py --mode prefix` gains `--rps-schedule
"t0:r0,t1:r1,..."` (piecewise constant, seconds from the end of warm-up). Seeded as before.

### 3.6 Host↔GPU transfer tracing (`xfer_trace`)

Every byte that crosses PCIe is recorded with who moved it, so the data-movement behaviour
and the host-bandwidth contention can be studied:

| source | how | record |
|---|---|---|
| inference connector (counts as *inf*), patch-free variant | inferred by the router: with prefix caching off every admission loads the request's prompt KV (⌈prompt/16⌉ × 2 MiB) from the DRAM tier, starting at admission (running-count rise / `start_load_kv` is inside the step) and finishing before the first token; every completion stores the request's new blocks. Gives bytes and a start/end bracket from the client's timeline, no engine code |
| inference connector, exact variant (optional) | in the engine process the `sitecustomize` wraps `SingleDirectionOffloadingHandler.transfer_async` / `get_finished` of vLLM's native CPU offload: each transfer already carries CUDA start/end events with timing enabled, and its copy plan carries `total_bytes`; direction is the handler's (GPU→CPU store, CPU→GPU load) | job_id, dir, bytes, t_submit (wall), t_done (wall), gpu_ms (event elapsed), stream |
| tenant (*ft*) | the worker's own copies — activation pack/unpack, weight streaming, snapshots — each timed with CUDA events on their stream | tag (act_out / act_in / weight_in / snapshot / resume), bytes, t_start, t_end, gpu_ms, tier |
| whole GPU (ground truth) | NVML `nvmlDeviceGetPcieThroughput` TX/RX (KB/s over its 20-ms window) sampled every 50 ms by the sampler | ts, tx_kbs, rx_kbs |

All records go to `results/<scheme>_xfer_{inf,ft}.csv` and `..._pcie.csv` on one wall clock
(`time.time()` in both processes; `/dev/shm` files, appended). Analysis: per-source bytes/s
over time against the NVML aggregate (does the sum match? what is the link's ceiling,
14.5 GB/s measured in the finetune study); the duration of each inf load as a function of
how many ft bytes were in flight at the same time (contention cost in ms per MiB of
competing traffic); the share of each tier's step spent waiting on copies.

### 3.7 Evaluation driver and analysis (`run_eval.sh`, `analyze_coloc.py`)

Schemes, each one continuous run against the same RPS schedule:

| scheme | engine budget | tenant | what it shows |
|---|---|---|---|
| A `decode_alone` | kvcached limit fixed at 12.65 GiB (all) | none | reference |
| B `static_split` | fixed at the 2-QPS need (6.7 GiB) | FT at a fixed tier 0 budget (6 GiB), gate on | today's answer: partition once |
| C `adjust` | VRAM scheduler | FT worker with tiers, gate on | the prototype |
| D `adjust_nogate` | VRAM scheduler | FT worker, no gate | isolates what the gate buys |

Per 1-s bin: offered RPS, λ̂, in-flight, kvcached limit/used, NVML, FT tier/footprint/tok/s,
decode TPOT (per request, binned by completion), TTFT, scheduler queue wait, delivered tok/s,
preemptions, failures. Per RPS phase (each constant segment of the schedule, skipping its
first 10 s): the same as a table, schemes side by side.

Figures: (0) transfer timeline: inf loads/stores and ft copies as bars per source with the NVML aggregate below; (1) the view as a timeline — a stacked memory map of GPU 0 over time (engine fixed,
in-flight KV used, KV limit line, FT footprint, free) with the RPS schedule above it;
(2) decode metrics over time for schemes A/B/C (TPOT, queue wait, delivered vs offered);
(3) FT progress over time per scheme with tier bands.

### 3.8 RPS schedule (≈ 11 min after warm-up)

| from (s) | RPS | case |
|---|---|---|
| 0 | 1.0 | low steady: tenant at tier 0 |
| 90 | 2.0 | moderate: tenant shrinks one tier |
| 180 | 3.0 | near the knee: tenant to tier 2–3 |
| 270 | 1.0 | fall: budget shrinks, tenant regrows |
| 360 | 4.0 | spike beyond lendable: tenant pauses, decode must stay healthy |
| 420 | 0.5 | collapse: maximal tenant |
| 480 | 2.0 → 3.0 → 2.0 → 1.0 | 30-s oscillation: hysteresis test |
| 600 | 1.0 | settle |
| 660 | end | |

## 4. What is measured and what would count as success

- Decode under scheme C within 10 % of scheme A on TPOT and delivered throughput in every phase, queue
  wait within 100 ms except during the ≤ 10 s after an RPS step, zero failures, zero
  preemptions attributable to the tenant.
- The tenant's tier follows the schedule within one VRAM scheduler tick (1 s) on the way down and
  within the hysteresis window (10 s) on the way up; no OOM in the engine ever; tenant progress
  (tokens/s integrated) above scheme B's whenever RPS ≤ 2 and never below zero progress loss at a
  kill (snapshot check: step counter continuity).
- Scheme B is expected to fail the decode SLO at 3–4 QPS (its 6.7 GiB cannot hold the in-flight
  set) and to waste the lendable 9 GiB at 0.5–1 QPS; scheme C should dominate it on both.

## 5. Risks and fallbacks

- MPS + kvcached: kvcached's VMM mappings under an MPS client have not been tried; if the engine
  fails under MPS, run without MPS (gate only, time-sliced) and say so.
- Busy window on 0.30 with async scheduling: if `execute_model`/`sample_tokens` do not bracket
  the GPU work, fall back to bracketing `execute_model` only (as the old study) and report the
  residual.
- The tenant's calibrated peaks may not fit the lendable range at 3–4 QPS; the pause tier is the
  guaranteed fallback.
- Host RAM: 28 GiB offload tier + 1 GiB pinned snapshot/offload buffers + engine RSS ≈ 36 GiB of
  60; the activation/weight offload buffers are bounded (< 3 GiB pinned).

## 6. Files

```
coloc_adjust/
  DESIGN.md            this document
  report/DECISIONS.md  prediction/outcome per run
  hp_patch/sitecustomize.py   observation only: busy-window page (+ optional connector transfer tracer); no scheduling
  router.py            streaming HTTP pass-through; request table (prompt, generated, max_tokens) → block demand
  controller.py        router demand → KV limit (kvcached API) → tenant budget, 100 ms ticks; same process as the router
  xfer_trace.py        shared writer for transfer records; the sitecustomize uses it for the connector
  ft_worker.py         tiered LoRA trainer (micro-batch, activation offload, weight offload, pause, snapshots, gate)
  layer_offload.py     LayerStreamer/OffloadManager (from the finetune study, unchanged)
  run_eval.sh          one scheme end to end (engine, MPS, VRAM scheduler, worker, load, teardown)
  analyze_coloc.py     bins, phase tables, figures
  results/, logs/
```
Environment: `kvcached_validation/.venv-main` (+ `peft`).

## 7. Noted for later (not in this prototype)

- Compute-side alternatives to the idle-window gate: CUDA green contexts to cap the decode
  node's SM share (an earlier result on this machine found a reduced SM share does not cost
  decode much), MPS active-thread caps on the tenant, the CUPTI kernel gate. The busy-window
  signal on vLLM 0.30 had to be rebuilt around a GPU completion event because Model Runner V2
  launches steps asynchronously; the first version measured launch time only (~5 % duty).
- A second baseline scheme: the VRAM scheduler adjusting the limit by load with **no tenant**
  (`adjust_notenant`), so the cost of resizing itself is separated from the cost of the tenant.
- Next compute-sharing candidate (Jiaxuan, 2026-11-05): **dynamic SM partitioning with CUDA
  green contexts** for *both* workloads — the VRAM scheduler would move SMs between the decode
  context and the tenant context with the load, the way it now moves VRAM. Feasibility notes:
  green contexts (`cuGreenCtxCreate`, CUDA ≥ 12.4) are driver-API objects; torch has no
  binding, so each process needs a small C extension that creates the green context + a stream
  on it and makes it current for the process's kernels. For the tenant that is our code; for the
  vLLM engine it means wrapping the worker's stream creation (observation-only patch again) or
  running the engine under a green context created before torch initialises. MPS active-thread
  percentages are the static version of the same lever (fixed at process start); green contexts
  are the dynamic one. Earlier finding on this machine: cutting the decode node's SM share
  does not cost it much (it is bandwidth-bound), which is what makes this promising.
