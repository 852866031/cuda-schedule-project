# Colocation VRAM-adjust prototype — report

Date: 2026-11-05/06. Branch `kvcached-validation`, directory `coloc_adjust/`. Design in
[DESIGN.md](../DESIGN.md), run-by-run predictions and outcomes in [DECISIONS.md](DECISIONS.md).

## 1. What was built

One RTX 5090 serves a Llama-3-8B decode node (vLLM 0.30 + kvcached main, native DRAM offload
with a 28 GiB tier, prefix caching off, so the GPU holds only in-flight KV) and, beside it, a
Qwen2.5-0.5B LoRA fine-tune. Three pieces, none of them inside vLLM's scheduler:

- **Router/controller** (`controller.py`, one process): every request passes through a streaming
  HTTP router that knows its prompt length, `max_tokens` and tokens generated so far, so the KV
  block demand of the running set and the queue is exact (KV-slots model: ⌈prompt/16⌉ blocks at
  admission, +1 block per 16 generated tokens). Every 100 ms: `need = (running final + waiting
  final) × 2 MiB + λ̂·1 s·0.78 GiB + 0.5 GiB`; the kvcached limit grows to `need` at once and
  shrinks only after 5 s below `need − 1 GiB` and never below what the running set holds; the
  tenant budget `31.35 − 17.7 − 1.0 − limit` GiB is published to `/dev/shm`, lowered at once on
  the way up and raised only once kvcached's mapped bytes are under the new limit.
- **Tenant** (`ft_worker.py`): before each optimizer step reads the budget and picks the
  highest configuration whose calibrated need fits (drop at once, climb after 3 s): full batch
  → micro-batches k = 2/4/8 (also forced to k ≥ 4 when the engine's busy duty exceeds 60 %) →
  + activation offload to pinned DRAM → + frozen base weights of 12/20 layers streamed from DRAM
  → pause (snapshot, everything to host). LoRA weights, optimizer, scheduler, step and RNG are
  snapshotted to DRAM every 10 steps and before every tier change; the worker resumes from the
  snapshot. An idle-window gate (engine busy page, per-block hooks, 1-block fence) is on in the
  main scheme; the torch allocator cap follows the budget so a mis-estimate OOMs the tenant, never
  the engine.
- **Transfer tracing**: every host↔GPU copy is recorded with its source — the engine's native
  offload transfers (bytes and CUDA-event time from vLLM's own transfer results, via an
  observation-only `sitecustomize`) and the tenant's activation/weight/snapshot copies — plus
  NVML's PCIe TX/RX counters every 100 ms.

Compute sharing: MPS daemon (user-space), engine and tenant as MPS clients.

![system](figures/01_system_architecture.png)

*Diagram 1 — the components and what triggers what: the router feeds demand to the controller, the controller sets kvcached's limit (red, inference first) and publishes the remainder to the tenant; the observation patch only reports.*

![controller](figures/03_controller_loop.png)

*Diagram 2 — one controller tick: grow at once when demand exceeds the limit, shrink only after 5 s of slack and never below what running requests hold.*

![tiers](figures/02_tenant_tiers.png)

*Diagram 3 — the tenant's ladder and the triggers that move it: budget polls inside the step, OOM as a signal, 3-s hold before climbing, DRAM snapshots around every move.*

### Tenant configurations (calibrated solo on GPU 1, `tiers.json`)

| cfg | tier | micro-batches k | activation offload | layers streamed | grad ckpt | need GiB | step ms | tok/s |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 full batch | 1 | – | 0 | off | 15.9 | 91 | 44.8k |
| 1 | 1 | 2 | – | 0 | off | 8.7 | 106 | 38.7k |
| 2 | 1 | 4 | – | 0 | off | 5.0 | 138 | 29.6k |
| 3 | 1 | 8 | – | 0 | off | 3.2 | 210 | 19.5k |
| 4 | 2 | 8 | yes | 0 | on | 2.5 | 1045 | 3.9k |
| 5 | 3 | 8 | yes | 12 | on | 2.2 | 1294 | 3.2k |
| 6 | 3 | 8 | yes | 20 | on | 1.9 | 1452 | 2.8k |
| 7 | 4 pause | – | – | – | – | 0.4 | – | 0 |

Activation offload moves ~4.8 GB per step each way at k = 8 and costs 5× in step time; with
gradient checkpointing already on it saves little memory (the floor, ~1.9 GiB, is the
lm_head/embedding plus one micro-batch's logits), so in practice the ladder is micro-batching
down to ~3 GiB, then the slow offload tiers, then pause.

## 2. Evaluation

Reference workload (32 sessions × 6144-token prefixes, +128 new tokens → 128 out, Zipf-1.1,
seeded) with a piecewise-constant RPS schedule after a 32-request warm-up:

| from (s) | 0 | 90 | 180 | 270 | 360 | 420 | 480 | 510 | 540 | 570 | 600–660 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RPS | 1 | 2 | 3 | 1 | 4 | 0.5 | 2 | 3 | 2 | 1 | 1 |

Four schemes, each one continuous 11-minute run (`run_eval.sh`): **A** decode alone (limit 13 GiB,
no tenant); **B** static split (engine fixed at 6.69 GiB = the 2-QPS need, tenant at a fixed
6 GiB budget → cfg 2, gate on) — "partition once"; **C** the prototype (controller + tiered
tenant + gate); **D** the prototype without the gate. Per-phase numbers skip the first 10 s of
each phase. Decode is judged on TPOT, scheduler queue wait, delivered throughput and failures
(decode-only TTFT is fetch + 128-token prefill, ~160 ms); the tenant on tokens/s and its
configuration.

### 2.1 What the schemes are

| scheme | engine KV limit | tenant | compute sharing |
|---|---|---|---|
| A `decode_alone` | fixed 13 GiB | none | – |
| E `adjust_notenant` | controller | none | – (baseline for the prototype: resizing without anyone using the freed VRAM) |
| B `static_split` | fixed 6.69 GiB | fixed 6 GiB budget (cfg 2) | gate (irrelevant, see below) |
| C `adjust` | controller | tiered | idle-window gate + MPS |
| D `adjust_nogate` | controller | tiered | MPS co-run, no cap |
| F `adjust_mps20` | controller | tiered | MPS, tenant capped at 20 % of SM threads |
| G `green32` | controller | tiered | tenant on a 32-SM green context, no gate |
| H `green_dyn` | controller | tiered | tenant SM count from the arrival rate (128/96/64/32/16 at 0/0.5/1/2/≥3 QPS) |
| I `engine_green` | controller | tiered | engine steps on a 128-SM green context, tenant as in H |
| J `green_opt` | controller, smoothed shrink (10-s max) | tiered | tenant SM table 96/64/32/16/8 |
| K `green_split` | controller | tiered | engine steps on SMs 0–127, tenant on the remaining 42 SMs (static, disjoint) |

![compute sharing](figures/04_compute_sharing.png)

*Diagram 4 — the compute-sharing mechanisms tried (source `../diagrams/04_compute_sharing.drawio`); the memory loop is the same in all of them. The dot before each title is the colour of that scheme's line in Figure 1; a dashed circle marks the two schemes Figure 1 leaves out.*

![schemes](figures/arms.png)

*Figure 1 — decode latency (a), scheduler queue wait (b) and fine-tune throughput (c) against the
arrival rate, pooling the phases that share a rate (marker = mean, bar = min..max across the
repeats; `plot_arms.py`). The baseline is the engine alone with its KV limit following the load
(scheme E; scheme A with a fixed 13 GiB limit is identical to it). Schemes G–J are left out: `green32`
coincides with the MPS 20 % cap, and the dynamic green-context schemes lost their tenant in the
first phase (table in 2.3). Reading: adaptive schemes sit on the baseline in (a) and (b) at 3–4 QPS,
where the tenant is paused, and pay a tenant-speed-dependent premium at 0.5–2 QPS; the static split
pays 20–80 % everywhere and queues seconds once the load passes 2 QPS.*

![tradeoff](figures/arms_tradeoff.png)

*Figure 2 — the same runs, one point per scheme and phase: decode slowdown over the baseline against
the tenant's throughput in that phase. The adaptive schemes fall on one line regardless of mechanism
(MPS, MPS cap, disjoint green partition): about +2.3 % TPOT per 1 000 tenant tokens/s. The static
split's points are off that line because its cost is KV starvation, not the tenant's speed.*

### 2.2 Decode under each scheme (TPOT p50 in ms; scheduler queue wait in ms)

| phase | A alone | E adjust, no tenant | B static | C gate | D no gate | F MPS 20 % | G green 32 |
|---|---|---|---|---|---|---|---|
| 1 QPS (0–90 s) | 12.9 / 107 | 12.9 / 107 | 16.8 / 126 | 13.1 / 107 | 16.4 / 130 | 14.6 / 120 | 14.4 / 119 |
| 2 QPS | 15.2 / 124 | 15.2 / 124 | 21.4 / 325 | 15.3 / 127 | 18.5 / 137 | 17.0 / 132 | 16.6 / 132 |
| 3 QPS | 21.2 / 267 | 21.2 / 270 | 22.9 / 7 934 | 21.3 / 303 | 21.3 / 380 | 21.9 / 366 | 21.3 / 370 |
| 1 QPS (270 s) | 12.5 / 105 | 12.5 / 105 | 16.8 / 7 966 | 12.6 / 106 | 16.1 / 138 | 13.8 / 126 | 13.6 / 126 |
| 4 QPS | 26.0 / 1 209 | 26.0 / 1 228 | 22.9 / 13 468 | 25.0 / 2 156 | 24.1 / 3 286 | 24.4 / 3 792 | 24.1 / 3 260 |
| 0.5 QPS | 12.0 / 95 | 12.0 / 94 | 21.7 / 26 008 | 12.1 / 95 | 15.9 / 133 | 13.3 / 120 | 13.0 / 122 |
| 2 → 3 → 2 → 1 (480–600 s) | 15.5–20.2 / 122–157 | same | 20.7–23.1 / 145–4 258 | 15.6–20.2 / 110–173 | 16.6–20.5 / 135–211 | 14.2–20.8 / 128–199 | 14.0–20.4 / 116–205 |
| failures / preemptions | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

### 2.3 The tenant under each scheme (tok/s; its configuration)

| phase | B static | C gate | D no gate | F MPS 20 % | G green 32 |
|---|---|---|---|---|---|
| 1 QPS | 9.6k (cfg 2, 5 GiB) | 0.05k | 11.0k (cfg 2) | 6.7k (cfg 2) | 6.3k (cfg 2) |
| 2 QPS | 12.7k | 0 | 4.9k (cfg 3) | 3.6k (cfg 3) | 3.3k (cfg 3) |
| 3 QPS | 12.6k | 0 | 0.15k (cfg 3 → pause) | 0.1k | 0.1k |
| 1 QPS (270 s) | 9.9k | 0 | 12.7k | 6.8k | 6.4k |
| 4 QPS | 12.8k | 0 | 0 (paused) | 0 (paused) | 0 (paused) |
| 0.5 QPS | 12.0k | 0 | 16.3k (cfg 1, 8.7 GiB) | 7.7k | 7.3k |
| whole run, steps | 1 844 | 32 (21 preempted) | 1 310 (26 preempted mid-step) | 735 | 696 |

Green-context schemes (`green_ctx.py`, ctypes on libcuda; details in DECISIONS D8–D10):

| scheme | decode at 1 / 2 QPS (TPOT, queue) | tenant at 1 / 0.5 QPS | note |
|---|---|---|---|
| G `green32` (tenant on 32 SMs) | 14.4 / 16.6 ms, 119 / 132 ms | 6.3k / 7.3k tok/s | same trade as the MPS 20 % cap |
| H `green_dyn` (SM count from λ̂) | tenant died in phase 1 | 1.4k then 0 | re-creating the context on every share change leaked allocator memory (7.7 GiB reserved for a 5 GiB config) → OOM to pause |
| I `engine_green` (engine on 128 SMs + H) | **12.9–13.2 / 15.2 ms, 109 / 124 ms — equal to the baseline** | tenant died as in H | a bandwidth-bound decode node gives up a quarter of its SMs for free |
| J `green_opt` (smoothed shrink + lower SM table) | tenant died as in H | 0.3k then 0 | same leak |
| K `green_split` (engine 128 SMs, tenant the other 42, static) | 16.3 / 18.3 ms, 123 / 145 ms | 10.5k / 13.5k | tenant progress of D, decode cost of D: **disjoint SMs do not remove the interference** |


### 2.4 Reading the tables (see `../results/*_timeline.png` for the per-second view, e.g. `adjust_nogate_timeline.png`)

- **Resizing costs decode nothing.** E equals A in every phase to within noise (TPOT within 0.1 ms,
  queue wait within 3 ms): changing kvcached's limit 3 ↔ 12.6 GiB with the load is invisible
  to the engine. The limit followed the schedule (5.0 / 8.0 / 12.6 / 4.5 / 12.6 / 3.0 GiB).
- **Static partitioning fails both ways.** B's engine at 6.69 GiB queues 8 s at 3 QPS and 13 s at
  4 QPS, and the backlog is still 26 s deep when the load has dropped to 0.5 QPS; at low RPS its
  tenant holds 5 GiB while 4–8 GiB of the engine's budget sit unused. Its decode pays +30 % TPOT
  even at 1 QPS because the tenant runs flat out beside it.
- **The memory loop works as designed.** In C, D, F, G the tenant's configuration follows the
  lendable budget (cfg 2 at 1 QPS, cfg 3 at 2 QPS, pause at ≥ 3 QPS, cfg 1 at 0.5 QPS); the
  engine never fails and never preempts; mid-step preemption (the budget polled every 50 ms from
  the per-block hooks) is what removed the 10–13 preemptions seen before it existed (DECISIONS
  D6/D7). The remaining decode cost of a tenant is at the 4-QPS spike (queue 2–4 s vs 1.2 s
  alone: the tenant's pause takes ~0.2 s and the engine's admissions pile up meanwhile) and,
  above all, at low RPS, where it is **compute interference, not memory**.
- **Time-sharing is dead on this engine.** The idle-window gate (C) protects decode perfectly —
  and the tenant trains nothing, because the Model Runner V2 engine keeps the GPU busy ~100 %
  of the time while any request runs (DECISIONS D5). Un-gated co-running under MPS (D) costs
  decode +27 % TPOT at 1 QPS, +22 % at 2 QPS and ~0 at ≥ 3 QPS (the tenant is paused there).
- **Space-sharing is the lever.** Capping the tenant to 20 % of the SM threads (F) or to 32 SMs
  on a green context (G) halves the decode cost (+13 % TPOT at 1 QPS, +12 % at 2) for ~60 % of
  D's tenant progress. The two mechanisms behave the same; the green context can be changed at
  runtime, the MPS percentage cannot.
- **The decode node does not need the whole SM array — but giving the rest to the tenant does
  not help it.** With its steps confined to 128 of 170 SMs (I, tenant absent) decode equals the
  baseline; with a tenant on the other 42 SMs (K) decode pays exactly what it pays under
  un-capped co-running (+26 % TPOT at 1 QPS) while the tenant gets the same ~10.5k tok/s as D.
  The interference is DRAM bandwidth: a decode step streams the 15 GiB of weights every
  ~13 ms, and the tenant's GEMMs on *any* SMs compete for the same memory system. Across the
  schemes the decode cost follows the tenant's throughput, not its placement: 0 tok/s → 12.9 ms,
  6.3–6.7k → 14.4–14.6 ms, 10.5–11k → 16.3–16.4 ms — ≈ +2.5 % TPOT per 1k tenant tok/s at 1 QPS.


## 3. Host↔GPU transfers

Every host↔GPU copy was recorded with its source (figure `../results/<scheme>_xfer.png`; records in
`../results/<scheme>_xfer_{inf,ft}.csv`; NVML PCIe counters in the sampler CSV).

| | per copy | per 11-minute run |
|---|---|---|
| inference loads (DRAM → GPU, one per admitted request) | 768 MiB in 55.7 ms = **13.5 GiB/s** (≈ 93 % of the 14.5 GB/s the link delivers) | 1 176 loads, 880 GiB, 1.36 GiB/s average |
| inference stores (GPU → DRAM, one per finished request) | 16 MiB in 1.2 ms | 1 219 stores, 51 GiB |
| tenant activation offload (tier 2) | 4.7 GB out and 4.7 GB back per step, 354 ms out | 6 steps in scheme D, 56 GiB |
| tenant pause / resume | ~1 GB in 195 / 98 ms | 8 / 9 times in scheme D |
| tenant snapshots | 8 MB, < 50 ms | 176 in scheme D |

Contention: inference loads that overlapped a tenant copy took 56.1–56.3 ms against 55.7 ms
alone (n = 32–74 per scheme) — no measurable slowdown, because the link is busy only ~10 % of the
time on average and the tenant's large copies are rare (activation offload is the slow tier that
the budget ladder seldom lands on). The NVML counters agree with the records on the time
structure (bursts of 15 GiB/s at every admission); their 20-ms sampling overstates peaks (a
35 GiB/s reading during an activation offload is a window artefact).


## 4. What this shows

Figure-centred summary (diagrams 1–4 above, sources in `../diagrams/*.drawio`; the measured curves in
Figures 1–2 above, `../results/<scheme>_timeline.png`, `../results/<scheme>_xfer.png`):

1. **The memory half of the problem is solved** (diagrams 1–3). A router in front of vLLM
   knows the exact KV demand of what is running and queued; the controller raises kvcached's
   limit the moment demand exceeds it and lowers it with hysteresis; the tenant climbs and drops a
   ladder of configurations (micro-batching → activation offload → weight streaming → pause) with
   a 50-ms preemption poll inside its steps and DRAM snapshots. Over an 11-minute schedule from
   0.5 to 4 QPS the engine never failed, never preempted and queued only at the 4-QPS spike
   (2–4 s against 1.2 s alone); resizing itself costs nothing (E = A). Static partitioning (B)
   lost on both sides: 8–26 s of queueing at and after 3–4 QPS, and idle VRAM at low load.
2. **The compute half has a price that no placement trick removes** (diagram 4). The engine is
   busy ~100 % of the time while any request runs, so the idle-window gate (C) only starves the
   tenant. Co-running costs decode TPOT in proportion to the tenant's token rate (+2.5 % per
   1k tok/s at 1 QPS, +20 % per 10k at 2 QPS, nothing at ≥ 3 QPS where the tenant is paused
   anyway), whether the tenant shares the SMs (D), is capped (F, G) or sits on disjoint SMs (K):
   the shared resource is DRAM bandwidth. The usable knob is therefore the tenant's rate —
   MPS percentage or green-context size — chosen from the TPOT inflation one is willing to pay:
   +12 % buys ~6.5k tok/s, +26 % buys ~11k.
3. **Three things that looked right on paper and were not**: the busy signal built on
   `execute_model`/`sample_tokens` measured CPU launch time (Model Runner V2 is asynchronous);
   a gated tenant that cannot be preempted blocks the engine's growth (fixed by polling the
   budget inside the gate wait); and switching a tenant's green context at runtime leaks the
   caching allocator's stream-bound blocks (static partitions only, for now).
4. **Host↔GPU traffic is not the bottleneck**: inference loads run at 13.5 GiB/s, average link
   use is 1.4 GiB/s, and loads that overlap tenant copies are not measurably slower.

What to do next: (a) a feedback controller for the tenant's rate — measure decode TPOT against
its no-tenant value for the current RPS and move the SM cap to hold a target inflation; (b) a
tenant whose steps are bandwidth-light (smaller models, or LoRA on a model whose weights are
already resident, i.e. DeltaServe-style co-serving) — the arithmetic says the interference is
per byte moved, not per SM; (c) a leak-free dynamic SM share: pre-create the two or three green
contexts at start and keep one allocator pool per stream, or move the tenant's switch into a
subprocess restart from its snapshot (the snapshot costs < 50 ms and ~1 s to resume).

