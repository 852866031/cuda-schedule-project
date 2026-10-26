# KV slots on the decode GPU (E0–E4, LMCache compiled kernels)

**Question.** On the decode engine of a prefill/decode-split Llama-3-8B, how do the KV
space on the GPU, the number of requests it can run, and the VRAM budget relate — and
what, exactly, does a request wait for?

**Answer in one paragraph.** The decode engine is a bank of *integer* KV slots. A request
holds 392–400 blocks of 16 tokens from admission to its last token, so the number of
requests that can be resident is ⌊grant ÷ 400 blocks⌋: 9.95 → 10 (with evictions), 12
and 15 at 24 / 26 / 28 GiB. Waiting is fully explained by "c slots, first come first
served, each request's measured residence". A FCFS replay with that slot count reproduces
the measured mean queue wait at every cell of the grid (within 0.1% at 26 / 28 GiB, 2% at
24 GiB). The residence itself depends on the slot count. Every admission copies 0.75 GiB
of KV from host RAM *synchronously inside the engine thread*, and the whole batch stops
for ~120 ms each time. A request among more neighbours sits through more of these stalls
(1.6 at 1 req/s → 8.7 at 28 GiB / 3 req/s). So +50% slots (24 → 28 GiB) buys +21%
saturation throughput: 2.97 → 3.58 req/s. Up to 2 req/s nobody waits at any budget, and
the budget does not matter.

**What changed from the first version.** The first version of this study
([report_kv_slots_fallback.md](report_kv_slots_fallback.md)) ran LMCache on its Python
fallback copy path, as did all nine earlier studies on this box. The PyPI wheel's
`c_ops.so` does not load against torch 2.9.1, and LMCache silently falls back. This
report reruns the whole grid with LMCache's own CUDA kernels, built for this torch.
The slot mechanism is unchanged. Every absolute number moved (§5).

Plan: [PLAN_KV_VRAM.md](../PLAN_KV_VRAM.md) (predictions, annotated); run log with each
prediction and outcome: [DECISIONS_KV_VRAM.md](../DECISIONS_KV_VRAM.md) (D6 = this grid).

---

## 1. Setup

| item | value |
|---|---|
| machine | 2× RTX 5090 (32 GB), PCIe Gen4 ×8 (14.47 GB/s pinned H2D measured), 32 cores, 60 GiB RAM |
| model | Meta-Llama-3-8B-Instruct, bf16; KV = 128 KiB per token |
| stack | vLLM 0.15.1. Prefill engine on GPU0 (util 0.80, APC on). Decode engine on GPU1 at the budget under test, APC off. KV handed over through LMCache 0.4.4: chunk 256, 6 GiB pinned L1 per engine, TCP store, **compiled `c_ops` on both engines**. The router forwards token #1 and admits everything. |
| decode engine | `--gpu-memory-utilization` = budget ÷ 31.35 GiB; FLASH_ATTN; CUDA graphs; async scheduling; nothing else on GPU1 |
| workload | 32 sessions × 6,144-token prefix + 128 unique tokens, 128 output tokens, zipf-1.1, open-loop Poisson, seed 0, 300 requests; only the rate changes |
| cells | 24 / 26 / 28 GiB × 1, 1.5, …, 5 req/s (27 cells), one fresh launch per cell; all `ok` |
| instrumentation | record-only engine trace on the decode engine: per request arrival / admission / first token / finish with free blocks at each; per step batch size and wall time; per `start_load_kv` call its duration |

**Kernel build.** LMCache's `c_ops` were built from the 0.4.4 sdist against torch 2.9.1+cu128
(sm_120) in a separate venv, and loaded only when `KV_COPS=1`. A guard fails any cell
whose engines report a different LMCache backend than the arm expects.

**Metrics.**
- *Decode queue wait*: vLLM's `request_queue_time` on the decode engine, from arrival at
  its scheduler to admission.
- *Capacity*: the highest rate with mean queue wait < 50 ms.
- *TPOT*: what the user sees. Token #1 → last token, ÷ 127. Token #1 comes from the
  prefill engine, so the first gap holds the decode queue wait and the KV load.
- *Residence*: admitted → last token.

## 2. Blocks (vLLM's unit) and slots (this study's unit)

![blocks](../figures/kv_blocks.png)

**Block — vLLM's unit.** PagedAttention stores KV in blocks of 16 tokens (2 MiB here).
A request is admitted only when its prompt's 392 blocks are free. It takes one more block
every 16 generated tokens.

![grant](../figures/kv_grant.png)

The KV grant is what the budget leaves after the model: budget − 16.23 GiB, in one shared
pool.

![slots](../figures/kv_slots.png)

**Slot — this study's unit.** Every request here is 400 blocks, so the pool reads as
whole requests: c = ⌊blocks ÷ 400⌋ = 9 / 12 / 15. vLLM only sees blocks, so the
fractional part matters. At 24 GiB, 3,980 − 9 × 392 = 452 ≥ 392, so a 10th request gets
in; when all ten grow past 3,980 blocks, one is evicted. At 26 / 28 GiB the remainder
(300 / 27 blocks) admits no extra request.

## 3. What the scheduler does with the blocks

![case 1](../figures/kv_case1_admission.png)

**Case 1 — admission.** The head of the FCFS queue needs 392 free blocks. If they are free,
it is admitted and its prefix KV is copied in from host RAM. If not, it waits, and so does
everyone behind it; that is the decode queue time. The copy runs inside the step that
admits the request (`start_load_kv` before the forward), so the whole batch waits ~115 ms.
The right panel is the alternative vLLM has a hook for: KV arriving in the background.
LMCache 0.4.4 cannot use it here (§6).

![case 2](../figures/kv_case2_eviction.png)

**Case 2 — eviction (24 GiB only).** Running requests take one more block every 16
tokens. When the pool is empty, the most recently admitted request is evicted to the
*head* of the queue. It returns once a slot frees, and its prefix is loaded a second time.

## 4. Results

### 4.1 The grid

![heatmaps](../figures/kv_grid_heat.png)

Mean decode queue wait in ms (output tokens/s in brackets):

| req/s | 1 | 1.5 | 2 | 2.5 | 3 | 3.5 | 4 | 4.5 | 5 |
|---|---|---|---|---|---|---|---|---|---|
| 24 GiB | 0 (128) | 0 (191) | 11 (255) | 143 (317) | 1,265 (367) | 6,852 (372) | 11,036 (377) | 14,453 (380) | 17,334 (378) |
| 26 GiB | 0 (128) | 0 (191) | 0 (255) | 38 (317) | 340 (378) | 2,565 (410) | 6,381 (418) | 9,998 (423) | 12,551 (429) |
| 28 GiB | 0 (128) | 0 (191) | 0 (255) | 1 (317) | 81 (379) | 894 (427) | 4,137 (441) | 7,184 (451) | 9,546 (459) |

- **Capacity** (queue < 50 ms): 24 GiB 2–2.5 req/s, 26 GiB 2.5–3, 28 GiB 2.5–3.
- **Saturation throughput** 2.97 / 3.35 / 3.58 req/s. +50% slots buys +21%; 26 → 28 GiB
  (+25% slots) buys +7%.
- **While nobody waits, the budget does not matter.** At 1–2 req/s all three budgets
  give the same throughput, 0 ms queue and the same TPOT (14.5 ms at 1 req/s).
- **Once requests queue, TPOT follows the queue.** At 3.5 req/s it is 72 / 44 / 36 ms at
  24 / 26 / 28 GiB, because the first token gap carries the queue wait.
- **Preemptions** only at 24 GiB: 0–1 per cell up to 2 req/s, 7–11 at 2.5–3, ~34 once
  saturated. None at 26 / 28 GiB.

![3D view](../figures/kv_grid_3d.png)

The same three quantities as surfaces. In (a) height is the mean wait. The flat floor is
"nobody waits", the cliff is where waiting begins. The red lines join (budget, rate)
points with equal mean wait (0.05, 0.5, 2, 5, 10 s), interpolated between cells. Their
slope says how much rate one GiB buys. (b) flattens at each budget's saturation
throughput. (c) is flat while nobody waits, then climbs, less steeply at larger budgets.

### 4.2 Is the waiting explained by integer slots alone? A simulation of our explanation

**Our explanation.** A request waits only when every slot is taken, and slots go first
come, first served. If that is the whole story, the waiting depends only on how many
slots there are and how long each request keeps one.

**The check.** Simulate exactly that with measured inputs. Picture c parking spaces. Cars
arrive at the recorded times. A car that finds a free space takes it; otherwise it joins
a FCFS line. Each car stays as long as its request actually stayed (traced). Both inputs
are measured, so this is a **re-simulation of a measured result, not a predictive
model**. It answers one yes/no question: does "c slots, FCFS" account for the waiting?

![re-simulation](../figures/kv_resim_schematic.png)

![queueing models](../figures/kv_mgc.png)

**Answer: yes.** With the slot count the engine actually ran with (10 / 12 / 15), the
simulated mean wait matches the measured one within 0.1% at every 26 / 28 GiB cell with a
wait. At 24 GiB it matches within 2% from 2.5 req/s up (−7% at 2 req/s, where the wait is 11 ms). With one
slot more or fewer it is off by 1.5–3×, so the match is not loose. The thin lines are
textbook curves for reference (Erlang C and Lee–Longton M/G/c). They have the right shape
near saturation but are off for small waits, because 300 arrivals are far from the steady
state they assume.

So the queue needs nothing beyond slots and residence. Everything the engine does to a
request shows up in its *residence* (§4.3).

![wait vs free space](../figures/kv_wait_vs_free.png)

Per request the rule is binary. A request that reaches the decode scheduler with ≥ 392
free blocks waits 0 ms. One that finds the pool full waits 0.02–5 s, depending on load.
The few requests that waited with 1–4 requests' worth of blocks free arrived during a
load call: the scheduler runs on the same thread as the synchronous copy, so it was
blocked (§4.3).

### 4.3 What the slots cost: the synchronous load stalls the batch

![time split](../figures/kv_time_split.png)

Per request, the decode time (token 1 → last) splits into steps that overlap no KV load
and steps stalled by someone's load:

| cell | mean running | step (no load) | decode | = own steps | + load stalls |
|---|---|---|---|---|---|
| 24 GiB, 1 req/s | 2.9 | 11.6 ms | 1.71 s | 1.54 s | 0.17 s (1.6 × 106 ms) |
| 24 GiB, 2 req/s | 5.8 | 12.5 ms | 2.24 s | 1.76 s | 0.48 s (4.2 × 114 ms) |
| 24 GiB, 3 req/s | 9.4 | 16.2 ms | 2.97 s | 2.02 s | 0.95 s (7.9 × 120 ms) |
| 26 GiB, 3 req/s | 10.4 | 16.3 ms | 3.03 s | 2.03 s | 1.00 s (8.3 × 121 ms) |
| 28 GiB, 3 req/s | 11.0 | 16.0 ms | 3.15 s | 2.08 s | 1.07 s (8.7 × 122 ms) |

From the lightest to the heaviest cell, decode time grows 1.44 s. Slower steps from the
larger batch account for 0.54 s; load stalls account for 0.90 s. At 24 GiB / 2 req/s the
stalls are 20% of decode wall time (fallback: 34%). One load call copies 0.75 GiB in
~115 ms (~7 GB/s), half the measured link speed. The kernel itself takes 2 ms per layer;
the rest is LMCache's L1/store fetch and Python around it.

This is why more slots give diminishing returns. More slots mean more admissions during
a residence, which means a longer residence. Decode time per request is roughly
128 steps + (loads during its residence) × ~120 ms. The second term is what an
asynchronous load path would remove.

### 4.4 Same request scheduling in VRAM 24 vs VRAM 28

**Why this section.** Everything above is a mean over 300 requests. Here the same 300
requests are run once at 24 GiB and once at 28 GiB, at 3 req/s. The arrival sequence is
identical because the seed is fixed. One request (#255) is followed through both runs. It
meets the same neighbours in both runs, so any difference is the effect of the slots
alone.

![slot timeline](../figures/kv_slots_timeline.png)

What to look at:

- **The followed request.** #255 reaches the decode scheduler at t = 84 s.
  - At 24 GiB it finds 19 free blocks, all ten lanes taken, and **waits 4,226 ms**.
  - At 28 GiB it finds 89 free blocks and **waits 750 ms**.
  - Once admitted, its load (112 / 120 ms) and decode (3.1 / 3.0 s) are nearly the same.
    The five extra slots change the wait, not the service.
- **Row (b): the queue.** At 24 GiB, 5–13 requests wait throughout this window, and #255
  joins a queue of ~13. At 28 GiB the queue empties within a second or two after each burst.
- **Row (c): the stalls of §4.3 as they happen.** Each dot is one engine step; each pale
  orange band is one KV load call. A step that contains a load takes 30–300 ms instead of
  ~16 ms, and no running request gets a token meanwhile. At 28 GiB there are more bands,
  because more admissions fit.

### 4.5 Observation — the slots hold duplicate prefixes

Prefix caching is off on the decode engine. Concurrent requests from the same session
therefore each hold their own copy of the 6,144-token prefix. Time-weighted, at 3 req/s
9.2 / 10.0 / 10.1 running requests span 6.1 / 6.5 / 6.4 distinct sessions (24 / 26 /
28 GiB). **About a third of the resident KV is a prefix already on the GPU.** This is
reported, not tested: decode-side prefix caching would change what the workload is, not
how the engine handles it.

## 5. Compiled kernels vs the Python fallback

Same flags, same workload, same grid. Only LMCache's copy path differs.

| | fallback | compiled kernels |
|---|---|---|
| copy per request (`start_load_kv`) | 180–200 ms | 112–124 ms |
| load, admitted → token 1, p50 at 3 req/s | 327–584 ms | 168–182 ms |
| stall share of decode wall time, 24 GiB / 2 req/s | 34% | 20% |
| residence at 3 req/s (24 / 26 / 28 GiB) | 4.05 / 4.36 / 5.18 s | 3.15 / 3.25 / 3.35 s |
| capacity (24 / 26 / 28 GiB) | 1.5–2 / 1.5–2 / 2–2.5 req/s | 2–2.5 / 2.5–3 / 2.5–3 req/s |
| saturation throughput | 2.38 / 2.66 / 2.75 req/s | 2.97 / 3.35 / 3.58 req/s |
| gain 24 → 28 GiB | +16% | +21% |
| TPOT at 1 req/s | 17 ms | 14.5 ms |
| TTFT p50, 24 GiB, 2.5 / 1 req/s | 163 / 38 ms | 48 / 32 ms |

- **The mechanism did not change.** It is still integer slots, FCFS, and a synchronous
  load per admission. The re-simulation now matches more tightly.
- **Each stall is shorter, but not gone.** Stall count per residence is about the same at
  equal load; each stall is ~120 ms instead of ~210 ms.
- **The prefill engine benefits too.** Its KV save to LMCache uses the same kernels, so
  TTFT is ~3× lower at 2.5 req/s.
- **What is not fixed.** LMCache's L1 staging pool still fails allocation ~3,400 times per
  run (≈ 11 per request) on the whole-request retrieve path. And the trace's
  zero-overhead check (trace on vs off) was done under the fallback, not repeated here.

## 6. Can the load leave the critical path?

### 6.1 Streaming KV per step: no

![streaming idea](../figures/kv_stream_idea.png)

**The idea.** If a request could hold only the layers being computed, with the rest
streamed from host RAM just in time, the same grant would hold several times more
requests.

![streaming benchmark](../figures/kv_stream_bench.png)

**Why it fails.** A decode step needs every layer's KV for every resident token. So
streaming moves the full 0.78 GiB per request *per step*. A stand-alone benchmark (no
vLLM) gives a streamed step of 58 ms at batch 1 vs 12 ms resident (4.9×), and 928 ms vs
18 ms at batch 16 (52×). Prefetch depth beyond 1 changes nothing, because the copy is the
bound. Pipelining layers can shorten the *one-time* load at admission; it cannot reduce
what must stay resident.

### 6.2 LMCache's load-path options

**`use_layerwise`** splits the one-time load at admission into per-layer pieces, hoping to
overlap copy with compute. Measured at 24 GiB / 2.5 req/s:

![layerwise measured](../figures/kv_layerwise_measured.png)

No gain: queue 143 → 136 ms, own load 0.18 → 0.19 s, residence 2.83 → 2.76 s, all within
run-to-run noise. The direct check of overlap is the time that load-carrying steps add to
a request's residence: 0.67 s without, 0.70 s with layerwise. If the copies overlapped the
forward, this would shrink.

Why there is no overlap: the admitting step, drawn to scale from the measured copy
(115 ms) and forward (14 ms):

![layerwise schematic](../figures/kv_layerwise.png)

Each layer's copy waits for that layer's L1/store fetch, and the forward for layer i
waits for copy i. The 32 pieces alternate but never run at the same time, so the step
takes the same total either way. The schematic shows *why*; the measured figure above
shows that this is what happened.

**The other two options are not usable on this stack.**
- `enable_async_loading` crashes the 0.4.4 engine (a `KeyError` on its lookup timeout
  path).
- The multi-process connector (`LMCacheMPConnector`), which takes the copy off the engine
  thread, needs the compiled kernels. Those now exist, so it is the remaining candidate;
  it has not been tested yet.

| lever | what it would change | status |
|---|---|---|
| asynchronous load (MP connector) | removes the ~120 ms batch stall per admission (1.07 s of 3.15 s decode at 28 GiB / 3 req/s) | untested; needs the compiled kernels, now built |
| fp8 KV | halves bytes per block → ~19 slots at 24 GiB; each slot still pays the stalls | not run; needs a bf16 control on the same attention backend |
| decode-side prefix caching | frees the duplicated third of resident KV (§4.5) | not tested: it changes the workload |

Related-work notes with sources: [related_work_notes.md](../output/kv_vram/related_work_notes.md).

## 7. Data

- `output/summary_kv.csv`: one row per cell. Rows `kv_cops_*` are this report; the
  `lmcache_backend` column holds `lmcache.c_ops`.
- `output/raw/kv_cops_*.json`: per-request client records and engine metric deltas.
- `output/kv_vram/trace/`, `output/kv_vram/timeline/`: engine traces and joined
  per-request / per-step tables.
- `output/kv_vram/mgc.csv`: the re-simulation.
- Figures: `KV_METHOD=cops .venv/bin/python scripts/plots/plot_kv_*.py`.
