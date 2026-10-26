# KV slots on the decode GPU — baseline study (E0–E3)

**Question.** On the decode engine of a prefill/decode-split Llama-3-8B, how do the KV
space on the GPU, the number of requests it can run, and the VRAM budget relate — and
what, exactly, does a request wait for?

**Answer in one paragraph.** The decode engine is a bank of *integer* KV slots: a request
occupies 392–400 blocks of 16 tokens from the moment it is admitted until its last token,
and the number of requests that can be resident is ⌊grant ÷ 400 blocks⌋ (9.95 → 10 with
evictions, 12, 15 at 24 / 26 / 28 GiB). Waiting is fully explained by "c slots, first
come first served, each request's measured residence": a FCFS replay with that slot count
reproduces the measured mean queue wait to within 2% at every one of 23 grid cells. But
the residence itself is not fixed: every admission copies 0.75 GiB of KV from host RAM
*synchronously inside the engine thread*, and the whole batch stops for ~200 ms each time.
A request resident among more slots sits through more of these stalls (1.8 at 3 running →
12 at 14), so adding slots lengthens residence: +50% slots (24 → 28 GiB) buys +17%
saturation throughput. Up to 1.5 req/s no one waits at any budget, so the budget does not matter at all.

Plan: [PLAN_KV_VRAM.md](../PLAN_KV_VRAM.md) (predictions, annotated); run log:
[DECISIONS_KV_VRAM.md](../DECISIONS_KV_VRAM.md); starting facts:
[NOTES_KV_SLOTS.md](../NOTES_KV_SLOTS.md). E4 (feasibility of APC / layerwise / async /
fp8) and E5 (re-running the grid with them) are not part of this report.

---

## 1. Setup

| item | value |
|---|---|
| machine | 2× RTX 5090 (32 GB), PCIe Gen4 ×8 (14.47 GB/s pinned H2D measured), 32 cores, 60 GiB RAM |
| model | NousResearch/Meta-Llama-3-8B-Instruct, bf16; KV = 128 KiB per token (32 layers × 2 × 8 KV heads × 128 × 2 B) |
| stack | vLLM 0.15.1; prefill engine on GPU0 (util 0.80, APC on); decode engine on GPU1 at the budget under test, APC **off**; KV handed over through LMCache 0.4.4 (chunk 256, 6 GiB pinned L1 per engine, TCP store, Python copy path); router forwards token #1 and admits everything |
| decode engine | `--gpu-memory-utilization` = budget ÷ 31.35 GiB; FLASH_ATTN; CUDA graphs; async scheduling; no MPS, nothing else on GPU1 |
| workload | 32 sessions × 6,144-token fixed prefix + 128 unique tokens, 128 output tokens, zipf-1.1 session popularity, open-loop Poisson arrivals, seed 0, 300 requests; only the rate changes |
| cells | budgets 24 / 26 / 28 GiB × 1, 1.5, 2, 2.5, 3, 3.5, 4 req/s (+ 2.25, 2.75 at 24 GiB); one fresh launch and warm-up per cell |
| instrumentation | record-only engine trace (`scripts/kv_vram/site/kv_trace.py`): per request arrival / admission / first token / finish and free blocks at each; per step batch size and wall time; per `start_load_kv` call its duration. Validated: trace on vs off differs by −0.56 ms decode TPOT, 0% throughput (two runs each) |

Metric conventions: *decode queue wait* = vLLM's `request_queue_time` on the decode
engine (arrival at its scheduler → admitted); *capacity* = the highest rate with mean
queue wait < 50 ms and no failed request; *TPOT* = what the user sees: the mean gap from
token #1 to the last token (token #1 is forwarded from the prefill engine, so the first
gap holds the decode queue + KV load); *residence* = admitted → last token. All 24 cells are `ok` (one cell was rerun after a router keep-alive race lost
a single request; the clean rerun is reported).

## 2. Blocks (vLLM's unit) and slots (this study's unit)

![blocks](../figures/kv_blocks.png)

**Block — vLLM's unit.** PagedAttention stores KV in blocks of 16 tokens (2 MiB each
for this model) and allocates whole blocks: a request is admitted only when its prompt's
392 blocks are free, and takes one more block every 16 generated tokens. No free block at
such a step → a request is preempted.

![grant](../figures/kv_grant.png)

The KV grant is what the budget leaves after the model (budget − 16.23 GiB); it is one
shared pool of blocks.

![slots](../figures/kv_slots.png)

**Slot — this study's unit.** Every request here is 400 blocks, so the pool reads as
"whole requests that fit": c = ⌊blocks ÷ 400⌋. vLLM only sees blocks — that is why the
fractional part matters (24 GiB: 9.95 → a 10th request gets in, then one is evicted).

Probed grants:

| budget | 22 | 23 | 24 | 25 | 26 | 28 GiB |
|---|---|---|---|---|---|---|
| KV tokens | 47,296 | 55,488 | 63,680 | 71,872 | 80,064 | 96,432 |
| blocks ÷ 400 | 7.39 | 8.67 | 9.95 | 11.23 | 12.51 | 15.07 |
| whole slots c | 7 | 8 | 9 | 11 | 12 | 15 |

The fractional part matters through the admission rule (§4.2): at 24 GiB the 0.95 is
enough to admit a 10th request, at 26 / 28 GiB the 0.51 / 0.07 is not.

## 3. What the scheduler does with the blocks

Two situations compete for the same pool of free blocks in every scheduler step:

- **Case 1 — admission.** The head of the FCFS waiting queue needs ⌈tokens/16⌉ free
  blocks (392 for a fresh request). If they are free it is admitted and its prefix KV is
  loaded; if not, it waits — and so does everyone behind it. That wait is the measured
  decode queue time.
- **Case 2 — eviction.** Running requests take one more block every 16 generated
  tokens. When one needs a block and none is free, the most recently admitted running
  request is evicted: blocks freed, progress reset, back to the *head* of the queue. It
  returns only when a slot frees, needing more blocks than before, and its prefix is
  loaded a second time.

![case 1](../figures/kv_case1_admission.png)
![case 2](../figures/kv_case2_eviction.png)

Admission is where the synchronous load lives (Figure "case 1", left): vLLM calls the
connector's `start_load_kv` before the forward pass of the step that admits the request,
LMCache 0.4.4 copies all 32 layers into the paged blocks on the engine thread, and the
step — the whole batch — waits. vLLM has a hook for the alternative (a request can sit in
`WAITING_FOR_REMOTE_KVS` while its KV arrives, right panel, dashed); whether LMCache's
options reach it is E4's question.

## 4. Results

### 4.1 The grid

![heatmaps](../figures/kv_grid_heat.png)

Each cell is one launch and one 300-request run: (a) mean decode queue wait (log colour
scale), (b) output tokens per second, (c) end-user TPOT (log colour scale). The same
numbers:

| req/s | 1 | 1.5 | 2 | 2.25 | 2.5 | 2.75 | 3 | 3.5 | 4 |
|---|---|---|---|---|---|---|---|---|---|
| 24 GiB | 0 (128) | 8 (191) | 153 (254) | 941 (282) | 4,680 (289) | 8,913 (292) | 10,977 (303) | 18,409 (298) | 21,413 (305) |
| 26 GiB | 0 (128) | 0 (191) | 56 (254) | – | 1,054 (309) | – | 6,210 (330) | 12,224 (335) | 16,252 (340) |
| 28 GiB | 0 (128) | 0 (191) | 10 (254) | – | 235 (316) | – | 4,511 (340) | 9,858 (348) | 14,294 (351) |

- **Capacity** (queue < 50 ms): 24 GiB between 1.5 and 2 req/s; 26 GiB ≈ 2 (56 ms);
  28 GiB between 2 and 2.5. The plan predicted 2.0–2.5 / 2.5–3.0 / 3.0–3.5 — each one grid
  step too high.
- **Saturation throughput** ≈ 2.3 / 2.65 / 2.7 req/s: +50% slots buys +17%.
- **While nobody waits, the budget is irrelevant:** at 1 req/s all three budgets give
  128 tok/s, 0 ms queue and the same 17 ms TPOT.
- **Once requests queue, TPOT follows the queue:** at 3 req/s it is 110 / 75 / 71 ms at
  24 / 26 / 28 GiB, because the first token-gap carries the decode queue wait.
- **Preemptions:** 1–35 per cell at 24 GiB, none at 26 / 28 GiB.

The same three quantities as surfaces. In (a) height is the mean wait for a slot: the
flat floor is "nobody waits", the cliff is where waiting begins. The red lines on the
surface join the (budget, rate) points with the same mean wait — 0.05 s (the capacity
line), 0.5, 2, 5 and 10 s, interpolated between the measured cells; their slope says
how much rate one GiB buys: about 0.25 req/s per 2 GiB along the 0.05 s line, less along
the 10 s line. (b) rises with rate until it flattens at each budget's saturation
throughput; (c) is flat while nobody waits and then climbs with rate, less steeply at
the larger budgets.

![3D view](../figures/kv_grid_3d.png)

### 4.2 Is the waiting explained by integer slots alone? A simulation of our explanation

**Our explanation of §4.1.** A request waits for one reason only: every slot is taken,
and the scheduler hands out slots first come, first served. If that is the whole story,
the waiting is fixed by two things — how many slots there are, and how long each request
keeps its slot — and nothing else (scheduler overhead, the router, the KV transfer) adds
to it.

**The check.** We simulate exactly that mechanism with the measured inputs and compare
it to the measured waiting. Picture c parking spaces: cars arrive at the recorded times,
a car that finds a free space takes it at once, otherwise it joins a first-come-first-
served line and takes the next space that frees; each car stays for the time its request
actually stayed (admitted → last token, as traced). The simulation uses, for each grid cell
of §4.1, that cell's own traced run — the unmodified stack at that budget and rate —
taking its 300 arrival times and each request's own residence from the trace; c comes
from the probe. It is a
**re-simulation of a measured result, not a predictive model**: both inputs are
measured, so it cannot forecast an unmeasured budget or rate. What it answers is a yes/no
question — does "c slots, FCFS" account for the waiting we saw?

![re-simulation](../figures/kv_resim_schematic.png)

![queueing models](../figures/kv_mgc.png)

**Answer: yes.** With the slot count the engine actually ran with — 10 at 24 GiB (9 whole
slots plus the admitted 10th, §2), 12 and 15 — the simulated mean wait lies on the
measured one at every rate: within 2% at 24 GiB, within 0.1% at 26 / 28 GiB (the dashed
line is hidden under the measured line). With one slot more or fewer it is off by
1.5–3×, so the match is not loose. The two thin lines are textbook curves for reference —
the M/M/c (Erlang C) formula and its Lee–Longton correction for the spread of residences
— computed from only the arrival rate, mean residence and c; they have the right shape
near saturation but underestimate small waits, because 300 arrivals of one Poisson sample
are far from the steady state those formulas assume.

What follows from the match: the queue needs no explanation beyond slots and residence.
Everything the engine does to a request shows up in its *residence* (§4.3), not in an
extra waiting term.

Why c = 10 at 24 GiB with a 9.95-request grant: the admission rule needs 392 free blocks,
3,980 − 9 × 392 = 452, so a 10th request is admitted while the nine are young; when all
ten grow (10 × 400 > 3,980) one is evicted. The trace confirms max Running = 10 and
9–35 evictions per cell; at 26 GiB (5,004 − 12 × 392 = 300 < 392) Running never exceeds
12 and nothing is evicted.

The closed-form Lee–Longton M/G/c approximation (mean and CV² of the measured residence)
matches only near ρ ≈ 0.9 (26 GiB / 2.5 req/s: 1,037 vs 1,054 ms); at ρ < 0.7 it
underpredicts up to 7× because a 300-request window of one Poisson sample is far from
steady state when waits are tens of milliseconds.

![wait vs free space](../figures/kv_wait_vs_free.png)

Per request the rule is binary: a request that reaches the decode scheduler with ≥ 1
request-worth of free blocks waits 0 ms; one that finds the pool full waits 0.1–10 s
depending on load. The handful of 28 GiB requests that waited with 1–3 requests free
arrived during a multi-request load call — the scheduler thread itself was blocked (§4.3).

### 4.3 What the slots cost: the synchronous load stalls the batch

![time split](../figures/kv_time_split.png)

Per request, residence splits into decode steps that overlap no KV load and steps
stalled by someone's load (the trace marks every `start_load_kv` call):

| cell | mean running | step (no load) | decode | = own steps | + load stalls |
|---|---|---|---|---|---|
| 24 GiB, 1 req/s | 3.3 | 12.2 ms | 1.90 s | 1.57 s | 0.34 s (1.8 × 185 ms) |
| 24 GiB, 2 req/s | 7.5 | 15.0 ms | 2.96 s | 1.87 s | 1.09 s (5.5 × 198 ms) |
| 26 GiB, 3 req/s | 11.6 | 17.9 ms | 3.93 s | 2.05 s | 1.86 s (8.6 × 217 ms) |
| 28 GiB, 3 req/s | 14.3 | 19.4 ms | 4.80 s | 2.22 s | 2.56 s (12.1 × 211 ms) |

Of the 2.9 s of extra residence from the lightest to the heaviest cell, the larger batch
makes the steps themselves slower by 0.65 s (decode is weight-read bound: ~0.5 ms per
extra request on an ~11 ms base) and load stalls add 2.2 s. At 24 GiB / 2 req/s the
stalls are 34% of all decode wall time. One load call: p50 171 ms, mean 198 ms, p95
300 ms for 0.75 GiB — ~4.4 GB/s, a third of the measured 14.47 GB/s link, because
LMCache's compiled kernels do not load against this torch and the Python fallback copies
chunk by chunk.

One consequence is visible in the grid: more slots → more admissions during a residence
→ longer residence, which is why 26 → 28 GiB barely moves saturation throughput. Decode
time per request = 128 steps + (other requests loaded while it is resident) × ~190 ms;
the second term is what the load path (E4) can remove.

### 4.4 Same request scheduling in VRAM 24 vs VRAM 28

**Why this section.** Everything above is a mean over 300 requests. Here the same 300
requests — the same arrival sequence, because the seed is fixed — are run once at 24 GiB
and once at 28 GiB, and one request (#256) is followed through both. It meets exactly the
same neighbours in both runs; the only difference is the number of slots. So whatever
happens differently to it is the effect of the slots alone, seen on one request at one
moment rather than inferred from an average.

![slot timeline](../figures/kv_slots_timeline.png)

What to look at:

- **The followed request.** #256 reaches the decode scheduler at t = 126.3 s. At 24 GiB
  it finds 14 free blocks (0.0 requests), all ten lanes occupied, and waits 1,306 ms for
  lane 1 to free. At 28 GiB the same request finds 888 free blocks (2.2 requests) and
  waits 0 ms. Same request, same neighbours, five more slots: the wait disappears.
- **Row (b): the queue.** How many requests are waiting for a slot at each moment; the
  dashed and dotted lines mark #256's arrival and admission. At 24 GiB it joins a queue of
  three and waits 1.3 s; at 28 GiB the queue is empty and it is admitted at once. The
  empty stretch at 130–135 s in both panels is the arrival process, not the engine:
  122–127 s brought 16 arrivals (3.2 /s) that filled the pool, 127–133 s brought 8
  (1.3 /s) while 17–20 finished.
- **Row (c): the stalls of §4.3 as they happen.** Each dot is one engine step; each pale
  orange band is one synchronous KV load call, and the step that contains it takes
  150–500 ms instead of ~15 ms — no running request gets a token meanwhile. The darker
  band is the call that loads #256: at 28 GiB it is admitted in a cohort of seven, so that
  one call copies seven requests and takes 477 ms.

### 4.5 Hypothesis 1 — the slots hold duplicate prefixes

With prefix caching off on the decode engine, concurrent requests from the same session
each hold their own copy of the 6,144-token prefix. Time-weighted over the traced cells,
10 running requests at 24 GiB span 6.4–6.5 distinct sessions, 12 at 26 GiB span 7.1–7.5,
15 at 28 GiB span 8.2–8.5: **about one third of the resident KV is a copy of a prefix
already on the GPU.** (Predicted from the arrival sequence: 10.3 requests / 6.8 sessions.)
This is the lever E4's decode-side APC test targets.

## 5. Why decode cannot stream its KV instead of holding it

**The motivation.** Everything in §4 follows from one fact: a request occupies 400 blocks
for its whole decode. If a request could hold *less* — say only the few layers the
forward pass is currently working on, with the other layers streamed in from host RAM
just in time and dropped afterwards — the same grant would hold several times more
requests, and the queue of §4.1 would shrink without any extra VRAM. The question is
whether the streaming can keep up with the compute.

![streaming idea](../figures/kv_stream_idea.png)

It cannot, and the reason is per-step traffic: a decode step needs every layer's KV for
every token of every resident request, so streaming moves each request's full 0.78 GiB
across PCIe on every step. The stand-alone benchmark below measures exactly that.

![streaming benchmark](../figures/kv_stream_bench.png)

A stand-alone benchmark (`scripts/kv_vram/bench_stream.py`, no vLLM) runs one
8B-shaped decode step with the KV either resident in HBM or streamed per layer from
pinned host memory with prefetch depth d:

| batch | resident step | copy floor (B × 0.78 GiB ÷ 14.47 GB/s) | streamed, d = 0 | streamed, d ≥ 1 |
|---|---|---|---|---|
| 1 | 12.0 ms | 58.1 ms | 70.3 ms | 58.5 ms |
| 4 | 12.1 ms | 232 ms | 244 ms | 232 ms |
| 16 | 18.0 ms | 928 ms | 946 ms | 928 ms |

Every decode step needs every layer's KV for every resident token, so streaming moves
0.78 GiB per request per step: 4.9× the resident step at batch 1 and 52× at batch 16.
Prefetch depth beyond 1 changes nothing (the copy, not the compute, is the bound), and
the compute that could hide copies is 12–18 ms per step — enough for ~6 of 32 layers of
*one* request at batch 1. Layer-wise pipelining can therefore only shorten the one-time
load; it cannot reduce what must be resident. (This is the distinction between the two
"layer by layer" ideas: streaming KV layer by layer *on every step* is what this section
rules out; LMCache's `use_layerwise` only splits the *one-time load at admission* into
per-layer pieces — the KV is resident for the whole decode either way.)

## 6. What this means for the methods to test (E4)

### 6.1 A first result: LMCache `use_layerwise` (feasibility smoke, 24 GiB, 2.5 req/s)

The one layer-by-layer idea that *is* available is to split the one-time load at
admission into per-layer pieces. The hope is to overlap the KV transfer with the
admitting forward pass: copy layer i, compute layer i, copy layer i+1, … The smoke (two
runs per arm) found a real gain, but not from overlap:

![layerwise](../figures/kv_layerwise.png)

| 24 GiB, 2.5 req/s | baseline (2 runs) | use_layerwise (2 runs) |
|---|---|---|
| decode queue wait | 4,680 / 3,665 ms | 1,409 / 1,556 ms |
| TPOT (end-user) | 64 / 56 ms | 39 / 41 ms |
| throughput | 289 / 294 tok/s | 308 / 307 tok/s |
| load, admitted → first token (p50) | 689 / 566 ms | 217 / 219 ms |
| preemptions per 300 requests | 35 / 34 | 12 / 12 |
| LMCache L1 allocation failures (decode) | 3,360 / 3,360 | 0 / 0 |
| excess step time per residence | 0.88 s | 1.40 s |
| KV grant / slots | unchanged | unchanged |

The diagnosis is in the last two rows. The baseline retrieve reserves staging space in
the 6 GiB pinned L1 for the whole request at once and, with the pool full, fails ~11
times per request and takes a slow path. The layer-wise retriever reserves one layer at
a time and never fails, so a load takes the bare copy time. Faster loads mean the ten
resident requests hit the wall less often (preemptions 35 → 12, each one a second load
avoided), residence falls 4.1 → 3.7 s and the queue 3×. What did **not** happen is
overlap: the copy is still synchronous inside the forward, split into 32 pieces but
never concurrent with compute (figure). The gain is a fixed LMCache staging bug, not
hidden transfer. LMCache's code *is* written to overlap — the per-layer copy goes on a
side CUDA stream and the forward is meant to run meanwhile — but on this box the copy
runs through the Python fallback (the compiled kernels do not load against this torch),
which synchronises the engine thread on every copy, and each layer first blocks on its
L1 fetch. Grouping layers would only trim per-layer overhead; real overlap needs the
compiled kernels.

The other two load-path options could not be tested: LMCache's `enable_async_loading`
crashes the 0.4.4 engine (a bookkeeping `KeyError` on its lookup timeout path), and the
multi-process connector — the one that would truly take the copy off the engine thread —
needs LMCache's compiled kernels, which do not load against this torch build (a source
build is in progress, DECISIONS E4).

| lever | what it changes | expected from this study |
|---|---|---|
| asynchronous / layer-wise load | removes or shrinks the ~200 ms batch stall per admission (2.2 s of the 4.8 s residence at 28 GiB / 3 req/s) | residence down → more throughput per slot; the 4.4 GB/s copy path is the bound, not PCIe |
| fp8 KV | halves bytes per block → c ≈ 19 at 24 GiB | capacity up, but each slot still pays the stalls; needs a bf16 control on the same attention backend (fp8 forces a backend change on sm_120) |

Decode-side prefix caching is deliberately left out. Its gain is not a property of the
engine but of the session mix: it helps only when requests from the same session are
resident together (§4.5: about a third of resident KV here), and every cell of this
study varies the request rate while holding the workload fixed. Testing it would change
what the workload is, not how the engine handles it; the duplicate-prefix share is
reported as an observation instead.

Related-work notes with sources: [related_work_notes.md](../output/kv_vram/related_work_notes.md).

## 7. Data

`output/summary_kv.csv` (one row per cell), `output/raw/kv_*.json` (per-request client
records, engine metrics deltas), `output/kv_vram/trace/*.jsonl.gz` (engine traces),
`output/kv_vram/timeline/*.csv` (joined per-request / per-step tables),
`output/kv_vram/mgc.csv`, `output/kv_vram/probe_base.json`,
`output/kv_vram/pcie_gpu{0,1}.json`, `output/kv_vram/bench_stream.json`. Figures:
`scripts/plots/plot_kv_*.py`.
