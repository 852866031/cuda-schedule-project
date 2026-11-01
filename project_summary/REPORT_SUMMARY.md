# The CUDA-scheduling project — summary of what was done, found and built

Companion to [REPORT_CHRONOLOGICAL.md](REPORT_CHRONOLOGICAL.md), which walks the same work in
date order. This report is organised by theme. Every number is taken from the branch documents
(reports, plans, decision logs, CSVs) as merged into `main` on 2026-11-01; where documents
disagree, the discrepancy is listed in Appendix B rather than silently resolved.

---

## 1. The project in one page

The project started from a measurement question — *what does "GPU utilization" actually
measure?* — and ended at a scheduling question — *what governs a decode engine's capacity when
its KV cache is offloaded to host memory?* Six branches, ten numbered studies on one 2× RTX 5090
workstation, March to October 2026.

1. **Utilization is not work** (`util_examples`). NVML "GPU-Util" is time-with-a-kernel-resident;
   DCGM `SM_ACTIVE`/`SM_OCCUPANCY`/`DRAM_ACTIVE` separate *how often* from *how full*; only
   kernel-level tools (Nsight, CUPTI) see efficiency. A CUPTI injection tracer costs ~2% on an
   LLM workload, an auto-profiler ~14%.
2. **Interference is resource- and scale-specific** (`gpu-interfere`). Reproducing *Understanding
   GPU Resource Interference One Level Deeper* (arXiv 2501.16909) on the RTX 5090: GPU-wide resources (DRAM
   bandwidth, L2) interfere even across disjoint SMs — SM partitioning (e.g. NVIDIA's
   Multi-Process Service, MPS) cannot help; per-SM (L1, block slots)
   and per-SM-sub-partition (SMSP: issue slots, FP pipes) resources interfere only when kernels share them. Two
   mechanisms recur: capacity cliffs (L2 5.6×, L1 5.7×) and serialization (2×). Per-kernel rate
   counters cannot predict capacity effects; direct colocation must be measured.
3. **A kernel scheduler can be built in user space, and stream priority is the lever that
   works** (`colocation`). An Orion-style LD_PRELOAD colocator with a four-timestamp CUPTI
   observer showed that two CUDA streams buy no concurrency against an SM-saturating kernel
   (latency client 0.46 → 20.8 ms p50) and that stream priority restores it (0.98 ms, −2%
   throughput, 89% overlap).
4. **The literature's gap is phase awareness** (`literature-review`). Of seven GPU-sharing
   systems reviewed against Orion, none schedules by prefill/decode or KV state; SM partitioning does not partition
   HBM/L2; data movement must be scheduled with compute. Tally reproduced (high-priority p99 8.4× better
   under colocation); NanoFlow reproduced (12,401 tok/s/GPU) but its SM-partition overlap lost
   14–32% on this card.
5. **VRAM limits** (`vram-limit`, studies 1–4). Under DRAM offload, inference hits a
   *concurrency* wall (in-flight KV must stay resident: 30 → 22 GiB costs 9% TTFT p95, then a
   two-step collapse), fine-tuning hits a *PCIe-bandwidth* wall (smooth: 3.25 GiB for 1.4% with
   the best prefetch variant), the wall belongs to the workload not the cache backend (LMCache: same cliff, but
   queues instead of stalling), and a prefill/decode split keeps TTFT flat at every decode budget
   while the decode GPU's wall is in-flight KV (~6 GiB at 2 req/s) with its SMs < 5% occupied.
6. **Colocation on the decode GPU** (studies 5–9). MPS is mandatory: without it time-slicing
   collapses decode 42× and two decodes serialise to 3.4× per-token latency. Given MPS, a
   resident neighbour costs ~20% per-token latency, but the honest price is *capacity*, read
   from scheduler queue wait. With N engines the first thing to break is host CPU threads. A
   prefill neighbour costs the decode **KV slots, not tokens** (a slot = the KV space one
   request occupies on the decode GPU, held for its whole *residence* = KV load + decode). MPS
   client priority protects the decode; a launch gate does not.
7. **The KV-slot model** (study 10). The decode engine is a bank of integer KV slots served
   first-come-first-served; waiting is fully accounted for by slot count and measured residence;
   below the knee the VRAM budget is irrelevant, above it capacity grows sub-linearly in the
   budget (+50% slots → +21%) because every synchronous KV load stalls the whole batch and more
   slots mean more stalls per residence. Taking the copy off the critical path (LMCache's MP
   connector) makes residence independent of the slot count and each slot worth a full slot.
   LMCache's compiled kernels had never loaded in any of the LMCache studies (3–9).

The thread running through all of it: **a single utilization or capacity number hides the
mechanism; the project's results come from instrumenting the mechanism** (CUPTI timestamps,
per-request engine traces, per-thread host counters) and reading capacity as queueing.

## 2. Platform, workload and method conventions

**Machine.** 2× RTX 5090 32 GB (sm_120, 170 SMs), PCIe Gen4 ×8 to the host — 14.47 GB/s host-to-device,
14.33 GB/s device-to-host, pinned, measured; no GPU peer-to-peer (GeForce); 32 cores; 60 GiB RAM; no
`sudo`, `ptrace_scope=1`, `perf_event_paranoid=4`. Every conclusion about offloading is "a
statement about PCIe Gen4 ×8, not about offloading as an idea" (study 1).

**Stack.** vLLM 0.15.1 (torch 2.9.1+cu128) pinned for all serving studies; lmcache 0.4.4 (the
only release pairing with it); vLLM 0.18.1 (torch 2.10) only for study 10's MP-connector arm.
Model: Meta-Llama-3-8B-Instruct, 0.125 MiB (128 KiB) of KV per token; neighbours Qwen2.5-0.5B /
3B, GPT-2 124M.

**The reference workload** (identical in every serving study): 32 sessions × 6144-token fixed
prefix = exactly 24.0 GiB of KV; request = prefix + 128 unique tokens; 128 output tokens;
sessions drawn zipf-1.1 (90% of traffic on 17 sessions = 12.75 GiB) or uniform; open-loop
Poisson, 2 req/s unless swept; 300 measured requests for the 8B (study 9's 3B neighbour ran
150 s windows); fixed seed. One request at decode = 6400
tokens ≈ 0.78 GiB. Its deliberate properties: prefixes never grow, so cache state alone
distinguishes a first request (~530 ms prefill) from later ones (~30 ms); the short unique
suffix maximises sensitivity ("relative improvements here are an upper bound").

**Terms used throughout.**
- *Split*: prefill runs on GPU0, decode on GPU1, and the prompt's KV is handed over through a
  shared LMCache store in host DRAM (studies 4–10). *Colocated*: one vLLM engine does both on
  one GPU (studies 1, 3).
- *Offload*: KV evicted from GPU memory is kept in pinned host DRAM and fetched back over PCIe
  instead of recomputed; a decoding request still needs all of its KV resident on the GPU.
- *KV grant*: the GPU memory vLLM reserves for KV blocks at a given `--gpu-memory-utilization`
  (here ≈ budget − 16.2 GiB; the 16.2 GiB is the 8B's weights, ~15 GiB, plus vLLM's activation
  and overhead reservation). *Slot*: the KV space one request of this workload occupies
  (6400 tokens = 400 blocks ≈ 0.78 GiB); whole slots c ≈ ⌊grant ÷ 400 blocks⌋ (§3.5 below
  refines this). *Residence*: the time a request holds its slot — KV load + 128 decode steps. *Slot utilization*: offered rate × residence ÷ slots — how many of the
  slots the offered load would keep busy (study 9); past ~1.0 the queue grows without bound.
- *Knee*: the load or budget at which the decode engine's queue wait leaves zero (capacity
  below, queueing above). *Capacity*: highest request rate with mean decode queue wait < 50 ms.
- *TTFT*: time to the first token. *TPOT*: (end-to-end − TTFT) ÷ 127, the per-token time the
  user sees, which includes any wait between token 1 and token 2. *ITL*: the per-gap
  inter-token distribution (its median is the decode step time). Studies 8–10 turn on this
  difference: a neighbour or a load stall leaves the median ITL unchanged and shows up in TPOT
  and residence.
- *MPS*: NVIDIA's Multi-Process Service, which lets kernels of different processes co-reside
  on the GPU instead of time-slicing it; its SM cap (`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`) and
  client priority (`CUDA_MPS_CLIENT_PRIORITY`) are the two knobs used.
- *Fallback*: LMCache 0.4.4's Python/torch copy path, used whenever its compiled `c_ops`
  kernels fail to load (studies 3–9 and study 10's first pass).
- *P1 / P2 / P3*: where a prefill neighbour's KV goes — dropped / its own DRAM store / a decode
  engine on the other GPU (studies 8–9).
- *FCFS*: first come, first served. *APC*: vLLM's automatic prefix caching.

**Conventions that emerged and were then enforced** (`CLAUDE.md` on `vram-limit`):
pre-registered plans whose predictions are annotated, never rewritten; per-run decision logs
(prediction before, outcome after); capacity = `vllm:request_queue_time_seconds` mean < 50 ms
with ≤ 1% failed, not delivered/offered throughput and not SM-active; a cell is `ok` only with
zero failed requests; split-system TTFT only with the token-forwarding router and no admission
cap; decode-only TTFT is not reported (it is queue + KV reload); p50 TTFT varies ±15 ms run to
run, throughput dips are the trustworthy knee signal; config, raw-file and telemetry names must
never collide (data was lost that way once); `OMP_NUM_THREADS` ≤ cores ÷ engines whenever more
than one engine runs; `PYTHONHASHSEED=0` for cross-process LMCache; stacks launched only through
budget-checking scripts with `mem_guard.sh` running, because pinned-memory overcommit
hard-freezes the box.

## 3. Findings by theme

### 3.1 Measuring GPUs: utilization, interference, and what counters can and cannot predict

- *GPU-Util* (NVML) measures the fraction of time any kernel is resident. Two kernels with the
  same GPU-Util can differ by orders of magnitude in work; DCGM's `SM_ACTIVE` (how often the GPU
  runs) and `SM_OCCUPANCY` (how full the SMs are when it runs) separate the two — the split
  study later reads a decode GPU at SM-active 51–68% with occupancy < 5%.
- Interference between colocated kernels depends on the shared resource and its scale (RTX
  5090): DRAM bandwidth is one pool for all 170 SMs (two copy kernels on disjoint SM halves each
  get ~60% of alone bandwidth; aggregate 1133 GB/s ≈ the 1204 GB/s ceiling); L2 (~96 MB) is one
  cache (5.6× latency at a 40 MB combined footprint); L1 is per SM (5.74× at 40 KB per block,
  gone at ≥ 64 KB); block slots serialize at 2 blocks/SM (2.0×); FP64 pipes serialize fully
  (1.8–2.0×); issue slots contend 40% toward serial. A `torch.mm` beside an FMA kernel slows
  5.6× at 512³ but only 1.1× at 4096³ — mid-size, compute-dense, under-filling matmuls are the
  danger zone.
- Nsight Compute (NCU) rate counters of each kernel alone predict same-resource contention (16 of 21 probe
  pairs) but miss capacity cliffs and shared-datapath effects (`l1+l1` predicted none, measured
  2.50×; `fma+fp64` 2.28×). Interference has to be measured by colocating.
- The colocator's four-timestamp trace made the hardware's behaviour legible: a plain second
  stream gives "launch → GPU-start" p50 ≈ 20 ms behind a 16 K-block SGEMM while software queue
  delay is 0.2 µs; priority −5 lets the latency client's blocks enter SMs as they free up. Two
  implementation hazards were found only by measurement: blocking syncs stall a scheduler loop;
  `cudaMemcpyAsync` into pageable memory is silently synchronous.

### 3.2 Kernel-level control and the literature

- **Mechanism works, policy is open.** The colocator (interpose → record → queue → replay on
  colocator-owned priority streams, passthrough for the scheduler's own calls) reached parity with
  Orion's mechanism, added cuBLAS interception and a Llama-3-8B decode client by 2026-08-04, and
  reproduced the interference ladder through its own queues (fma+fma 1.8×, dram+dram 2.0×, l2+l2
  1.8×). A throttle policy showed the cost of software control: each stall costs ~30 µs of
  completion-detection latency. Green-context SM partitioning costs a bandwidth-bound GEMV only
  ~5% on half the SMs. No interference-aware policy was written.
- **What the literature does and does not cover** (seven GPU-sharing systems reviewed against
  Orion, four LLM-serving papers). Hummingbird/Tally/Bless/LithOS/MMK share one architecture — transparent interception,
  profiled or predicted kernel behaviour, a policy, an enforceable unit (split kernel, slice,
  persistent block, squad, atom) — and differ in the unit of control. Three unresolved problems
  the review names: preemption granularity vs overhead; **SM isolation does not isolate L2, HBM,
  copy engines, PCIe**; upper-layer placement and lower-layer kernel scheduling are
  disconnected. "For LLM systems specifically, the most important extension is phase awareness":
  none of them understands TTFT/TPOT, prefill/decode or KV state (Hummingbird and LithOS run
  LLMs on real serving traces; Tally, MMK and Usher include LLMs as generic models). The LLM-serving set
  (NanoFlow, ShadowServe, MuxWise, DuetServe) adds: compute partitioning does not partition
  bandwidth; the right policy (overlap vs isolate vs disaggregate) changes with prefill shape,
  batch, context and GPU; **data movement (KV fetch) should be scheduled with computation**.
- **Reproductions.** Tally on sm_120 after four fixes: the high-priority (HP) client's p99
  2.103 → 0.250 ms under colocation (8.4×) with the best-effort (BE) client's throughput
  unchanged — but the BE cuBLAS kernels were only
  launch-gated, not transformed. NanoFlow: Llama-3-8B 12,401 tok/s/GPU (paper: 12,756 on an
  A100-80GB) after ~10 porting fixes; its headline SM-partitioned overlap *lost* 14–32% here
  because a 32 GB card forces a small decode share; the win it did reproduce came from dense
  batching, nano-batched chunked prefill, async scheduling and CUDA graphs. Both artefacts had
  rotted (hard-coded compute-capability lists, dead mirrors, missing files, a KV-page leak).

### 3.3 VRAM limits under DRAM offload (studies 1–4)

| study | knob | binding wall | shape | key numbers |
|---|---|---|---|---|
| 1 inference, native offload | GPU budget 30 → 18 GiB, 24 GiB pinned DRAM tier | **concurrency** (in-flight KV) | flat, then two-step collapse | 30 → 22 GiB: TTFT p95 305 → 332 ms (+9%), 0 preemptions; 20 GiB 135/1528 ms; 19 GiB 2078/5451; 18 GiB engine stall. Fetch 56 ms vs recompute 530 ms. PCIe 5.7% used |
| 2 LoRA fine-tune, weights streamed | layers offloaded 0–32 | **PCIe bandwidth** | smooth, additive | on-demand step = 1.144 s + GiB×2/14.47 GB/s (±0.4%); best prefetch variant: 3.25 GiB for 1.4%, 6.5 GiB for 12.7% (plain depth-1 tail prefetch 1.6% / 15.3%); depth 2 and interleaving move it ≤ 3% |
| 3 inference, LMCache tier | same sweep | same wall (19–20 GiB) | queue, not spiral | 98% tier coverage vs native 93–95%; uniform p95 321–455 vs 594–606 ms; 18 GiB: 143–172 tok/s, all 300 requests complete, 0 stalls |
| 4 prefill/decode split | decode budget 30 → 18 GiB | **in-flight KV** on the decode GPU | TTFT flat; throughput/TPOT knee at 22 → 20 GiB | TTFT 74–111 ms at every budget; tok/s 254 → 208 → 134; TPOT 28 → 132 → 535 ms; prefill GPU SM-active 1.8–3.4%; decode GPU DRAM-active 45–62%, occupancy < 5% |

What these four established:

- **Offload is a prefix cache, not paging.** An actively decoding sequence still needs all its KV
  in VRAM; offload rescues reuse across requests. Its value scales with the *lack* of locality
  (1.1× for zipf, 7.7× for uniform at full VRAM) but the knee location is set by in-flight count
  × per-request KV and is independent of locality and of the cache backend.
- **The knee is computable in advance**: Little's law gives the mean in-flight count
  (2 req/s × 1.83 s ≈ 3.7), its Poisson p95 (7) × 0.766 GiB = 5.36 GiB; the data bracket it
  (5.77 GiB clean, 3.77 broken).
- **Past the knee (budgets below ~20 GiB), the failure mode is a backend property**: vLLM's native path recomputes
  evicted KV (~530 ms) and a less-KV → more-misses → more-in-flight loop spirals into
  preemption and a deterministic engine stall; LMCache re-admits via a ~151 ms fetch, so
  overload queues.
- **Streaming KV during decode is not an option on this link** (study 10 later quantified it:
  58 ms × batch per step vs 12–18 ms resident).
- **The split isolates TTFT but not capacity**: token #1 comes from the prefill GPU, so TTFT is
  flat down to 1.77 GiB of decode KV — at the price of ~2× the colocated system's TTFT (105 vs
  56 ms) and a second GPU; the decode GPU's floor is weights + in-flight KV ≈ 21 GiB
  for this workload's concurrency. Retrieval (~430 ms on the fallback path) lands in the token
  1 → 2 gap, which is why split TTFT must be measured with a router that forwards prefill's
  token.
- **Two idle resources** fell out of study 4 and set the agenda: a prefill GPU 97% idle and a
  decode GPU whose compute units are < 5% occupied.

### 3.4 Colocation on the decode GPU (studies 5–9)

The decode GPU (8B split, GPU1) was lent to four kinds of neighbour. Results by neighbour:

| neighbour | without MPS | with MPS | best additional lever | the cost that matters |
|---|---|---|---|---|
| LoRA trainer (study 5) | decode TPOT 28 → 1193 ms, 102 timeouts (time-slicing) | 10% SM cap: +23% TPOT, 0 failures; uncapped +67% | idle-window / CUPTI gate: TTFT back to baseline, TPOT +41–46% | TPOT; an unattributed +45–70 ms TTFT offset in every MPS-cap arm (control skipped) |
| 0.5B decode tenant (study 6) | 8B TPOT 28 → 95–98 ms; capacity 327 → 245 tok/s (−25%) | TPOT 33–34 ms; capacity ~300 (−8%) | gate on top of MPS at saturation: 298 → 315 tok/s, tenant forced eager | capacity; a tenant whose KV streams from DRAM (TTFT 7.7 s → 348 ms with MPS; these offload cells were swap-confounded — the no-MPS run paged in 27× more — so the "fits" cells are the clean result) |
| N identical engines (study 7) | not run (by decision) | 0.5B: aggregate near-linear to N=6, TPOT 1.56 → 5.70 ms at N=8; 3B knee N≈2–3, VRAM holds 4 | `OMP_NUM_THREADS` ≤ cores ÷ N | per-token latency (lost batching: 1.97 ms on 1 engine vs 7.78 ms on 8 at equal load), host CPU threads, host RAM at N=6 offload |
| 3B prefill, 8B at its knee (study 8) | — (MPS in every cell) | 8B −1 / −7 / −19% throughput at 0.5 / 1 / 2 req/s; ITL unchanged 14.4 → 14.5 ms | async-save patch (3B TTFT 576 → 234 ms); router readiness gate (300/300 full KV hits) | **KV slots**: residence 3.1 → 4.3 s, 8.6 slots needed > 7 |
| 3B prefill, 8B with headroom (study 9) | — | fair at 8 req/s: 8B −18% / 18 s queue (23 GiB, bf16 3B) vs −3% / 1.3 s (25 GiB, fp8 3B) | **MPS client priority**: TPOT 166 → 88 ms, 209 → 227 tok/s for +20–70 ms 3B TTFT; launch gate ≈ fair and starves the 3B | slot utilization = rate × residence ÷ slots |

The general findings:

- **MPS is mandatory** for either kind of neighbour measured (trainer or second decode);
  without it processes time-slice the GPU
  (the telemetry tell: adding a tenant *lowers* GPU1's measured utilization because the GPU
  waits for turns). Given MPS and a resident footprint, a fine-tune neighbour (+23% at a 10%
  cap) and a second decode (+20%) cost about the same.
- **Price colocation as capacity, not as a latency point**: a single 2 req/s cell "flatters by
  ~3×"; capacity is read from scheduler queue wait, which is why the queue-wait criterion was
  adopted.
- **The host breaks before the GPU**: each vLLM EngineCore sizes its OpenMP pool to all 32
  cores (~135 threads); four engines with DRAM-offloaded KV spin-wait, the GPU idles at 5%
  SM-active and 489 of 1200 requests fail (4 of 4 runs); `OMP_NUM_THREADS=4` fixes it at no cost.
  Host RAM (60 GiB; pinned pools + stores + ~5 GB RSS per engine) bounded study 6 (a second
  full split deferred), study 7 (swap at N=6 offload) and study 8 (P3 only swap-backed with
  pools cut to 4 GiB), and forced study 9's rule that cells with > 0.5 GB swapped in are rerun.
- **KV reloads queue before the GPU is full.** In study 7 the 3B engines with offloaded KV hit a
  limit before GPU time, VRAM or RAM did: decode queue wait 82 ms → 4.7 s from N=1 to N=4 with
  only ~2 requests running and KV 43% used, because the per-request KV load stretched 85 →
  401 ms. The cause was left open there; study 10 later found the synchronous load stall.
- **A neighbour costs the decode slots, not tokens.** Its median step is untouched; what
  stretches is each request's *residence* (KV load + decode), and the decode queues once offered
  rate × residence approaches the number of requests its KV grant holds. All non-gated cells of
  study 9 fall on one curve of queue wait vs slot utilization: 0.1–0.3 s below 0.7, 1–2 s at
  0.8–0.87, 4–8 s at 0.92–0.99, unbounded past 1.0. A lighter neighbour (fp8 3B: residence
  growth +54% → +34%) and more decode KV (23 → 25 GiB: more slots) act on the two sides of the
  ratio and each closed about half the gap.
- **Priority: MPS client priority works, launch gating does not.** `CUDA_MPS_CLIENT_PRIORITY=1`
  on the prefill engine recovers about a third of the capacity fair sharing gives up (8B alone
  281 tok/s at 2.5 req/s; fair 241–246; with priority 255); the Orion-style gate (hold the
  neighbour's launches while the decode's `execute_model` is busy) gives the decode nothing over
  fair sharing against a prefill neighbour, starves the prefill (TTFT 30 s), and forces the
  gated engine to run eager because CUPTI callbacks break CUDA-graph capture. The gate did
  protect TTFT against a fine-tune neighbour (study 5) and gave a small capacity top-up against
  a 0.5B tenant at saturation (study 6). Why it fails against a prefill is not settled in the
  sources: study 5 attributes its residual TPOT cost to LMCache's KV-reload copies, which run
  outside the busy window the gate watches; study 9 names the same copies as "a candidate, not
  measured".
- **Stock-stack defects found along the way**: LMCache saves KV synchronously inside the step
  in front of token #1 (fixed by the async-save patch); the stock prefill → DRAM → decode handoff
  races (26–28% partial KV hits; fixed by a router readiness marker); the LMCache store never
  evicts (store cap added); the 3B store's port occasionally refuses to bind (cause unknown;
  launcher falls back).

### 3.5 The KV-slot model of the decode engine (study 10)

Setup: the 8B split alone on GPU1, no MPS daemon (studies 5–9's baselines ran under MPS, so
their absolute numbers are not directly comparable); decode budget 24 / 26 / 28 GiB × 1–5
req/s; a record-only trace on the decode engine; one launch per cell. TPOT here is per token
(token 1 → last ÷ 127); earlier studies' client measured per streamed chunk, which differs by
< 1 ms where chunks match.

**Arithmetic.** KV grant = budget − 16.23 GiB (3,980 / 5,004 / 6,027 blocks of 16 tokens). A
request is admitted when its prompt's 392 blocks are free and grows to 400; so whole-request
slots c = ⌊blocks ÷ 400⌋ = 9 / 12 / 15, and the fractional remainder matters: at 24 GiB the 452
leftover blocks admit a 10th request that is evicted (to the head of the queue, reloading its
prefix) once the ten grow past the grant; at 26 / 28 GiB the remainder admits none.

**The queue is accounted for by slots + FCFS + residence.** A replay with the effective slot
count (10 / 12 / 15) and each request's *measured* residence reproduces the mean decode queue
wait within 0.1% at every 26 / 28 GiB cell and within 2% at 24 GiB; c ± 1 is off by 1.5–3×.
The report is explicit that this is "a re-simulation of a measured result, not a predictive
model" (the decision log calls it "partly circular"): it establishes that waiting needs nothing
beyond slots and residence, not that residence can be predicted. Per request, wait ≈ k ×
(residence ÷ c) where k is the number of running requests that must finish first (in the six
cells plotted, 1,258 of 1,258 requests with k = 0 waited ≤ 0.2 ms). Textbook M/G/c formulas
(Erlang C, Lee–Longton) fit only near ρ ≈ 0.9, because 300 arrivals are far from steady state.

**The grid** (compiled LMCache kernels; mean decode queue wait, ms; output tok/s in brackets):

| req/s | 1 | 1.5 | 2 | 2.5 | 3 | 3.5 | 4 | 4.5 | 5 |
|---|---|---|---|---|---|---|---|---|---|
| 24 GiB | 0 (128) | 0 (191) | 11 (255) | 143 (317) | 1,265 (367) | 6,852 (372) | 11,036 (377) | 14,453 (380) | 17,334 (378) |
| 26 GiB | 0 (128) | 0 (191) | 0 (255) | 38 (317) | 340 (378) | 2,565 (410) | 6,381 (418) | 9,998 (423) | 12,551 (429) |
| 28 GiB | 0 (128) | 0 (191) | 0 (255) | 1 (317) | 81 (379) | 894 (427) | 4,137 (441) | 7,184 (451) | 9,546 (459) |

- **Below the knee the budget does not matter**: 1–2 req/s give the same throughput, 0 ms queue
  and 14.5 ms TPOT at every budget.
- **Capacity** (queue < 50 ms): 2–2.5 / 2.5–3 / 2.5–3 req/s. **Saturation throughput**: 2.97 /
  3.35 / 3.58 req/s — **+50% slots buys +21%, +25% slots (26 → 28) buys +7%**.
- **Why sub-linear**: every admission copies 0.75 GiB from host RAM synchronously inside the
  engine thread (`start_load_kv`, ~115–120 ms, ~7 GB/s — half the link; the kernel is 2 ms per
  layer, the rest is LMCache's L1/store fetch and Python), and no running request gets a token
  meanwhile. Decode time per request ≈ 128 steps + (loads during its residence) × ~120 ms; with
  more slots more admissions happen during a residence (1.6 stalls at 1 req/s → 8.7 at 28 GiB /
  3 req/s), so residence grows with c: at 3 req/s 3.15 / 3.25 / 3.35 s. From the lightest to the
  heaviest cell decode time grows 1.44 s, of which larger batches explain 0.54 s and stalls
  0.90 s.
- **Same request, two budgets** ([figures/kv_slots_timeline.png](figures/kv_slots_timeline.png)): #255 at 3
  req/s waits 4,226 ms at 24 GiB (19 free blocks, all ten lanes taken, ~13 queued ahead) and
  750 ms at 28 GiB; its load (112 / 120 ms) and decode (3.1 / 3.0 s) are the same.
- **Duplicates**: with decode-side prefix caching off, 9–10 running requests span only 6.1–6.5
  distinct sessions — about a third of resident KV is a copy of a prefix already on the GPU
  (reported, not tested, because enabling it changes the workload).
- **Streaming is not an alternative to residency**: a decode step needs every layer's KV for
  every resident token; streaming moves 0.78 GiB per request per step — 58 ms × batch, i.e.
  4.9× the resident step at batch 1 and 52× at batch 16; prefetch depth beyond 1 changes
  nothing; at most ~6 of 32 layers could stream for free at batch 1.

**The compiled-kernel discovery.** LMCache 0.4.4's `c_ops.so` (built for torch 2.10) never
loaded on torch 2.9.1; LMCache silently fell back to a Python copy path in every LMCache study
(3–9) and in study 10's first pass.
Rebuilt from source: per-layer copy 24.1 → 1.95 ms, per-request copy 190 → 114 ms, stall share
of decode time 34% → 20% (each stall ~120 ms instead of ~200), saturation +25–30%, TPOT at
1 req/s 17 → 14.5 ms, TTFT at 24 GiB /
2.5 req/s 163 → 48 ms (the prefill engine's KV save uses the same kernels). The slot mechanism
is unchanged; every absolute number in the earlier studies' KV-load figures is an upper bound,
and part of their "~400 ms KV load" was an LMCache L1-staging-pool allocation thrash, not PCIe.

**Load-path levers** (what each changes: c, residence, or both):

| lever | effect | status |
|---|---|---|
| compiled LMCache kernels | copy 190 → 114 ms; saturation +25–30% | baseline |
| `use_layerwise` | no overlap (per-layer fetch and copy alternate; forward for layer *i* waits for copy *i*); on the fallback it cut queue 3× only by avoiding the L1-pool thrash | not useful here |
| `enable_async_loading` | store fetch moves before admission (~100 ms prefetch, no slot held); residence −5 to −16%; +7% saturation; no capacity on vLLM's queue time; crashes at 24 GiB ≥ 4 req/s (lookups compete for the 6 GiB staging pool) | marginal |
| **MP connector** (vLLM 0.18.1) | copy runs in a separate process over CUDA IPC while the batch decodes: stall 0.67 → 0.05 s per request; residence 1.99 s at 3 req/s at *every* budget (no longer grows with c); capacity +0.5–1 req/s on queue time (+1–1.5 on slot wait); throughput +40%; **but** evictions now *recompute* the 6144-token prefix (connector TODO), so a lent 16th slot at 28 GiB collapses it at 4.5–5 req/s (174 evictions, 303 tok/s) | best lever; not version-matched (0.18.1's grant is 0.5–0.6 GiB larger, step ~2 ms shorter) |
| fp8 KV | would halve bytes per block (≈ 20 slots at 24 GiB) | not run: fp8 KV "will likely move the engines off FLASH_ATTN" on sm_120, so a bf16 control on that other attention backend is needed first |
| decode-side prefix caching | would free the duplicated third | not tested (changes the workload) |

## 4. What was implemented (catalogue)

**Measurement and tracing.** CUPTI injection tracer and auto-profiler (`util_examples/cupti`);
CUPTI four-timestamp observer joined to an issue log (`colocation`); nsys/NVML/`%smid` recorder
and replay visualizer (`gpu-interfere/visualizer`, `colocation/visualizer`); DCGM per-second GPU
monitor (`gpu_monitor.py`), host monitor by process role (`host_monitor.py`), per-thread `/proc`
sampler (`thread_sampler.py`); PCIe calibration (`calibrate_pcie.py`); record-only decode-engine
trace via PYTHONPATH `sitecustomize` (`kv_vram/site/kv_trace.py`); timeline joiner and FCFS /
Erlang-C / Lee–Longton re-simulator (`kv_timeline.py`, `kv_mgc.py`); streaming-vs-resident
decode micro-benchmark (`bench_stream.py`).

**Scheduling mechanisms.** The Orion-style colocator (`libcolocator.so`, `libsched.so`,
`libobserver.so`; FCFS and throttle policies; stream priorities; green-context SM limits; cuBLAS
interposers); the resource-probe kernels (`probe.cu`, six probes); the idle-window gate
(`hp_patch`/`hp_signal.py` publishing a busy window from `execute_model` into `/dev/shm`) and
the CUPTI kernel gate (`cupti_gate.so`, credit scheme K/maxpend); MPS SM caps and MPS client
priority as launch options; `OMP_NUM_THREADS` capping in all multi-engine launchers.

**Serving stacks and drivers.** Case-A sweep harness with workload generator, open-loop client,
server lifecycle and stall watchdog (`scripts/inference/simple`); LoRA layer streamer with
depth-*k* prefetch and tail/interleaved placement (`layer_offload.py`); the hand-built P2P split
with four connector patches (`split_simple/p2p_patch`) and the shared-LMCache split with a
token-forwarding router (`lmc_proxy.py`) and an `SO_REUSEADDR`/capped store wrapper
(`lmc_server_main.py`); colocation drivers for a trainer neighbour (`inf_ft_coloc`), a second
model (`inf_inf_coloc`), N engines (`inf_multi_coloc`: `multi_launch.sh`, `multi_sweep.py`,
`probe_kv.sh`), a prefill neighbour (`inf_pf_coloc`, `pfc_launch.sh big|small`) and priority arms
(`inf_pf_prio`, `pfp_sweep.py --arm …`); the stripped decode-only stack (`kv_vram`:
`kv_launch.sh`, `kv_sweep.py`, methods `base/cops/lw/async/mp18`, compiled-kernel shim,
LMCache-0.4.4 MP connector re-export); the async-save patch (`--sm-async-save`) and the router
readiness marker; `mem_guard.sh` (kills engines when free RAM collapses) and heartbeat/stall
watchdogs on every stack.

**Reproductions and patches to others' code.** Tally sm_120 fixes
(`tally_local_changes.patch`); NanoFlow sm_120 port (arch flags, configurable KV pool,
fallback-path fixes, KV-page release, device-sized SM partitions, a timing entry script).

## 5. Discoveries that corrected earlier beliefs

| believed / predicted | found | where |
|---|---|---|
| Offload's floor is bounded by fetch time (`t_load` ≈ 85 ms) | the wall is admission: in-flight KV must be resident; TTFT reached 2–5 s with fetch at 56 ms | study 1 |
| 8192 tokens/step is the fine-tune design point | batch 4 OOMs on fp32 logits; 4096 tokens/step, below the ~6,900 break-even | study 2 |
| deeper prefetch should help at 12–16 layers | depth 2 does nothing: bandwidth-bound, not scheduling-bound | study 2 |
| a split's KV transfer is a fixed ~64 ms in TTFT | retrieval ~430 ms, and whether it lands in TTFT is router policy | study 4 |
| the decode node survives to ~8 GiB (as the split report describes the plan's expectation; `PLAN_DECODE.md` itself puts a batch-1 floor at 17.1 GiB) | floor ≈ 21 GiB = weights + in-flight KV for this concurrency | study 4 |
| host CPU would knee N identical models at N≈6–8 | GPU time knees latency; the host breaks first only via thread oversubscription, fixed for free | study 7 |
| delivered/offered ≈ capacity; SM-active ≈ load | delivered/offered reads 92–94% with zero queueing; SM-active fell while throughput doubled | study 7 |
| a prefill neighbour costs the 8B 40–80% per-token latency | per-token time unchanged; cost is capacity via residence (slots) | study 8 |
| "store the KV to DRAM" is free for TTFT | stock save is synchronous in front of token #1 (2.7× TTFT); the handoff races | study 8 |
| gating is a strong one-sided priority; MPS client priority is weak | the reverse: MPS priority protects the decode, the gate starves the prefill and helps the decode not at all | study 9 |
| slots = grant ÷ 0.78 GiB (8.7 / 11.2) | the scheduler holds whole requests: c = 8 / 11, and the 392-block admission rule can lend one more | studies 9 → 10 |
| capacity 24 / 26 / 28 GiB ≈ 2–2.5 / 2.5–3 / 3–3.5 req/s | fallback: one grid step lower at every budget; compiled kernels: as predicted at 24 / 26 GiB, one step short at 28 (2.5–3); sub-linear gain confirmed; the mechanism is the synchronous load stall | study 10 |
| `use_layerwise` overlaps copy with compute | it does not; its fallback gain was a staging-pool bug avoided | study 10 |
| LMCache's compiled ops cannot be used on this stack | the wheel targets torch 2.10; a source build works and halves the copy | study 10 |
| `pageable` PCIe ≤ 8 GB/s | 14.2 GB/s at ≥ 64 MiB | study 10 |

## 6. Limitations that apply across the project

- One model (Llama-3-8B), one request shape (6272 in / 128 out), one access pattern family
  (fixed 6144-token prefixes, zipf-1.1 or uniform), one output length. Everything about
  "slots" is clean because every request is the same 400 blocks; the project itself notes the
  128-token suffix makes relative gains an upper bound.
- Most cells are n = 1 (repeats where done agree within 1–3%; p50 TTFT ±15 ms). Capacity is
  resolved to the 0.5 req/s grid step.
- Consumer hardware: PCIe Gen4 ×8, no P2P, 60 GiB host RAM; the memory-bandwidth and
  copy-cost ratios would differ on NVLink/C2C-class machines, and host RAM bounded three
  colocation studies.
- Version pinning: vLLM 0.15.1 + lmcache 0.4.4 throughout; the MP-connector result mixes the
  connector with vLLM 0.18.1 version effects (no same-version control).
- All LMCache measurements in studies 3–9 ran on the Python fallback copy path; their KV-load
  costs are upper bounds (study 10 §5 quantifies the gap).
- Baselines differ: studies 5–9 ran under MPS, study 10 without; study 10 also redefined TPOT
  from per-chunk to per-token (< 1 ms difference where chunks match). Cross-study absolute
  comparisons are indicative only.
- Study 10's re-simulation uses each request's measured residence; it confirms the slot/FCFS
  structure but does not predict residence. Study 9's slot-utilization rule "is fitted on one
  8B, one workload and P2 only; the residence a neighbour adds has to be measured, not derived".
- Study 6's offload cells are swap-confounded; study 5's +45–70 ms TTFT offset under MPS caps
  is unattributed; study 10's MP-connector arm has no same-version control.
- Workload knowledge: every study assumes the workload shape is known and fixed; nothing was
  measured under varying prompt lengths, output lengths, reuse patterns or arrival rates within
  a run.

## 7. Where this points — the insight to build on

The project's most load-bearing result is the one studies 8, 9 and 10 converge on (study 4
found the in-flight-KV wall they explain):

> Under KV offload, below a load-dependent knee the decode engine's performance does not depend
> on its VRAM budget at all; above the knee it depends on the budget only through the number of
> whole requests the KV grant admits. The engine behaves as *c* integer KV slots served FCFS,
> where *c* is set by the admission rule (⌊grant ÷ per-request blocks⌋, plus one when the
> remainder exceeds a prompt — a lent slot that then evicts), and the queue is governed by how
> offered rate × residence compares with *c*. Residence R is not fixed: on the stock load path
> it grows with every synchronous KV load that happens while a request is resident
> (R ≈ R₀ + n_loads × t_stall), which is why more slots give diminishing returns (+50% slots,
> 24 → 28 GiB, → +21% saturation throughput) and why taking the load off the critical path made
> residence independent of the slot count.

What in that statement is measured on this box, and with what caveats:

- The integer-slot / FCFS structure: a replay with measured residence matches the queue within
  0.1–2%, c ± 1 does not. This is a consistency check (it uses each request's measured
  residence), not a predictive model.
- The residence inflation by synchronous loads: 0.90 s of the 1.44 s growth across the grid is
  load stalls, 0.54 s is batch size; removing the stall (MP connector) held residence at 1.99 s
  at every budget — one arm, on a different vLLM version, with no same-version control, and with
  a collapse at 28 GiB when evictions recompute.
- The slot-utilization curve across neighbours (study 9) — fitted on one model, one workload,
  one placement, under MPS, with fractional slot counts (8.7 / 11.2) later corrected to integer
  (8 / 11); study 10 ran without MPS. Textbook ρ = λR/c formulas fit only near ρ ≈ 0.9 in a
  300-request window.

What is not yet formulated:

- (a) A per-request KV footprint that varies. Every slot here is 400 blocks; with mixed prompt
  and output lengths *c* becomes a bin-packing quantity and the lent-slot / eviction effect
  becomes the common case rather than the edge case.
- (b) A *locality* term. A third of resident KV was a duplicate prefix (reported, not tested —
  decode-side prefix caching was left off because enabling it changes the workload); how
  prefix sharing across resident requests (and, not studied here, the prefill side's hit rate)
  changes the effective footprint and the load cost is unmeasured.
- (c) A *load-path* term under contention: fetch and copy costs that depend on store depth,
  staging-pool pressure and neighbours (at 26 / 28 GiB and 4–5 req/s the async arm's prefetch
  itself queued for 1.4–3.9 s — the store fetch becoming the next bottleneck; at 24 GiB it
  crashed).
- (d) The host-side constraints the colocation studies found bind first — CPU threads, host
  RAM, pinned-memory budgets, MPS as a precondition — which any policy that lends decode VRAM
  to a neighbour has to respect.

And the methodological gap the project itself keeps naming: every experiment used a known,
fixed workload. A real deployment does not announce its prefix lengths, reuse pattern, output
lengths or arrival rate, and they change within minutes. The experiments' tools — per-request
engine traces, slot accounting from the block grant, queue-wait as the capacity signal, the
re-simulation harness — are the instruments such a system would need to run *online*; the
policy that uses them (how much VRAM to hold for KV vs lend, when to admit, what to evict,
where to fetch from) was never written. That is the open design space these two reports hand
over.

---

## Appendix A — Where the data lives (per branch, on `main`)

| branch | reports | plans / decision logs | data |
|---|---|---|---|
| `util_examples` | per-directory READMEs | — | `cupti/output/*.json`, `*.csv`, DCGM/NVML CSVs |
| `gpu-interfere` | `EXPERIMENTS_SUMMARY.md`, `section_4.{1,2,3}/report.md`, `profiler/reports/interference_report.md` | `CLAUDE.md`, `visualizer/PROPOSAL.md` | `section_*/results/*.csv`, `visualizer/recordings/` |
| `colocation` | `README.md`, `SUMMARY.md` | `PROPOSAL.md`, `CLAUDE.md` | `runs/`, `replays/` (gitignored, workstation only) |
| `literature-review` | `LITERATURE_REVIEW.md`, `interesting work/LITERATURE_REVIEW.md`, `reproduce/{tally,nano}-reproduce.md` | `reproduce/README.md` | `reproduce/results/`, `reproduce/tally_local_changes.patch` |
| `vram-limit` | `reports/report_*.md` (10 files) | `PLAN*.md` (6), `DECISIONS_*.md` (3), `PROMPT_KVCACHE_VRAM.md` | `output/summary_*.csv`, `output/*_sweep.json`, `output/gpumon/`, `figures/`; `output/raw/` and `output/logs/` gitignored |
| `kvcache-vram_size` | `reports/report_kv_slots.md`, `reports/report_kv_slots_fallback.md` | `NOTES_KV_SLOTS.md`, `PLAN_KV_VRAM.md`, `DECISIONS_KV_VRAM.md` | `output/summary_kv.csv`, `output/kv_vram/{trace,timeline,mgc.csv,async_summary.csv,related_work_notes.md}`, `figures/`, `figures/slides/` |

## Appendix B — Discrepancies found while fact-checking the branch documents

Collected so that the reports above are not read as more certain than their sources.

1. Study 1's headline table gives GPU-hit rates of 46 / 37 / 28% at 20 / 19 / 18 GiB; the
   summary CSVs give 9.3 / 3.6 / 2.5%. The DRAM-hit column matches the CSVs; treat the GPU-hit
   values at ≤ 24 GiB as unreliable.
2. Native fetch of a 0.75 GiB prefix is quoted as 56 ms (study 1) and ~65 ms (studies 3, 4).
3. Study 3 says "zero preemptions in all 16 configs"; `uniform_b24_offload_lmcache` records 3.
4. Study 4 says "zero preemptions at every point" for its five tabulated budgets; the untabulated
   24 GiB runs show 11–13 (acknowledged in `PLAN_DECODE.md` as an admission-arithmetic pocket).
   Its 18 GiB TPOT is 534.7 ms in the table and "524 ms" in the text; its "105 vs 56 ms" headline
   uses a max-inflight-28 run while the table's 30 GiB row (110 ms) uses max-inflight 999. It
   also says "a prediction from the plan" expected the decode node to survive to ~8 GiB;
   `PLAN_DECODE.md` contains no such figure (its batch-1 floor is 17.1 GiB).
5. Study 2's limitations say depth-2 prefetch was untested while §2 reports it; its per-layer
   size is 0.467 GiB in the plan and 0.406 GiB in the report (the plan's 14.96 GiB includes
   embeddings the layer knob does not stream).
6. Study 9's report counts ~8.7 / 11.2 slots; the next study's notes correct this to integer
   c = 8 / 11 (engine logs: Running never exceeds 8 at 23 GiB, 11 at 25). `PLAN_PF_PRIO.md`'s
   outcome bullet says config A ran at 23.5 GiB; the decision log and report show 23 GiB. The
   log and plan call the report's config C "config B".
7. Study 10: `CLAUDE.md` still says LMCache's compiled ops "never load against torch 2.9.1 — a
   source build deadlocks"; this branch's source build did not deadlock and is the main report's
   baseline. Fallback saturation is "2.3 / 2.65 / 2.7" in one place and "2.38 / 2.66 / 2.75" in
   another (the latter labelled "rates ≤ 4"). Capacity deltas for the async and MP arms are
   quoted on two bases (vLLM queue time vs slot wait) in different documents; both are stated.
8. `gpu-interfere`: L1 worst case 5.741× measured, "6.0×" in summaries; the profiler report's
   pair table says 138% / 153% while its conclusion says 137% / 150%; §4.1.1's scope is labelled
   "GPU-wide sched → per-SM slots" in one file and "per-SM" in another.
9. `colocation`: 8 vs 9 workloads and 28 vs 36 pairs across documents (36 = 8 workloads with
   self-pairs; `llmdecode` is the ninth); the visualizer README predates cuBLAS interception and
   calls `llmdecode` non-colocatable; PyTorch 2.8 (proposal) vs 2.11 (summary).
10. `literature-review`: the review lists SMore's code as available; `reproduce/README.md` says
    no repository was found. The Tally report says "three fixes" in its verdict and lists four.
11. `util_examples`: `overhead_compare.csv` records the tracer run at 23.397 s vs 23.407 s in
    the JSON; profiling cycle 3's metrics are NaN and `num_replay_passes` is 1 everywhere.
12. Studies 5 and 6 have no plan or decision log and no dates in their documents; study 3 has
    no plan of its own (LMCache is an optional arm in study 1's plan); studies 1–6 are dated
    here only by their first commits.
