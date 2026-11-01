# The CUDA-scheduling project, in the order it happened

**Repository:** `852866031/cuda-schedule-project` — branches `util_examples`, `gpu-interfere`,
`colocation`, `literature-review`, `vram-limit`, `kvcache-vram_size` (all merged into `main`
under `branches/<name>/` on 2026-11-01).
**Machine throughout:** one workstation with 2× RTX 5090 (32 GB each, sm_120, 170 SMs), PCIe
Gen4 ×8 to the host (14.47 GB/s host-to-device, pinned, measured), no GPU peer-to-peer (GeForce), 32 cores,
60 GiB RAM, no `sudo`/`ptrace`/`perf`.
**How the dates were established.** Git commit dates, cross-checked against dates written in the
documents (plans, decision logs, `SUMMARY.md`). Studies 1–6 have no dates inside their documents;
their dates below are the commits that first added each report. Where a study was written up
days after its runs, both are given. Discrepancies between the branch documents and their data
found while writing are listed in [REPORT_SUMMARY.md](REPORT_SUMMARY.md) Appendix B; terms
(slot, residence, knee, MPS, split, fallback, ITL vs TPOT) are defined in its §2.

---

## Timeline at a glance

| when (2026) | branch | what | one-line result |
|---|---|---|---|
| Mar 18 – Jul 23 | `util_examples` | GPU-utilization demos: NVML → DCGM → Nsight Systems → Nsight Compute → CUPTI tracer/profiler | "GPU-Util" is time-with-a-kernel-resident, not work done; each tool answers a different question |
| Jul 14 – 23 | `gpu-interfere` | Reproduce "Understanding GPU Resource Interference One Level Deeper" (§4.1–4.3) on the RTX 5090 | Interference depends on *which* shared resource and at what scale it is shared; one utilization number cannot predict it |
| Jul 27 – Aug 6 | `colocation` | Orion-style kernel colocator prototype (LD_PRELOAD interception incl. cuBLAS, per-client queues, FCFS/throttle, CUPTI observer, Llama-3-8B decode client) | Two streams alone give no concurrency against an SM-saturating kernel (45× p50); stream priority restores it (0.98 ms p50, −2% throughput, 89% overlap) |
| Aug 6 – 17 | `literature-review` | Review of 7 GPU-sharing systems against Orion + 4 LLM-serving papers; Tally and NanoFlow reproduced on sm_120 (Aug 11) | Generic kernel schedulers lack prefill/decode semantics; SM partitioning does not partition HBM/L2; Tally: high-priority p99 8.4× better under colocation |
| ≤ Aug 18 | `vram-limit` study 1 | vLLM inference, KV cache oversubscribed, DRAM offload, budget 30 → 18 GiB | 30 → 22 GiB (−58% KV) costs 9% TTFT p95; then two-step collapse. Wall = **concurrency** (in-flight KV), not cache or PCIe |
| ≤ Aug 18 | `vram-limit` study 2 | LoRA fine-tune with base weights streamed from DRAM | 3.25 GiB freed for 1.4% throughput with the best prefetch variant, 30% without; wall = **PCIe bandwidth**, smooth |
| Aug 25 | `vram-limit` study 3 | Same inference sweep with LMCache as the DRAM tier | Same cliff at 19–20 GiB — the wall is the workload's; below it LMCache queues instead of stalling |
| Aug 25 – 26 | `vram-limit` study 4 | Prefill/decode split across the two GPUs, decode VRAM swept | TTFT flat (74–111 ms) at every budget; wall at 20–22 GiB = **in-flight KV**; prefill GPU 97% idle; decode GPU DRAM-bound |
| Aug 26 – Sep 4 | `vram-limit` study 5 | LoRA trainer colocated on the decode GPU | Time-slicing collapses decode (TPOT 28 → 1193 ms); MPS (NVIDIA's Multi-Process Service, which lets kernels of several processes co-reside on the GPU) with a 10% SM cap: +23% TPOT; idle-window gate restores TTFT |
| Sep 4 – Oct 1 | `vram-limit` study 6 | Second (0.5B) decode model colocated with the 8B decode | Without MPS two decodes time-slice: 8B TPOT 28 → 95–98 ms, −25% capacity; MPS undoes it (34 ms, −8%) |
| Oct 1 – 13 | `vram-limit` study 7 | N identical engines (0.5B: N=1–8; 3B: N=1–4) under MPS | Colocation costs per-token latency (lost batching), not capacity at the loads swept — once each engine's CPU threads are capped (the 3B's per-engine sequence cap does bind under load); **host CPU threads** break first (4 offloading engines collapse; `OMP_NUM_THREADS` cap fixes it) |
| Oct 13 – 17 | `vram-limit` study 8 | A 3B prefill on the 8B's decode GPU | 8B at its knee loses capacity (−19% at 2 req/s) with per-token time unchanged; LMCache's synchronous KV save costs the 3B 2.7× TTFT; async-save patch |
| Oct 17 – 19 | `vram-limit` study 9 | Priority between the 8B decode and the 3B prefill (MPS priority vs CUPTI gate) | All cells fall on one curve: 8B queue vs **slot utilization** = rate × residence ÷ slots. MPS client priority protects the 8B; the launch gate does not |
| Oct 25 – 26 | `kvcache-vram_size` study 10 | Decode engine alone: KV slots × request rate × VRAM budget grid | Decode = bank of **integer KV slots**, FCFS; +50% slots buys +21% saturation because every synchronous KV load stalls the batch; LMCache's compiled kernels had never loaded in any LMCache study (3–9); MP connector removes the stall |
| Nov 1 | `main` | All branches merged under `branches/` | — |

---

## Phase 0 — What does "GPU utilization" measure? (`util_examples`, 2026-03-18 → 07-23)

The project begins before any scheduling: a set of small experiments asking what the
utilization numbers people read actually mean. The repository is organised as a diagnosis
path — *detect* (NVML) → *monitor* (DCGM) → *locate* (Nsight Systems) → *explain* (Nsight
Compute) → *automate* (CUPTI).

What was built and shown:

- `cpu_util/`: a CPU-utilization baseline for the concept.
- `nvml/two_cases.cu`: two workloads with the same NVML "GPU-Util" and very different work
  done — GPU-Util is "the fraction of time *some* kernel was resident".
- `DCGM/`: traces of `SM_ACTIVE`, `SM_OCCUPANCY`, `TENSOR_ACTIVE`, `DRAM_ACTIVE` on a real
  workload (install notes for Ubuntu 24.04 included).
- `nsight_compute/`: a two-GPU kernel experiment showing similar aggregate activity with very
  different kernel efficiency.
- `nsight_systems/`: a two-GPU inference-pipeline example (`inf_sys.py`) showing similar GPU
  activity with different pipeline throughput/latency.
- `cupti/` (added Jul 23, the last commits on the branch): two injection libraries that attach
  to an unmodified CUDA process via `CUDA_INJECTION64_PATH` — a **tracer** (per-kernel launch
  counts and GPU time into CSV hotspot tables) and an **auto-profiler** that alternates
  tracing and CUPTI Profiler-API hardware-counter collection (AutoRange + KernelReplay) on
  the hottest kernel — tested on a Llama-3-8B inference workload (256 requests, 8192
  generated tokens). Stored end-to-end numbers: baseline 357.8 tok/s; under the tracer
  350.0 tok/s (+2.2% wall time); under the profiler 313.8 tok/s (+14% wall time). The hottest
  kernel in every profiling cycle was a cutlass wmma f16 GEMM (~50 µs, 63–93 k launches per
  10 s window). No dates appear anywhere in this branch; the dates above are commit dates.

Why it matters later: the CUPTI injection work is the direct ancestor of the colocator's
observer (Phase 2) and of the CUPTI kernel gate used in studies 5, 6 and 9. The DCGM metrics
(`SM_ACTIVE`, `SM_OCCUPANCY`, `DRAM_ACTIVE`, later tensor-pipe active) are the telemetry every
later study reads.

## Phase 1 — Reproducing GPU interference "one level deeper" (`gpu-interfere`, 2026-07-14 → 07-23)

A reproduction of the paper *Understanding GPU Resource Interference One Level Deeper*
(Elvinger, Strati, Jerger, Klimovic; SoCC '25, arXiv 2501.16909; the paper's hardware is an
H100 NVL and an RTX 3090) — its §4.1–§4.3 experiments on the RTX 5090 (170 SMs, ~96 MB L2,
128 KB L1/SM, ~1.2 TB/s achieved DRAM bandwidth) — plus a colocation-interference profiler
and a browser visualizer. Every experiment colocates two kernels and compares wall-clock
latency (median of 10 runs) in three modes, *alone / sequential / colocated*: colocated ≈ alone
means no interference, colocated ≈ sequential means full interference. What differs between
experiments is the shared resource.

Results, ordered by the scale at which the resource is shared (`EXPERIMENTS_SUMMARY.md`; the L1
worst case is 5.74× in the measured table, rounded to "6.0×" in the summaries):

| scope | experiment | shared resource | copies on the 5090 | worst slowdown | mechanism |
|---|---|---|---|---|---|
| GPU-wide | §4.1.3 memory bandwidth | DRAM bandwidth | 1 | each kernel → 60% (aggregate capped at peak) | fixed bandwidth budget split — even on separate SMs |
| GPU-wide | §4.1.2 L2 cache | L2 (~96 MB) | 1 | 5.6× | capacity cliff when the combined footprint exceeds L2 |
| GPU-wide sched → per-SM slots | §4.1.1 thread-block scheduler | per-SM thread/warp/block slots | 170 | 2.0× | serialization once each kernel fills an SM's slots |
| per-SM | §4.2.1 L1 cache | L1 (128 KB/SM) | 170 | 6.0× | capacity cliff |
| per-SMSP (SM sub-partition) | §4.2.2 warp scheduler | issue slots (4/SM) | 680 | 1.4× | issue-slot contention |
| per-SMSP | §4.2.3 compute pipeline | FP64 pipeline | 680 | 2.0× | pipeline serialization |
| per-SMSP (real kernel) | §4.3 ML matmul | FP32 FMA pipeline + warp scheduler | 680 | 5.6× | the §4.2 mechanisms on a production `torch.mm` |

The conclusion recorded on the branch, in its words: whether two kernels interfere "depends
entirely on *which* of these resources they jointly stress, and at *what scale* it is shared"; a
single whole-GPU utilization number "collapses all six rows into one and therefore cannot predict
any of it". Two mechanisms recur at several scales: the **capacity cliff** (L2 and L1) and
**serialization** (block scheduler and pipeline). GPU-wide resources cannot be escaped by
partitioning SMs (so MPS cannot help there); per-SM and per-SMSP contention can.

The §4.3 result is the one most relevant to serving: a production `torch.mm` colocated with a
compute-bound FMA kernel slows 1.4× / **5.6×** / 4.3× / 1.1× / 1.1× at 256³ / 512³ /
1024³ / 2048³ / 4096³ — worst for mid-size, compute-heavy matmuls that launch too few blocks
(≲ 170) to dominate the FMA pipe; large matmuls are safe because they dominate the pipeline by
block count, not because they become memory-bound. The branch compares mechanisms, not
numbers, with the paper: "the *mechanisms* reproduce exactly; the *thresholds* differ with the
hardware" (50 MB vs 96 MB L2, 3.3 vs 1.2 TB/s, strong vs crippled FP64).

Two tools came out of this phase. A **profiler** with six **resource probes** (`sleep`,
`dram`, `l2`, `l1`, `fma`, `fp64` — one kernel per shared resource, one block per SM) that
(Part 1) profiles each kernel alone with Nsight Compute (NCU) for % of peak on seven resources and (Part 2) directly
colocates every pair; a combined-demand rule (A% + B% > 100%) predicted 16 of 21 pairs and
missed five — all under-predictions caused by cache *capacity* (invisible to throughput
counters: `l1+l1` 2.50×) and shared FP datapath/issue-slot effects (`fma+fp64` 2.28×, `l2+fma`
1.59×). Conclusion recorded: per-kernel rate counters predict same-resource contention but
direct colocation measurement is still required. A **visualizer** records nsys GPU-metric
sampling + NVML + per-block `%smid` traces and replays them on an architecture-shaped GUI
(the recorded Llama-3 prefill run's `meta.json` logs 265.9 ms unprofiled vs 274.4 ms profiled
on the compute region, +3.2%); an NVBit feasibility check (2026-07-21) found
NVBit instruments compiled binaries exactly but produces zero events inside any Python process
on driver 580, so per-SM traces of opaque PyTorch kernels were deferred. The six probes were
ported into the colocator demo (Phase 2) as workloads.

## Phase 2 — An Orion-style kernel colocator (`colocation`, verified 2026-07-27, committed 08-06)

The first scheduling artefact: a minimal reimplementation of Orion's mechanism (EuroSys '24) —
intercept the CUDA runtime calls of unmodified PyTorch programs with `LD_PRELOAD`, buffer them in
per-client software queues, and re-issue them on colocator-owned CUDA streams under a pluggable
policy (FCFS shipped) — plus what Orion lacks: a passive **CUPTI observer** that records four
timestamps per operation (intercept → issue → GPU start → GPU end) on one clock, an analysis
pipeline, and a browser replay visualizer.

Built: three C++ shared libraries — the LD_PRELOAD interposer (`libcolocator.so`; the calling
thread's TID decides whether a call is queued or passed through), the scheduler (`libsched.so`;
one priority stream per client, a pluggable `Policy.pick_next()`, sync ops completed by polling
rather than blocking) and the CUPTI observer (`libobserver.so`) — plus a custom tiled-SGEMM
torch extension (stock `torch.matmul` goes through cuBLAS and would bypass `cudaLaunchKernel`
interception) and two client workloads: `latency` (small SGEMM chains, 0.46 ms/iter) and
`throughput` (one 4096³ SGEMM saturating all 170 SMs, ~21 ms/iter).

Results (RTX 5090, 30 iterations per client, FCFS):

| config | latency p50 / p95 | throughput iters/s | kernel overlap |
|---|---|---|---|
| solo | 0.46 / 0.47 ms | 48.3 | — |
| plain two streams, no colocator | 20.8 / 21.7 ms | 47.4 | ~0 |
| colocated FCFS, priorities 0,0 | 20.8 / 21.7 ms | 47.6 | 42% of A's busy time |
| colocated FCFS, priorities **−5**,0 | **0.98 / 5.2 ms** | 46.9 | **89%** |

Findings the four-timestamp trace made provable: (1) two streams alone buy nothing when one
client saturates the GPU — the latency client's delay is entirely hardware-side (`launch →
gpu_start` p50 ≈ 20 ms behind the SGEMM's 16 K blocks; software queue delay 0.2 µs); (2) stream
priority restores latency at ~2% throughput cost, with real SM sharing; (3) interference is
measurable per kernel class (elementwise 1.5 → 3.1 µs, SGEMM 90 → 112 µs while overlapping);
(4) colocator overhead is negligible (queue delay p95 ≤ 1 µs); (5) two scheduler hazards found by
measurement — blocking sync ops stalled the loop, and D2H copies into *pageable* memory make
`cudaMemcpyAsync` silently synchronous, stalling the scheduler and coupling both clients.
Cross-validated with Nsight Systems (kernel counts match exactly).

Limitations recorded on 07-27: cuBLAS/cuDNN not interposed (so stock `torch.matmul` would
bypass the colocator), inference-only clients, single GPU, queue depth 1, FCFS only.

**State by 2026-08-04** (`CLAUDE.md` on the branch, written after the summary): all phases of
the proposal implemented "including cuBLAS API interception" (`cublasGemmEx`,
`cublasSgemm_v2`); a real **Llama-3-8B decode client** (`llmdecode`) runs through the colocator
(solo 8.6 ms/pass → managed alone 8.9 ms → paired with the `l2` probe 10–17 ms with 38% of its
kernel-busy time overlapped); a **throttle policy** that caps in-flight ops (cap 2/4: 50/25
ms/pass for the pair, because the cap empties a ~2,600-op hardware backlog — but alone it costs
27.0/18.5/15.3 ms/pass at caps 2/4/8, i.e. ~30 µs completion-detection latency per stall);
**green-context SM partitioning** (`COLOCATOR_SM_LIMIT`; a bandwidth-bound GEMV loses only ~5%
on 88 of 170 SMs); and full 36-pair matrices of the 8 workloads (2 clients + 6 probes) with and
without throttle, whose self-pairs reproduce the interference ladder through the colocator
(fma+fma 1.8×, dram+dram 2.0×, l2+l2 1.8×, sleep+sleep ~free; the gpu-interfere profiler
measured 1.60× / 1.98× / 2.01× / 1.00× for the same pairs). The interference-aware *policy*
Orion has was not written; the mechanism and the measurement were. The `runs/` and `replays/`
data for this branch are gitignored and exist only on the workstation.

## Phase 3 — Literature review and two reproductions (`literature-review`, 2026-08-06 → 08-17)

Two review documents were written. The top-level one (directory originally called
`orion-citation`) reviews seven GPU-sharing systems against Orion (EuroSys '24) as the
reference point — eight papers in all: Hummingbird (arXiv 2026), Tally (ASPLOS '25), Bless
(EuroSys '25), LithOS (SOSP '25), MMK (ACM TACO 2026), SMore (IEEE TPDS 2025), Usher (OSDI '24).
Each is classified by scheduling object, control layer, interception, kernel transformation and
fit to Orion-style scheduling. A second document (`interesting work/`) reviews four LLM-serving
systems: NanoFlow, ShadowServe, MuxWise, DuetServe. Two more PDFs (BlendServe, Parallax) sit in
a `not relevant/` folder on the branch and are discussed nowhere.

The reviews' own synthesis, which the later experiments keep returning to:

- "**SM isolation does not isolate shared resources**" — L2, HBM bandwidth, copy engines,
  PCIe/NVLink remain shared (the same point Phase 1 measured).
- "For LLM systems specifically, the most important extension is **phase awareness**" — generic
  kernel schedulers intercept prefill/decode kernels but "do not automatically understand
  TTFT/TPOT semantics". Of the GPU-sharing systems, only Hummingbird and LithOS evaluate LLMs on
  real serving traces (BurstGPT, an Azure trace); Tally, MMK and Usher include LLMs as generic
  models; none schedules by phase or KV state.
- "**Data movement should be scheduled with computation**" (from ShadowServe/NanoFlow): KV fetch
  can dominate TTFT and interfere with TPOT although it is outside the Transformer graph.

Reproductions (both dated 2026-08-11, both on the 2× RTX 5090 box, both requiring sm_120 porting):

- **Tally** (prebuilt container; four fixes: compute-capability list `{"90","86","80"}` → add
  `"120"`/`"89"`, mount host CUDA 12.8 `nvcc`, pass the shipped `roudi_config.toml` with
  `--shm-size=16g`, pass `cuGetExportTable` through because driver 580's NVML calls it). A
  micro-benchmark — a high-priority (HP) small MLP vs a best-effort (BE) stream of back-to-back
  4096³ matmuls — gave HP p99 2.103 ms under the
  hardware scheduler vs **0.250 ms under Tally PRIORITY (8.4×)**, with BE throughput unchanged
  (154.65 vs 154.57 it/s). Caveat recorded: the BE cuBLAS kernels were only launch-gated, not
  transformed into preemptible form. The source fixes survive as
  `reproduce/tally_local_changes.patch` (identical to `reproduce/results/tally_sm120_fixes.patch`).
- **NanoFlow** (`Nanoflow-python` branch, vendored): Llama-3-8B at **12,401 tok/s/GPU** vs the
  paper's 12,756 on one A100-80GB, after ~10 porting fixes (sm_120 arch flags, a hard-coded
  106 GB KV pool, a missing auto-search JSON, two crashing fallback-path bugs, a KV-page leak,
  H100-sized SM partitions). The paper's green-context SM-partition overlap **lost 14–32%** in
  every configuration tried, attributed to the small decode share the 32 GB card forces.

## Phase 4 — Studies 1 and 2: how far can VRAM shrink before DRAM offload stops saving you? (`vram-limit`, committed 2026-08-18)

The `vram-limit` branch opens the question that the rest of the project pursues: an LLM
workload with a ~39 GiB demand on a 32 GB card, with host DRAM absorbing the overflow over
PCIe Gen4 ×8. Two cases were pre-registered in `PLAN.md`/`PLAN_FINETUNE.md` with predictions
that the reports then grade.

**The reference workload** is fixed here and never changes afterwards: Llama-3-8B-Instruct
(fp16/bf16, 0.125 MiB of KV per token), **32 sessions × 6144-token prefixes = exactly 24.0 GiB
of KV**, each request = one session's prefix + 128 unique tokens, 128 output tokens, sessions
drawn zipf-1.1 (90% of traffic on 17 sessions) or uniform, open-loop Poisson at 2 req/s, 300
measured requests after a warmup, fixed seed, prompts sent as token IDs so prefixes are
block-aligned. `--gpu-memory-utilization = budget / 31.35 GiB`; weights and overhead never
yield, so every GiB removed comes out of the KV cache.

**Study 1 — inference with vLLM's native DRAM offload** (`report_simple_inference.md`). Offload
here is a second-level *prefix cache*, not paging: a decoding sequence still needs all its KV in
VRAM. Constants measured: fetching a 6144-token prefix (0.75 GiB) from DRAM 56 ms vs recomputing
it 530 ms (9.5×). Headline (zipf, offload arm):

| budget GiB | GPU KV GiB | TTFT p50 / p95 ms | preemptions |
|---|---|---|---|
| 30 | 13.77 | 54 / 305 | 0 |
| 24 | 7.77 | 59 / 315 | 0 |
| 22 | 5.77 | 73 / 332 | 0 |
| 20 | 3.77 | 135 / 1528 | 4 |
| 19 | 2.77 | 2078 / 5451 | 7 |
| 18 | 1.77 | 4856 / 10666 | engine stalled |

Conclusions as written: "You can cut the VRAM budget from 30 GiB to 22 GiB — a 58% reduction in
KV cache — for a 9% cost in tail latency. Below that it collapses, by two orders of magnitude,
within two steps." The collapse "is **not** caused by running out of cache, and **not** by
running out of PCIe bandwidth. It is caused by running out of room to hold *concurrently
running requests*" — a threshold computed in advance (Poisson p95 of 7 in flight × 0.766 GiB =
5.36 GiB) and bracketed by the data (5.77 clean, 3.77 broken). Four "walls" were computed from
measured constants: cache capacity (hit the whole sweep), concurrency (hit at 20 GiB), PCIe
bandwidth (5.7% used), prefill compute (not hit). Offload's value depends on locality — 1.1×
(zipf) vs 7.7× (uniform) at full VRAM — but the knee's location does not. Biggest stated
limitation: at a fixed 2 req/s open loop, throughput is a conservation law (254.7 tok/s in every
healthy config), not a dependent variable. Pre-registered predictions: flat-then-knee right;
"floor bounded by `t_load`" wrong (the wall is admission, not fetch cost). The 18 GiB offload arm
stalled deterministically twice at a byte-identical engine state.

**Study 2 — LoRA fine-tuning with frozen base weights streamed from DRAM**
(`report_finetune.md`). LoRA r=16 on q,k,v,o (13.6 M params, 208 MB of state), gradient
checkpointing mandatory, 2 × 2048 tokens/step (batch 4 OOMs on the fp32 logits), the last *N*
of 32 layers' weights held in pinned DRAM and crossing PCIe twice per step. Fetch-on-demand is
exactly additive (step = 1.144 s + GiB × 2 ÷ 14.47 GB/s, within 0.4%) — i.e. zero overlap in the
naive implementation. With the best prefetching variant at each point: 4 / 8 / 12 / 16 layers
(1.63 / 3.25 / 4.88 / 6.50 GiB freed) cost 0.7 / 1.4 / 5.6 / 12.7% of throughput, vs 17 / 30 /
39 / 46% without prefetch (plain depth-1 tail prefetch alone: 0.7 / 1.6 / 7.8 / 15.3%). Depth-2
prefetch and interleaved placement move the result by ≤ 3% ("prefetching is the whole story; the
elaborations are rounding errors" — "bandwidth-bound, not scheduling-bound"). `accelerate`'s `device_map` offload cannot train (meta-device backward), so a
custom layer streamer (`layer_offload.py`) was written. The report's comparison table is the
project's first general lesson: *the same offloading idea hits opposite walls* — inference reads
KV once per reuse and hits a concurrency/capacity cliff; training reads weights every step and
degrades smoothly against PCIe bandwidth. "If you must run out of memory somewhere, training is
the better place."

## Phase 5 — Study 3: the same sweep with LMCache as the DRAM tier (`vram-limit`, 2026-08-25)

One variable changed: `LMCacheConnectorV1` + lmcache 0.4.4 (256-token chunks, token-hash keys,
`PYTHONHASHSEED=0` required) instead of vLLM's native `OffloadingConnector` (16-token blocks).
A caveat that turns out to matter for every later study was recorded here: lmcache 0.4.4's
compiled CUDA ops do not load against torch 2.9.1, so it ran on its Python fallback (fetch of a
0.75 GiB prefix ~151 ms vs ~65 ms native).

Result: "the wall does not move, but the failure mode changes sides." Both backends break at
19–20 GiB — "the wall belongs to the workload, not to the cache" (decoding must read KV from VRAM
every step; at 19 GiB the GPU holds ~3.5 requests against 5+ needed). Above the wall LMCache's
slower fetch raises uniform medians ~1.5× but coarser (chunk-level) eviction avoids
"Swiss-cheesed" prefixes, so it covers 98% of requests vs native's 93–95% and tightens p95
(321–455 vs 594–606 ms uniform). Below the wall LMCache **never preempted and never stalled**: at
18 GiB it still delivered 143–172 tok/s with every request completing — "Overload became a deep
queue instead of a death spiral." Environment facts fixed here: lmcache 0.5.x fails to import
against vLLM 0.15.1; pip upgrades `transformers` to 5.x during install.

## Phase 6 — Study 4: a prefill/decode split on two GPUs (`vram-limit`, planned 08-25, reported 08-25/26)

The first use of both GPUs: GPU0 prefill only, GPU1 decode only, GPU1's budget swept 30 → 18 GiB.
The report is "two acts".

**Act I — the hand-built split with `P2pNcclConnector`, abandoned.** Made to work with four
patches loaded via a PYTHONPATH `sitecustomize` (request-id agreement — vLLM rewrites ids so
producer and consumer "could never agree on a key"; a deadline on the consumer's unbounded wait;
the D2H spill copy on its own stream, +31% decode throughput; backpressure on the send queue).
This phase also produced the machine's hardest lesson: the connector pins **32 GiB of host memory
per instance** by default; two instances on a 60 GiB box is not an OOM kill but a **hard
freeze** — the workstation rebooted once. `mem_guard.sh` and budget-checking launch scripts date
from here. Two structural problems ended the arm: with DRAM for decode only, prefill recomputes
(~31% of requests paid ~530 ms); and sharing one pinned pool across processes "is not a patch —
it is building a KV-cache service."

**Act II — the split over a shared LMCache server** (prefill stores once, decode retrieves per
request; decode VRAM is working memory, not a cache). A methodology finding changed how split
TTFT is measured: demo-style routers discard the token prefill produces and report "time until
the decode node has re-materialized the KV" (~580 ms); the fixed router forwards token #1 (**all later
studies require `--forward-first-token` and no admission cap**).

| decode budget | decode KV | TTFT p50 / p95 | tok/s | e2e p50 | TPOT |
|---|---|---|---|---|---|
| 30 | 13.77 | 110 / 570 ms | 254.4 | 3.73 s | 27.7 ms |
| 26 | 9.77 | 93 / 544 | 254.4 | 3.79 s | 27.9 |
| 22 | 5.77 | 111 / 573 | 249.8 | 4.35 s | 35.2 |
| 20 | 3.77 | 84 / 549 | 207.7 | 14.8 s | 132.4 |
| 18 | 1.77 | 74 / 523 | 133.6 | 65.8 s | 534.7 |

"First-token latency never sees the decode budget." The wall arrives between 22 and 20 GiB and
shows in throughput and TPOT: at 2 req/s with ~4 s in the system, ~7–8 requests decode
concurrently, needing ~6 GiB of resident KV. "The split moves the working-set problem off the
decode GPU but not the in-flight one." Telemetry: the prefill GPU is ~97% idle at every budget;
the decode GPU runs SM-active 51–68% with DRAM-active 45–62% and SM occupancy < 5% — memory-bound
compute sitting almost empty. Those two numbers (idle prefill GPU, decode GPU's empty SMs) are
the stated motivation for everything that follows: "resizing decode VRAM live and colocating
other workloads beside it".

## Phase 7 — Study 5: lend the decode GPU to a LoRA trainer (`vram-limit`, 2026-08-26 → 09-04)

(First committed as `report_colocation_decode.md` on 08-26, renamed `report_colocation_ft.md` on
09-04; no plan or decision log exists for this study; §1–5 of the report are a proposal —
including never-measured ideas for resizing vLLM's budget live — and §6 the measurements.)

A LoRA trainer is started on GPU1 beside the 8B decode (budget 25–26 GiB). The measured arc:

- **Time-slicing** (no MPS, Qwen2.5-0.5B trainer flat out): decode TPOT 28 → **1193 ms**, e2e
  3.9 → 152.8 s, **102 requests time out**, while the trainer loses only 11% — pure compute
  contention, zero preemptions. A 25% duty cycle rescues decode (TPOT 44.5 ms) at 16% trainer
  progress.
- **MPS SM caps** (GPT-2 trainer): a 10% cap costs +23% TPOT (34.2 ms), 50% +32%, uncapped +67%;
  zero failures; the trainer reaches 72–77% of its capped solo ceiling. A +45–70 ms TTFT offset
  appears in all MPS arms; it was left unattributed (the decode-alone-under-MPS control was
  skipped).
- **Idle-window gate** (Orion's policy across two processes): the decode engine publishes a busy
  flag from `execute_model` into `/dev/shm`; the trainer checks it per transformer block. TTFT
  returns to baseline; TPOT +41%. A one-block event fence was needed (unbounded async issue drained
  a whole idle gap's kernels into the next decode step).
- **CUPTI kernel gate** (`cupti_gate.so`, ~180 lines): because `LD_PRELOAD` runtime shims saw
  ~0 of the trainer's launches (cuBLAS/Triton fetch driver entry points via `cuGetProcAddress`), a
  CUPTI callback sleeps at every driver-level launch while the flag is set — 16.7 million launches
  gated in one run, zero overhead solo. It reproduces the block-level gate within noise (+46%
  TPOT). The residual TPOT cost is attributed to LMCache's KV-reload copies, which happen outside
  `execute_model` and so outside the busy window.

Recorded conclusion: "time-slicing (catastrophic) → duty throttle (safe, wasteful) → MPS caps
(safe, ~3× more efficient, TTFT-taxed) → idle-window scheduling (TTFT clean, TPOT mid-frontier,
no tuning knob)". Also: pricing a neighbour at a single 2 req/s point misses its real cost,
capacity headroom — "requires a QPS sweep per arm".

## Phase 8 — Study 6: a second model's decode beside the 8B decode (`vram-limit`, 2026-09-04 → 10-01)

A Qwen2.5-0.5B tenant (util 0.08, ~1.04 GiB KV) runs on GPU1 beside the 8B decode at 26 GiB;
three placements: whole tenant on GPU1 (A), decode-only with prefixes pre-populated by a
temporary GPU0 engine (B), and a second full Qwen split (C — defined, then deferred because an
offload cell drove host RAM below the memory guard's floor); tenant
workloads that fit in VRAM or oversubscribe it 3.2× (streaming 14.6 GiB per window from DRAM).

Without MPS the two decode processes **time-slice GPU1**: 8B TPOT 28 → 95–98 ms (3.4×) for a
tenant that "fits", and a QPS sweep prices the cost honestly — the 8B's capacity falls 327 →
245 tok/s (−25%). **MPS undoes it**: TPOT 33–34 ms, capacity ~300 (−8%). The telemetry tell: when
the tenant is added without MPS, GPU1's mean utilization *falls* (SM 0.52 → 0.48) — the GPU is
waiting for turns. The one hazard MPS does not fix is a tenant whose KV streams from DRAM: 8B
TTFT 7.7 s without MPS, still 348 ms with it — though the report marks these offload cells as
confounded (the runs sat at the RAM edge and the no-MPS control paged in 27× more), so only the
"fits" cells are clean. The
idle-window gate on top of MPS reclaims 298 → 315 tok/s at saturation but forces the tenant to
run eager (CUPTI event recording breaks vLLM's CUDA-graph capture). Conclusions: "MPS is
mandatory for either neighbor"; given MPS, a resident right-sized tenant — fine-tune or second
model — costs ~20%; "price colocation as capacity" (the single 2 req/s point "flatters by ~3×");
keep tenant KV resident.

The report went through several corrections between 09-29 and 10-01 (one commit is titled
"MPS overturns the two-axis model"). An earlier draft's claim that the offload collapse was a
pure host-bandwidth wall was withdrawn. A faulty tenant placement that re-fetched its prefix on
every request turned out to be a prefix-caching-off bug.

## Phase 9 — Study 7: N identical models on one GPU (`vram-limit`, plan 2026-10-01, runs 10-01/02, write-up to 10-13)

The first study with a timestamped decision log (pre-registered plans existed from study 1;
studies 5 and 6 had neither). N Qwen2.5-0.5B
engines (N = 1–8; util 0.08, 1.04 GiB KV grant each) or Qwen2.5-3B engines (N = 1–4; util 0.22,
`max-num-seqs 16`, 0.94 GiB grant) share GPU1 under MPS, with KV resident ("fits") or 3×
oversubscribed to DRAM ("offload"), decode-only or full, 2 req/s per model. New tooling:
`multi_launch.sh`/`multi_sweep.py` (engines up one at a time, all cells against one launch, QPS
sweeps), `host_monitor.py` (CPU by process role, RAM, swap), `thread_sampler.py` (per-thread
`/proc` counters, since `perf`/`ptrace` are unavailable).

Results: 0.5B fits under MPS scales near-linearly in aggregate throughput to N=6 (274 → 1522 tok/s)
with TPOT 1.56 → 3.63 ms, knee N≈6–8 (TPOT 5.70 ms at N=8, SM-active 0.86). But the knee is a
*latency* knee: N=8 still serves 16 req/s per model with zero queueing. The 3B knees at N≈2–3
(TPOT 6.8 → 17.3 ms, DRAM-active 0.44 → 0.82), VRAM holds 4, and under load the per-engine
`max-num-seqs=16` cap binds (12 / 8 / 4 req/s per model at N = 1 / 2 / 4). At equal total load
(32 req/s) one 0.5B engine does 1.97 ms/token and eight engines 7.78 ms — **the cost of N is lost
batching**.

The 3B with offloaded KV showed a fourth limit before GPU, VRAM or RAM were full: its
decode-only queue wait grows 82 ms → 4.7 s from N=1 to N=4 while only ~2 requests run and KV is
43% used — the per-request KV load stretched 85 → 401 ms. The cause was left open ("KV-reload
queueing"); it is the precursor of study 10's load-stall finding.

The surprise: **4 offloading 0.5B engines collapse** (TPOT 35 ms, 489 of 1200 requests failed,
GPU SM-active 0.05; 4 of 4 stock runs). The host trace showed the EngineCores, not the stores,
pinning all 32 cores: each engine's torch/OpenMP pool is sized to the whole machine (~135
threads), and with DRAM-offloaded KV they spin-wait. `OMP_NUM_THREADS=4` removes it (2/4/8
behave identically); 8 resident engines storm transiently in 2 of 4 stock runs, 0 of 3 capped.
Two measurement conventions were fixed here and hold for the rest of the project: **capacity =
scheduler queue wait** (`vllm:request_queue_time_seconds` mean < 50 ms, ≤ 1% failed) — not
delivered/offered (which reads 92–94% at high QPS with zero queueing) and not SM-active (it fell
while throughput doubled); and a cell is `ok` only with zero failed requests. Conclusions: "The
failure that actually stops colocation is on the host, not the GPU"; "colocate for isolation,
not for efficiency"; "size the KV grant and the sequence cap together".

## Phase 10 — Study 8: another model's prefill on the decode GPU (`vram-limit`, plan 2026-10-13, write-up 10-17)

A Qwen2.5-3B prefill engine (fresh 6144-token prompts, no reuse, 0.5 / 1 / 2 req/s) runs on GPU1
beside the 8B decode at 22 GiB — the smallest budget that held 2 req/s in the split study and the
only one that leaves room for a 3B. Three placements for the 3B's KV: dropped (P1), saved to its
own DRAM store (P2), handed to a 3B decode on GPU0 (P3).

GPU1 SM-active rises 0.52 → 0.66 and the 3B barely slows (+7% TTFT). But the 8B pays in
**capacity, not per-token time**: ITL p50 14.4 → 14.5 ms unchanged, while its residence on the
decode engine stretches 3.1 → 4.3 s; with ~7 requests' worth of KV grant (5.77 GiB ÷ 0.77 GiB)
and 2 req/s × 4.3 s = 8.6 slots needed, the queue grows without bound — −1 / −7 / −19%
throughput at 0.5 / 1 / 2 req/s of 3B prefill. The report's own reading: "the cost is not a
property of prefill-as-neighbour so much as of the 8B having no headroom" (it was already queueing
~1 s alone at 22 GiB).

Two stack defects found and fixed: (1) stock vLLM 0.15.1 + LMCache 0.4.4 saves KV to the store
**synchronously inside the step**, in front of token #1 (a 55 ms copy, 2.7× TTFT at 2 req/s); an
**async-save patch** (`--sm-async-save`, three connector methods) returns TTFT 576 → 234 ms; (2)
the stock prefill → DRAM → decode handoff **races**: 26–28% of decode requests found only part of
their prompt's KV and re-prefilled; a router marker-file gate (wait until the L1 → store push
finishes) gives 300 of 300 full hits. Host RAM bounded P3 again (pinned pools cut to 4 GiB,
swap-backed). The stock LMCache store never evicts, so a store cap (`LMC_STORE_CAP_GB`) was
added.

## Phase 11 — Study 9: who gets priority, 8B decode or 3B prefill? (`vram-limit`, plan 2026-10-17, overnight run 10-17/18, report 10-18, revised 10-19)

Motivated by study 8's confound (prefill-neighbour cost vs no-headroom cost), the 8B now runs
with headroom in three configurations — A: bf16 3B + 8B at 23 GiB; B: fp8 3B + 23 GiB; C: fp8
3B + 25 GiB — against a 3B prefill of fresh 2048-token prompts at 2 / 4 / 8 req/s whose KV is
saved to its own DRAM store (study 8's placement P2, with the async-save patch in every cell), in
five arms:
`fair`, `mps8b` / `mps3b` (`CUDA_MPS_CLIENT_PRIORITY=1` on the low-priority engine), `gate8b` /
`gate3b` (the CUPTI launch gate driven by the other engine's busy window). Swap-backed cells were
rerun. DCGM tensor-pipe-active was added as the compute metric.

Fair sharing at 8 req/s costs the 8B −18% throughput and an 18 s queue in A, −3% and 1.3 s in C.
A lighter neighbour (fp8) and more decode KV each close about half the A → C gap. **MPS client
priority protects the 8B** (A, 8 req/s: TPOT 166 → 88 ms, 209 → 227 tok/s) for +20–70 ms of 3B
TTFT — predicted to be weak, it was the effective lever. **The launch gate does not**: ≈ fair for
the 8B and it starves the 3B (TTFT 30 s at 8 req/s); a tighter gate is worse on both sides.

The study's thesis, which seeds study 10: **"A prefill neighbour costs a decode KV slots, not
tokens."** A *slot* here is the KV space one 6400-token request occupies on the decode GPU
(0.78 GiB); *residence* is how long a request holds it (KV load + 128 decode steps). Every
non-gated cell falls on one curve of 8B queue wait vs **slot utilization = offered rate ×
residence ÷ (KV grant ÷ 0.78 GiB)**: ~0.1–0.3 s below 0.7, 1–2 s at 0.8–0.87, 4–8 s at
0.92–0.99, 18–20 s past 1.0. (All cells of this study ran under MPS; the report notes the rule
"is fitted on one 8B, one workload and P2 only; the residence a neighbour adds has to be
measured, not derived".) An early reading that the gate was *worse* than fair turned out to be
a swap-backed run (107 → 64 ms when rerun) — the origin of the rule that cells with > 0.5 GB
swapped in are rerun. The 8B alone at 23 GiB saturates at ~285 tok/s (≈ 2.2 req/s),
below the Little's-law estimate because residence grows with the batch. The handoff prompt
written after this study (`PROMPT_KVCACHE_VRAM.md`) corrected the report's ~8.7 / 11.2 slots to
integer counts (engine logs: Running never exceeds 8 at 23 GiB, 11 at 25) and listed two
suspected wastes to test next: duplicate same-session prefixes on the decode (prefix caching is
off there) and a synchronous 32-layer KV load that may stall the whole batch at every admission.

## Phase 12 — Study 10: KV slots on the decode GPU (`kvcache-vram_size`, 2026-10-25 → 10-26)

A new worktree, stripped to the 8B split stack with nothing else on GPU1 and no MPS daemon
(`scripts/kv_vram/`; the earlier colocation baselines ran under MPS, so absolute numbers are
not directly comparable with study 9's), with a pre-registered plan (`PLAN_KV_VRAM.md`), a dated decision log (`DECISIONS_KV_VRAM.md`,
entries D0–D11 for decisions and E0–E5 for the planned experiments) and a record-only trace patch on the decode engine (per request: arrival, free
blocks at arrival, admission, first token, finish; per step: batch, wall time, KV-load calls).
Grid: decode budget 24 / 26 / 28 GiB × 1–5 req/s, one fresh launch per cell.

**Day 1 (10-25) — probe, trace, baseline grid.** Probe: KV grant = budget − 16.23 GiB; 3,980 /
5,004 / 6,027 blocks; whole requests c = 9 / 12 / 15 (a request holds 392–400 blocks of 16
tokens). E1 confirmed the stall hypothesis: steps that overlap a KV load take p50 173 ms vs
14 ms; 34% of decode wall time at 24 GiB / 2 req/s is load stall. The baseline grid was run on
the Python-fallback LMCache.

Two ideas were closed the same day. The streaming micro-benchmark (E3) ruled out "stream KV per
step": a streamed step costs 58 ms × batch (exactly 14.47 GB/s) vs 12–18 ms resident — 4.9× at
batch 1, 52× at batch 16; prefetch depth beyond 1 changes nothing. `use_layerwise` cut the
fallback's queue 3× — but by sidestepping an L1-staging-pool allocation thrash in LMCache's
whole-request path, not by overlapping copy with compute (per-layer fetch and copy alternate,
never run concurrently). `enable_async_loading` crashed LMCache 0.4.4 on the fallback.

A pre-registered check of the first reading of the stall data failed: batching admissions
through the router (five at a time) halved the stalls per residence (4.0 → 2.1) but lengthened each stall
(308 → 508 ms), leaving TPOT unchanged — so the rule was corrected to "stall time per residence =
requests loaded while this request is resident × ~190 ms, independent of how admissions are
grouped".

Then the discovery that reframes studies 3–9: **LMCache's `c_ops.so` had never loaded against
torch 2.9.1 (the wheel was built for torch 2.10) in any LMCache study on this box.** Rebuilt
from the 0.4.4 sdist in a separate venv (`.venv-lmc`), the per-layer copy goes 24.1 → 1.95 ms
and the per-request copy 190 → 114 ms. The fallback report was frozen as
`report_kv_slots_fallback.md` and the whole grid rerun.

**Day 2 (10-26) — the compiled-kernel grid.** 27 cells, all `ok`; the full table is in
REPORT_SUMMARY §3.5. Capacity (queue < 50 ms) 2–2.5 / 2.5–3 / 2.5–3 req/s and saturation
throughput 2.97 / 3.35 / 3.58 req/s at 24 / 26 / 28 GiB. Two readings of the per-request
wait figure were corrected during the day (free blocks alone, then an off-by-one in the queue
count) before the final rule below.

(1) **The decode engine is a bank of integer KV slots** — effective 10 (the 392-block
admission rule lends a 10th at 24 GiB, which then evicts), 12, 15; a FCFS replay with that slot
count and each request's measured residence reproduces the mean wait within 0.1% (26/28 GiB) and
2% (24 GiB), and c ± 1 is off by 1.5–3× (the report calls this "a re-simulation of a measured
result, not a predictive model": it shows waiting needs nothing beyond slots and residence); per request, wait ≈ k × residence ÷ c where k is the
number of running requests that must finish first (k = 0 → ≤ 0.2 ms wait in 1,258 of 1,258
cases). (2) **Below the knee the budget is irrelevant**: at 1–2 req/s all budgets give the same
throughput, zero queue and 14.5 ms TPOT. (3) **Above it capacity scales sub-linearly**: +50%
slots (24 → 28 GiB) buys +21% saturation throughput (2.97 → 3.58 req/s), 26 → 28 only +7% —
because every admission copies 0.75 GiB *synchronously inside the engine thread* (~120 ms) and a
request among more neighbours sits through more stalls (1.6 at 1 req/s → 8.7 at 28 GiB / 3
req/s), so residence grows with the slot count. (4) The same request (#255) at 3 req/s waits
4,226 ms at 24 GiB and 750 ms at 28 GiB; its load and decode are the same — slots change the
wait, not the service. (5) About a third of resident KV is a duplicate of a prefix already on the
GPU (decode-side prefix caching is off; left untested because it changes the workload).

**Levers tried** (D9–D11): `enable_async_loading` with compiled kernels moves the store fetch
before admission (residence −5 to −16%, +7% saturation) but every request pays a ~100 ms prefetch
and it crashes at 24 GiB ≥ 4 req/s; LMCache's **multi-process connector on vLLM 0.18.1** (the
copy runs in a separate process over CUDA IPC while the batch decodes) removes the stall (extra
step time 0.67 → 0.05 s per request), makes residence independent of the slot count (1.99 s at 3
req/s at every budget), lifts capacity by 0.5–1 req/s on queue time (1–1.5 on slot wait) and
throughput +40% — but is not version-matched (0.18.1's grant is 0.5–0.6 GiB larger), and at 28
GiB a lent 16th slot causes evictions that this connector *recomputes* rather than reloads,
collapsing at 4.5–5 req/s (174 evictions, 303 tok/s). fp8 KV (≈ 20 slots at 24 GiB) and
decode-side prefix caching remain un-run. Two checks were left undone: the trace's
zero-overhead validation was repeated only under the fallback, and LMCache's L1 staging pool
still fails ~3,400 allocations per run. Slide figures were produced and reviewed twice.

## 2026-11-01 — consolidation

Uncommitted data on `vram-limit` (the `infc_B_fits` eager/gate/gateq4 tenant runs and the
study-10 handoff prompt), the Tally patch, and the reference slide decks were committed; all six
branches were merged into `main` under `branches/<name>/` with history preserved. These two
reports were written from that snapshot.
