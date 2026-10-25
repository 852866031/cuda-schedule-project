# PLAN — KV slots on the decode GPU

Pre-registration. Predictions below are written before any run and are never rewritten;
outcomes are added as `> **Outcome (date):**` annotations. Run-level decisions and their
predictions go to [DECISIONS_KV_VRAM.md](DECISIONS_KV_VRAM.md). Starting facts:
[NOTES_KV_SLOTS.md](NOTES_KV_SLOTS.md).

## Question

On the decode engine of the 8B split (prefill GPU0, decode GPU1, KV through a shared
LMCache store), how do **KV space**, **requests running** and **VRAM budget** relate, and
how much does each candidate technique raise the load a budget can carry? Capacity is
the CLAUDE.md definition: the highest per-model QPS with decode queue wait
(`vllm:request_queue_time_seconds`) mean < 50 ms and ≤ 1% failed.

Working model, to be tested rather than assumed: the decode engine is an **M/G/c queue**.
c = ⌊KV-grant blocks ÷ blocks per request⌋ (integer slots); service time S = KV load +
decode of 128 tokens; arrivals ≈ Poisson at the router's rate (smoothed by prefill).
Everything that helps does one of two things: raises c (fewer bytes per resident
request, or sharing) or shortens S (faster load, no batch stalls).

## Fixed setup

| item | value |
|---|---|
| model | NousResearch/Meta-Llama-3-8B-Instruct, bf16 |
| prefill | GPU0, util 0.80, APC on, no chunked prefill, 32768 batched tokens |
| decode | GPU1, util = budget ÷ 31.3536 GiB, `--no-enable-prefix-caching` (baseline), CUDA graphs on, max-num-seqs default (256) |
| KV path | LMCache 0.4.4 (Python fallback), chunk 256, L1 pinned 6 GiB per engine, TCP store :8300 uncapped |
| router | `lmc_proxy.py --forward-first-token --max-inflight 999` |
| GPU1 tenants | the 8B decode only; **no MPS daemon** (the ninth study's `base` cells ran under MPS — not directly comparable, see E0) |
| host | `OMP_NUM_THREADS=8`, `PYTHONHASHSEED=0`, LD_PRELOAD, mem_guard + heartbeat via launch script |
| workload | reference: 32 sessions × 6144 prefix + 128 suffix, 128 out, zipf-1.1, Poisson, seed 0, 300 requests; only the QPS changes |

Code: `scripts/kv_vram/` copied from `scripts/inf_pf_prio/` with the 3B neighbour, MPS,
gate and priority arms removed. Driver `kv_sweep.py`, launcher `kv_launch.sh`,
instrumentation `site/kv_trace.py`. Raw names `kv_<method>_b<budget>_q<qps>[_rN]`;
telemetry and logs per launch tag `kv_<method>_b<budget>_L<n>`.

**One launch per (method, budget); all QPS cells of that budget against it**, each
preceded by the standard warm-up and separated by a drain check (decode Running = 0,
Waiting = 0). Cells run in a fixed shuffled order and the first cell is repeated last
as a drift check (difference must be within ±15 ms p50 TPOT and ±10% queue wait,
otherwise that budget is rerun one launch per cell). Rationale: KV state that persists
across cells (store contents, prefill APC) is already warm after the first warm-up, and
the decode engine holds nothing across cells in the baseline; saves ~3 min of
start-up per cell.

## E0 — probe and calibration

1. **Grant probe.** Launch the stack at decode budgets 22, 23, 24, 25, 26, 28 GiB, read
   `GPU KV cache size` from the decode log, stop, verify GPU1 shows < 100 MiB between
   probes. Record blocks and c = ⌊blocks ÷ 400⌋.
   - **Prediction:** grant = budget − 16.2 GiB ± 0.1; tokens 22/23/24/25/26/28 ≈
     47.3k / 55.5k / 63.7k / 71.9k / 80.1k / 96.4k; **c = 7 / 8 / 9 / 11 / 12 / 15**.
2. **PCIe calibration.** `scripts/common/calibrate_pcie.py` on both GPUs (pinned and
   pageable H2D/D2H, 1 MiB–1 GiB).
   - **Prediction:** pinned H2D 13–14.5 GB/s at ≥ 64 MiB (the earlier file reads ~14);
     pageable ≤ 8 GB/s.
3. **No-MPS baseline check.** One cell, 24 GiB, 2 QPS, no instrumentation, compared to
   the ninth study's b23/b25 `base` rows.
   - **Prediction:** TPOT p50 within ±3 ms of 27.7–29.4 ms; throughput ≈ 254 tok/s.

## E1 — per-request timeline instrumentation

`site/kv_trace.py`, loaded via PYTHONPATH sitecustomize into the decode EngineCore only
(env switch `KV_TRACE=1`), record-only. Wraps the v1 `Scheduler` and `EngineCore.step`:

- per request: `t_arrive` (add_request), free blocks at arrival, `t_sched` (first
  scheduled), blocks allocated, external (LMCache) tokens hit, `t_first` (first output
  token), `t_done`, preemptions.
- per step: wall start/end, running, waiting, free blocks, number of requests admitted
  this step and their external tokens (= the step that carries a synchronous KV load).

Writes one JSONL per launch to `/dev/shm`, copied to `output/kv_vram/trace/` at teardown.
Clocks: `time.monotonic()` inside the engine plus a wall-clock anchor, joined to the
client's per-request records by request id.

- **Validation (smoke):** 24 GiB, 2 QPS, with vs without `KV_TRACE=1`, same launch
  order. Pass if TPOT p50 differs ≤ 1 ms, throughput ≤ 1%, queue wait within run-to-run
  noise. **Prediction:** passes; overhead < 0.1 ms/step (list append per step).
- **Prediction on hypothesis 2 (stalls):** steps that admit a request take
  **≥ 150 ms** (vs ~15 ms otherwise); they account for most of the gap between ITL p50
  (~15 ms) and mean (~28 ms). If admit steps are < 50 ms the hypothesis is rejected and
  the ITL tail has another cause (to be found from the step trace).

## E2 — baseline grid (sweeps A and B)

Budgets {24, 26, 28} GiB × QPS {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0}, plus 2.25 and 2.75
at 24 GiB (knee region). 23 cells, ~25 min of GPU per budget. Sweep A is the 24 GiB
row; sweep B is the column at the QPS where 24 GiB is most sensitive (chosen from A;
predicted 2.5). Metrics per cell: throughput, TTFT p50/p95, TPOT p50/p95, ITL
p50/mean/p95, decode queue wait mean/p95, failed, swap-in, and the E1 trace.

- **Prediction — capacity:** 24 GiB (c = 9): 2.0–2.5 QPS; 26 GiB (c = 12): 2.5–3.0;
  28 GiB (c = 15): 3.0–3.5. Capacity grows **sub-linearly in c**, because S grows with
  batch size (4.14 s at Running ≤ 8 → 4.89 s at ≤ 11 in the ninth study).
- **Prediction — shape:** below the knee, queue wait < 50 ms and TPOT flat; above it,
  queue wait grows roughly linearly with QPS (fixed 300-request window, so finite);
  throughput saturates at ≈ 128 × c ÷ S tok/s. TPOT **rises** with budget at equal
  high QPS (bigger batches), as in the ninth study.
- **Prediction — M/G/c:** with c from E0 and S from E1, an M/G/c estimate
  (Lee–Longton, plus a discrete-event simulation that replays the measured S
  distribution) predicts mean queue wait within a factor 2 below ρ = 0.9, and the knee
  QPS within ±0.25. Beyond ρ ≈ 1 the finite window dominates and only the simulation is
  compared.
- **Prediction on hypothesis 1 (no sharing):** in the trace, at Running = c, the
  running set spans ~0.7 × c distinct sessions (from the arrival simulation: 10.3
  requests / 6.8 sessions).

## Figures (deliverable 1)

1. `kv_grid_3d.png` — 3D surface, VRAM × QPS → queue wait; companion
   `kv_grid_heat.png` heatmaps with contours for queue wait, throughput, TPOT p50 (the
   heatmap is the readable one; the 3D is kept).
2. `kv_slots_timeline.png` — **the "free KV space vs performance" figure.** One budget
   pair (24 vs 28 GiB at the same QPS, same arrival sequence), stacked panels sharing a
   time axis: (a) the c slots as Gantt lanes, each request a bar coloured by phase
   (KV load / decode), (b) free KV space in units of "requests that still fit" (step
   line) with queue length as a filled area, (c) one annotated request followed across
   both budgets. Title states the metric; legend grouped above the panels.
3. `kv_wait_vs_free.png` — scatter, x = free slots when the request reached the decode
   engine (0, 1, 2…, jittered), y = its decode queue wait, one colour per budget;
   shows that wait is ≈ 0 whenever a slot is free and only otherwise depends on load.
4. `kv_mgc.png` — measured vs M/G/c queue wait vs QPS, one panel per budget.

The report gives concrete timeline examples ("request #N arrives at t = … with all 9
slots full and … free blocks; admitted when #M finishes at t = …; at 28 GiB the same
request finds 3 free slots and loads immediately").

## E3 — system diagrams and the prefetch micro-benchmark (deliverable 2)

Diagrams (matplotlib, house style, physical placement honest): (1) a request's route
with device and measured stage times; (2) how the queue forms — integer slots, Poisson
arrivals, c servers, overlaid on a measured timeline; (3) LMCache's KV movement in
decode — 256-token chunks, L1 pinned pool, store, synchronous `start_load_kv` into the
paged blocks — and the VRAM/host layout (32 per-layer tensors × blocks; chunk format);
(4) why decode cannot stream per layer.

**Micro-benchmark** `scripts/kv_vram/bench_stream.py` (`.venv`, no vLLM): one 8B-shaped
decode step (32 layers; per layer QKV/O/MLP GEMMs at Llama-3-8B shapes, paged-free GQA
attention over 6400-token contexts) for batch B ∈ {1, 2, 4, 8, 12, 16}. Arms: KV resident
in HBM; KV streamed per layer from pinned host on a copy stream with prefetch depth
d ∈ {0, 1, 2, 4, 8} layers ahead (d = 0: copy then compute, no overlap). Output: step
time vs B per arm.

- **Prediction:** resident step 12–25 ms over B = 1–16. Streamed step ≈ max(compute,
  B × 0.78 GiB ÷ BW): ≈ 60 ms × B at 14 GB/s, i.e. ~4× slower at B = 1 and ~50× at
  B = 16; d > 1 gives no further gain once copy-bound (d only hides compute, which is
  the smaller term). The crossover where streaming is free does not exist above B = 0.
- **(c) layerwise load**, if E4 shows `use_layerwise=True` runs: compare the admit-step
  duration (E1) with and without. **Prediction:** load overlaps with the first forward
  but stays ≥ 200 ms (bounded by the Python-fallback copy, ~2 GB/s, not PCIe).

## E4 — feasibility smokes (deliverable 4)

Each: does it start, does it hang (watchdogs), KV grant and c, one cell (24 GiB,
2.5 QPS) TPOT and queue wait, and output equality on 8 fixed prompts (greedy) vs
baseline.

| method | change | prediction |
|---|---|---|
| decode APC | drop `--no-enable-prefix-caching` on decode | starts; lossless (identical text); same-session requests share the 384 prefix blocks once the first has loaded, and freed prefixes stay as evictable cache, so effective c rises from 9 to **≥ 14 concurrent requests** at 24 GiB and queue wait at 2.5 QPS falls **≥ 5×**. Risk: LMCache adapter reports external tokens only beyond the local hit — expected to work (the API passes `num_computed_tokens`). |
| LMCache `enable_async_loading` | LMCache config flag | **does not take effect** in 0.4.4's vLLM-v1 adapter (the check is commented out at `vllm_v1_adapter.py:542`); verify, record as infeasible if so |
| LMCache `use_layerwise` | LMCache config flag (both ends) | starts; admit-step stall shrinks by < 30%; c unchanged |
| fp8 KV | `--kv-cache-dtype fp8` on prefill **and** decode | the open question is whether LMCache 0.4.4's fallback store/load handles 1-byte KV and whether the sm_120 backend accepts fp8 KV. If it runs: 64 KiB/token, 200 blocks/request, **c = 9 → 19** at 24 GiB; first 32 tokens of greedy output may differ on ≥ 1 of 8 prompts |
| ShadowKV / NEO | assessment only | read the repos: GPU/arch and framework requirements; predicted not runnable on sm_120 without porting (record why) |

New venvs only if a smoke needs another vLLM version; `.venv-matched` is never modified.

## E5 — rerun the grid for feasible methods (deliverable 5)

Every method that passes E4 runs the identical E2 grid (or the subset that shows its
knee — confirmed with the user before queueing). Figures overlay it on the baseline
(baseline thick/dark, methods lighter), and each method is attributed to "raises c",
"shortens S", or both, from E0-style grant numbers and the E1 trace.

- **Prediction:** decode APC moves 24 GiB capacity above 3.5 QPS (raises effective c);
  fp8 roughly doubles c, capacity +60–80% at equal budget (S slightly longer from the
  larger batch); layerwise changes capacity by < 10%.

## Deliverable 3 — related work

One schematic per family, each with: what it saves (resident bytes / time resident /
bytes per token), lossless or not, expected effect on this workload, and verified
citations (papers and docs checked, no numbers from memory). Families: prefix sharing
(vLLM APC, SGLang RadixAttention); load/compute overlap (LMCache layerwise/async,
CachedAttention/AttentionStore); KV quantization (vLLM fp8 KV, KIVI, KVQuant); offload
+ sparse retrieval (InfiniGen, ShadowKV, Quest, MagicPIG); CPU-side attention (NEO,
FastDecode); token dropping (H2O, SnapKV, StreamingLLM); structural (GQA, MLA,
cross-layer KV sharing).

## Order and budget

E0 (≈ 30 min) → E1 smoke (≈ 15 min) → E2 grid (≈ 1.5 h) → E3 bench (≈ 20 min) and
diagrams → E4 smokes (≈ 1–2 h incl. compatibility work) → E5 (≈ 1.5 h per method) →
related work and report. Strictly one stack at a time; nothing runs in `../vram-limit`
meanwhile. Swap-in > 0.5 GB → rerun that cell, report the lower-swap run.
