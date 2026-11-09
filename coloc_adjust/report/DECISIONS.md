# Decision log — colocation VRAM-adjust prototype

## D1. Tier calibration (21:08–21:12, GPU 1 solo, `tiers.json`)

Qwen2.5-0.5B LoRA r=16, 8×512 tokens per optimizer step. Need = 1.1 × peak allocated + 0.25 GiB.

| cfg | tier | k | act offload | layers offloaded | ckpt | need GiB | step ms | tok/s |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 1 | – | 0 | off | 15.9 | 91 | 44.8k |
| 1 | 1 | 2 | – | 0 | off | 8.7 | 106 | 38.7k |
| 2 | 1 | 4 | – | 0 | off | 5.0 | 138 | 29.6k |
| 3 | 1 | 8 | – | 0 | off | 3.2 | 210 | 19.5k |
| 4 | 2 | 8 | yes | 0 | on | 2.5 | 1045 | 3.9k |
| 5 | 3 | 8 | yes | 12 | on | 2.2 | 1294 | 3.2k |
| 6 | 3 | 8 | yes | 20 | on | 1.9 | 1452 | 2.8k |
| 7 | 4 (pause) | – | – | – | – | 0.4 | – | 0 |

First calibration had gradient checkpointing on for every tier; then activation offload gave no
memory gain (checkpointing already keeps only layer inputs) — tiers 0–1 now run without
checkpointing (faster, larger), tiers 2–3 with it. Activation offload moves ~4.8 GB per step each
way at k=8 and costs 5× in step time; weight offload of 20/24 layers takes another 0.6 GiB off.
The floor (~1.9 GiB) is the lm_head/embedding + one micro-batch's logits; below that the tenant
pauses. Against the lendable table in DESIGN §2: 0.5 QPS → cfg 1, 1–2 QPS → cfg 2, 3 QPS → cfg 3,
4 QPS → pause.

## D2. Smoke tests of the loop (21:14–21:28, GPU 0, engine without MPS)

- Busy page: `execute_model → sample_tokens` bracket live (2220 steps, 2.9 s busy in ~35 s at
  2 QPS, i.e. the decode node is idle ~90 % of the time — the gap the tenant uses).
- Connector tracer: stores of 784 MiB in 57 ms (14.3 GB/s), loads similar; 133 records in the
  first 30 s.
- Router demand matches the engine (running 1 → 400 blocks; running 7 → 2.8k blocks).
- Bug found and fixed: once paused, the worker fell through to a k=0 training step
  (ZeroDivisionError) and never resumed. Added a 3-s hold before climbing a tier (the budget
  flaps with the in-flight count at a fixed RPS; dropping stays immediate).
- 3 → 0.5 QPS step: limit 3 → 12.65 GiB within one tick, tenant cfg 1 → 3 → 5 → pause within
  ~8 s; at 0.5 QPS the limit shrank to 2–4 GiB after the 5-s hold and the tenant resumed at
  cfg 1 (9.2 GiB). Decode at 2 QPS beside the gated tenant (no MPS): TTFT p50 178 ms, TPOT p50
  14.8 ms (solo 13.3), p95 27 ms, 0 failures.
- VRAM scheduler look-ahead shortened from 2 s to 1 s of arrivals (the tenant drops a tier in
  ≤ 1 step, 0.1–1.5 s); at 3 QPS the 2-s horizon alone was 4.7 GiB.
- MPS: the daemon exits silently when the pipe directory path is long (Unix socket limit);
  `/tmp/coloc_mps/pipe` works.

**Prediction for the schemes (DESIGN §4):** scheme C keeps TPOT within 10 % of scheme A in every phase
except the 10 s after each RPS rise; queue wait < 100 ms away from transitions; 0 failures;
tenant cfg follows the lendable table; scheme B (6.7 GiB static, tenant 6 GiB fixed) queues
heavily at 3 and 4 QPS and leaves ~5 GiB idle at 0.5–1 QPS; scheme D (no gate) costs decode
+30–60 % TPOT.

## D3. Pipeline test under MPS (21:26–21:31, `results/pipeline_test/`)

Scheme C with a 2-minute schedule (1 QPS → 3 QPS at 60 s): everything ran together (MPS daemon,
engine as MPS client with kvcached, router, sampler, gated tenant, load through the router).
Phase 1 QPS: TPOT p50 16.3 ms, queue wait 119 ms, 128 tok/s delivered = offered, tenant cfg 2
(k=4, 5 GiB, 8.4k tok/s, 69 % of its step time gated). Phase 3 QPS: limit 12 GiB, tenant paused
within the first ticks, TPOT 19.1 ms, queue 196 ms, 0 failures, 0 preemptions. Connector loads
(768 MiB each): median 55.7 ms with no concurrent tenant copy, 61.3 ms when one overlapped
(n=15). The analysis's per-request ok flag was a list, fixed. Open question for the schemes:
decode TPOT at 1 QPS beside the tenant is 16.3 vs 13.3 ms solo (no MPS) — how much is MPS
itself (scheme A runs decode alone under MPS) and how much the tenant (scheme D without the gate).
Full schemes launched 21:33: A decode_alone, B static_split, C adjust, D adjust_nogate, ~15 min each.

## D4. First full schemes (21:33–22:35)

Schemes A (decode alone) and B (static split: engine 6.69 GiB, tenant 6 GiB fixed, cfg 2) completed
and are kept. Schemes C/D: the tenant died at ~115 s (2 QPS phase) with a CUDA OOM *inside the pause
transition*: `to_cpu()` first re-bound the streamed layers to the GPU (`set_weight_offload(0)`)
while the allocator cap had already been lowered to the ~0 budget, so the 2 MiB allocation for
the bind failed and the process exited; decode ran alone for the remaining 9 minutes (which is
why schemes C/D equal scheme A from 180 s on). Fix: pausing now moves streamed layers straight to
their pinned host copies (no GPU allocation), tier climbs lift the cap transiently before
re-binding, and any OOM during a transition falls back to pause instead of exiting. Schemes C/D
re-run after a pause/resume unit test on GPU 1.

What the completed phases already say: at 1 QPS the gated tenant (cfg 2, 8.6k tok/s) costs
decode +3.6 ms TPOT (16.5 vs 12.9), queue wait +26 ms, same throughput, 0 failures; the gate
made no difference to TPOT (nogate 16.5) — the old study's TPOT floor again. Scheme B at 3 and 4 QPS:
queue wait 7.9 s and 13.5 s, delivered 317/325 tok/s against 402/517 offered, and the backlog
still 26 s deep at the 0.5 QPS phase; its tenant trains at 12.7k tok/s throughout.

## D5. The busy signal measured launch time, not GPU time (22:40–23:00)

Jiaxuan asked what "decode busy" was. It was the wall-clock bracket `execute_model →
sample_tokens`, which on vLLM 0.30's Model Runner V2 returns before the GPU finishes (async
scheduling); the tenant saw a 5 % duty and the gate was effectively never closed, which is why
schemes C and D were identical. Rebuilt: busy is raised at `execute_model` entry and cleared when a
CUDA event recorded after `sample_tokens` completes (a side thread waits on it), busy time as
the union of overlapping steps. Measured with the rebuilt signal (8 sessions, 25-s loads):

| RPS | steps/s | GPU busy duty |
|---|---|---|
| 0.5 | 49 | ~100 % (accounting shows 110) |
| 1 | 72 | 93 % |
| 2 | 75 | ~100 % |
| 3 | 66 | 100 % |

The decode GPU has **no idle window** while any request is running: steps are launched
back-to-back and overlap, and each one streams the 15 GiB of weights (~12 ms) whether the batch
holds 1 or 7 requests. The 30–50 % idle the vLLM 0.15 study saw was CPU time between steps that
MRV2 removed. Consequence: the idle-window gate can only let the tenant run when the engine is
empty (≈ 15 % of the time at 1 QPS, ~0 at ≥ 2 QPS); compute sharing here has to be spatial.
Added scheme `adjust_mps20` (tenant under `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20`, no gate) as the
first spatial variant; green contexts and the CUPTI gate are noted for later. The tenant's
"collision" rule (split when duty > 60 %) will now be permanently on while requests run.
Schemes E (adjust, no tenant), C, D, F launched 23:0x with the rebuilt signal.

## D6. Schemes with the rebuilt busy signal (22:47–23:48) and mid-step preemption

Results in `results/no_midstep_preempt/` (C, D, F) and `results/adjust_notenant_*` (E).
- **E, VRAM scheduler without tenant = A exactly** (TPOT 12.9/15.2/21.2/12.5/26.0 … in both, queue
  wait within 3 ms, same throughput, 0 preemptions): resizing the limit live with load costs
  decode nothing. The limit followed the schedule: 5.0 / 8.0 / 12.6 / 4.5 / 12.6 / 3.0 GiB.
- **C (gate) with the true busy signal: the tenant trained 9 steps in 11 minutes** — the engine
  is never idle while a request runs, so the gate never opens. Worse: the tenant sat inside the
  gate's spin loop holding its 5 GiB, never reached the between-steps budget check, so when the
  VRAM scheduler raised the limit kvcached could not map pages and the engine queued for seconds
  (4.4 s at 4 QPS, up to 15 s later) — a gated tenant that cannot be preempted is the worst of
  both worlds. **Fix:** the tenant now polls the budget every 50 ms from the per-block hooks and
  inside the gate wait and aborts the step (`Preempt`) when the budget no longer fits its config;
  unit test on GPU 1: budget drop detected after 40 ms, step aborted, tier 2 → 5 in 2.3 s,
  back to 2 when the budget returned.
- **D (no gate, MPS co-run)**: decode TPOT +28 % at 1 QPS (16.6 vs 12.9), +22 % at 2 QPS, +3 %
  at 3 QPS; 10 and 13 preemptions at 3 and 4 QPS and queue wait 3.5 s at 4 QPS (vs 1.2 alone) —
  the tenant's release lags the limit's growth by a tier change (0.2–1.5 s), during which
  kvcached cannot map and vLLM preempts. Tenant 11.3k tok/s at 1 QPS, 5.3k at 2, 0.2k at 3, 0 at
  4, 17k at 0.5.
- **F (MPS 20 % cap, no gate)**: TPOT +13 % at 1 QPS (14.6), +12 % at 2 QPS, +1 % at 3 QPS; 0
  preemptions; queue 2.2 s at 4 QPS; tenant 6.7k / 3.1k / 0.15k / 0 / 8.2k tok/s. Half of D's
  decode cost for 60 % of its tenant progress.
- **B (static split)**: queue 7.9 / 13.5 s at 3 / 4 QPS, backlog still 26 s at the 0.5-QPS phase,
  TPOT +30 % at 1 QPS; tenant 9.6–13.1k throughout.
C, D, F re-run with mid-step preemption (prediction: C's decode returns to E's numbers with the
tenant at ~0 progress; D/F lose their preemptions at 3–4 QPS and most of the 4-QPS queueing).

## D7. Schemes with mid-step preemption (23:50–00:35) and the green-context round

- **C (gate + mid-step preemption)**: decode now equals the no-tenant baseline in every phase
  (TPOT 13.1 vs 12.9 at 1 QPS, queue wait 107 vs 107 ms, 0 preemptions, 4 QPS queue 2.2 s vs
  1.2 s) — and the tenant made 32 steps in 11 minutes (21 of them preempted). The gate protects
  decode by starving the tenant: with the engine ~100 % busy there is nothing to share in time.
- **D (no gate)**: TPOT +27 % at 1 QPS (16.4), +22 % at 2 QPS, +0.5 % at 3 QPS; **0 preemptions**
  now (26 mid-step preemptions on the tenant side instead); 4 QPS queue 3.3 s (1.2 alone).
  Tenant 11.0k / 4.9k / 0.15k / 12.7k / 0 / 16.3k tok/s by phase.
- **F (MPS 20 %) was invalid**: the VRAM scheduler's loop died at its first tick (a helper called
  before its definition in the SM-share edit), so the tenant saw no budget, held 8.7 GiB (cfg 1)
  throughout and the engine's queue grew to minutes. Kept in `results/mps20_broken_controller/`
  as the "what if nobody adjusts" accident; fixed and re-running.
- Green contexts work from Python through ctypes on libcuda (cuGreenCtxCreate etc.; driver
  580): a matmul on 16/32/64/128 of the 170 SMs runs at 24/48/95/187 TFLOP/s vs 241 on the full
  device; the tenant at 32 SMs trains 11.8k tok/s (29.6k on the full device at cfg 2).
- Schemes launched 00:40: F re-run, `green32` (tenant on 32 SMs, no gate), `green_dyn` (tenant SM
  share from λ̂: 128 / 96 / 64 / 32 / 16 at 0 / 0.5 / 1 / 2 / ≥3 QPS), `engine_green` (engine's
  steps on a 128-SM green stream + dynamic tenant). Prediction: green32 ≈ F; green_dyn gives the
  tenant more at low RPS (64–128 SMs) for the same decode cost at high RPS; engine_green either
  fails to start (CUDA-graph replay on a green stream) or costs decode little while giving the
  tenant disjoint SMs.

**Green-context engine feasibility (00:25–00:35, GPU 1, no DRAM tier because /dev/shm was full):**
vLLM 0.30 + kvcached starts with its steps wrapped in a 128-SM green stream (CUDA-graph replays
included), greedy outputs identical to plain on 8/8 prompts. Under a prefill-heavy load (2 QPS,
6k-token prompts recomputed every time, engine overloaded: TTFT 6 s) the 128-SM engine was 19 %
slower per token (TPOT 87.6 vs 73.6 ms) and TTFT 76 % longer — prefill is compute-bound, so
fewer SMs hurt it; the decode-only question is left to the `engine_green` scheme.

## D8. Green-context round, first pass (00:40–01:45)

- **F (MPS 20 %, valid re-run)**: TPOT +13 % at 1 QPS (14.6), +12 % at 2 QPS, +3 % at 3 QPS; queue
  wait +13 ms at 1 QPS; 4 QPS queue 3.8 s; tenant 6.7k / 3.6k / 0.1k / 6.8k / 0 / 7.7k tok/s.
- **green32 (tenant on 32 of 170 SMs, no gate)** ≈ F: TPOT 14.4 / 16.6 / 21.3 / 13.6 ms, queue
  119 / 132 / 370 ms, tenant 6.3k / 3.3k / 0.1k / 6.4k. Green contexts reproduce the MPS cap's
  trade-off without MPS's static percentage.
- **engine_green: the decode node on 128 of 170 SMs costs nothing** — TPOT 12.9 / 15.2 / 21.2 /
  12.5 / 26.1 ms and queue wait identical to the no-tenant baseline in every phase (the tenant
  was absent, see next point), confirming that a bandwidth-bound decode step does not need the
  whole SM array: 42 SMs can be handed to a tenant *exclusively*.
- **green_dyn / engine_green tenants died after 12 steps**: the VRAM scheduler's interpolated SM
  share changed by a few SMs almost every tick, the tenant re-created its green context every
  step, and the caching allocator's blocks bound to the old streams were never reused — reserved
  memory grew 6.2 → 8.1 GiB in four steps and the tenant OOM-ed down every tier. Fix: compare the
  *requested* count (the driver rounds), switch at most every 5 s, `empty_cache()` on a switch;
  VRAM scheduler quantises the share to multiples of 16 SMs.

## D9. Green-context round, second pass (01:50–02:40)

- **engine_green, decode side**: with the engine's steps on 128 of 170 SMs, TPOT 13.2 / 15.2 /
  20.9 / 12.5 / 27.0 / 12.0 ms and queue wait 109 / 124 / 249 / 105 / 912 / 95 ms — equal to the
  no-tenant baseline (12.9 / 15.2 / 21.2 / 12.5 / 26.0 / 12.0; 107 / 124 / 270 / 105 / 1 228 / 94).
  A bandwidth-bound decode node can give up a quarter of its SMs for free.
- **Dynamic switching of the tenant's green context is not robust in-process**: even with the
  requested-count compare, the 5-s rate limit and `empty_cache()`, reserved memory crept up
  across switches (7.7 GiB reserved for a 5 GiB configuration) and the tenant OOM-ed its way to
  pause within the first phase in all three schemes (green_dyn 15 OOMs, engine_green 9, green_opt 3;
  the decode side of those runs is therefore "tenant mostly absent"). Allocator blocks are bound
  to the stream they were allocated on; switching streams repeatedly fragments the pool.
- Decision for the last scheme: **no switching**. `green_split`: engine on the first 128 SMs, tenant
  on the *remaining* 42 (`cuDevSmResourceSplitByCount` remainder), exclusive by construction, the
  memory ladder unchanged. Prediction: decode = baseline in every phase (as engine_green); tenant
  ≈ 13k tok/s at 1 QPS (42 SMs; 32 gave 11.8k solo), cfg following the budget, 0 preemptions.

## D10. Final scheme, green_split (02:35–02:50) — and what the whole ladder says

Engine on SMs 0–127, tenant on the other 42, no switching: tenant 10.5k / 4.7k / 0.15k / 12.8k /
0 / 13.5k tok/s (≈ the un-capped co-run D), 0 engine preemptions, 21 tenant mid-step
preemptions, 6 tenant OOM retries. **Decode paid the same as D**: TPOT 16.3 ms at 1 QPS (+26 %),
18.3 at 2 QPS (+20 %), queue wait 123 / 145 ms, 4-QPS queue 4.0 s. Disjoint SMs did not remove
the interference, so it is not SM contention: the decode step streams the 15 GiB of weights
every ~13 ms and the tenant's GEMMs compete for the same DRAM bandwidth. Across the schemes the
decode cost tracks the tenant's throughput, not where it runs: 0 tok/s → 12.9 ms (C, E);
6.3–6.7k → 14.4–14.6 ms (G, F); 10.5–11k → 16.3–16.4 ms (K, D) — about +2.5 % TPOT per 1k
tenant tok/s at 1 QPS. Prediction (decode = baseline) was wrong; the earlier `engine_green`
decode-equals-baseline result was with the tenant absent. The lever that remains is the
tenant's *rate* (SM cap or green-context size), and it is a linear trade: pick the TPOT budget,
get the tenant throughput. Dynamic in-process green-context switching was not made robust
tonight (allocator pool bound to streams); a static partition plus the memory ladder is the
working configuration.

## D11. Overnight round 2 — start (2026-11-09 01:20): 500 W cap, accounting fix, reference re-run

Both GPUs capped at 500 W (persistence on) before anything ran. The VRAM scheduler now measures the
engine's non-KV footprint from NVML at start (16.97 GiB with the MPS daemon; the design constant
was 17.7), carries a separate tenant-fixed term (2.0 GiB: CUDA context + paused allocator floor), a
0.5 GiB safety term, and may raise the KV limit only up to what is physically free right now
(kvcached mapped + NVML free − safety); a wanted limit above that is reached in steps as the tenant
releases (`grow_partial` / `grow_wait` actions). The tenant hands cached-but-free blocks back as
soon as its budget shrinks. Smoke test (2 min, 1 → 3 req/s): grows 26, partial 6, waits 581 ticks,
KV in use never within reach of the tenant, 0 physical oversubscription.
**Prediction for the re-run set** (`adjust_notenant`, `static_split`, `adjust`, `adjust_nogate`,
`adjust_mps20`, `green_split`, then `tpot_slo5`, `tpot_slo15`): baseline TPOT within 3 % of the 600 W
run; tenant solo throughput −5–10 %; Figure-2 slope 2.0–2.6 %/k; "tenant holds more than its
budget" episodes < 10 s per run; at 4 req/s the queue wait rises (max limit ≈ 11.9 instead of 12.65
GiB). `tpot_slo5` / `tpot_slo15`: decode at baseline + SLO ± 3 points at 0.5–2 req/s, tenant ≈ 2k /
≈ 6.5k tok/s, points on the Figure-2 line.

## D12. P4 (a) — exportable kvcached pages for LMCache's VMM IPC (01:32, GPU 1, `.venv-lmc`)

kvcached main rebuilt with `prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`
behind `KVCACHED_EXPORTABLE=1`; LMCache 0.5.6 `lmcache_driven` + `use_vmm_api`, per-layer layout.
**Outcome: as predicted, one check further and no further.** The exportability check passes; the
connector then raises `tensor bytes [0, 428933120) exceed the 2097152-byte physical chunk at
0x1f0000000000: the tensor spans more than one cuMemCreate physical chunk … one exported handle maps
exactly one chunk` (`results/p4_vmm_exportable_startup.txt`, `logs/p4_vmm_exportable_vllm.log` in
`kvcached_validation`). kvcached backs each KV tensor with 2 MiB pages; LMCache's importer needs one
handle per tensor. Fixing this is on LMCache's side (import a page table, not a chunk). Stopped here.

## D13. P4 (b) — `engine_driven` under the stress load that crashed kvcached 0.1.6 (01:35–01:45, GPU 1)

Same load as kvcached_validation D7's stress variant (6 sessions, 6000-token first turns, +512 per
turn, think 2 s, 64 output tokens) plus a 2 G / 12 G resize every 15 s, 5 minutes, but on **kvcached
main (9106e0f) in `.venv-lmc`** instead of 0.1.6. Prediction (plan P4b): the fault reproduces within
1.5–2.5 min. **Outcome: it did not.** 180/180 requests ok, 166 stores / 123 retrieves, engine alive
at the end, no Xid in the kernel log, the only kvcached messages are the expected "pool exhausted →
scheduling miss" at the 2 G cap. Reading: the fault seen on 0.1.6 looks fixed upstream between
0.1.6 and main — main orders page unmaps behind the GPU's progress (`_kvcached_install_ordered_unmap`,
`physical_release_barrier`), which is exactly the window an asynchronous gather of a freed block
needs. The pin layer written for this step (`kvcached/integration/vllm/lmcache_pins.py`, env
`KVCACHED_LMCACHE_PINS=1`) is therefore kept optional and not required for the next test.
Next: a 10-minute run with resizes (`p4_ed_long`) to make the negative solid, then the output
check against the earlier plain-engine + connector outputs.
Output check (same 8 fixed prompts, greedy): the new kvcached-main + LMCache engine_driven engine
matches the no-connector plain engine and the no-connector kvcached engine **8/8** (cold and warm
identical). On 0.1.6 + the same LMCache, the connector changed 4/8 outputs (D7); that difference is
gone in this environment as well — the earlier connector runs agree with the new one on only 4/8.
**10-minute run (`p4_ed_long`, 01:42–01:53, 2 G / 12 G resize every 15 s):** 341/341 requests ok,
329 stores / 233 retrieves, engine alive, 0 Xid, outputs 8/8 identical to the plain engine; the only
kvcached warnings are scheduling misses and "shrink refused (in-use pages above target)" at the 2 G
cap, both expected. **Conclusion for P4: on kvcached main, LMCache 0.5.6's `engine_driven` MP
connector works with kvcached under heavy store traffic and live resizing; the 0.1.6 fault is gone
and the outputs are now correct. The pin layer was not needed. The server-side zero-copy modes
remain blocked by LMCache's one-handle-per-tensor import.** Time used: 25 min of the 3-h box.

## D14. P3 bring-up — the prefill tenant and /dev/shm (01:54)

First attempt (Qwen2.5-3B fp8, kvcached instance `vllm_c`, native host-RAM KV tier of 8 GiB) died at
start: `Insufficient space in /dev/shm: 8192 MiB required, 2283 MiB free`. vLLM's native offload
tier lives in /dev/shm (31 GiB on this box) and the 8B's 28 GiB tier is already there. Decision: in the
prefill-tenant schemes the 8B's tier is 25.5 GiB (32 prefixes × 0.766 GiB = 24.5 GiB still fit) and
the 3B's is 3.5 GiB (≈ 30 prompts of 3072 tokens; LRU beyond that — it is a sink for streamed KV,
not a cache the tenant reads back). Because the 8B's tier changes, a `pf_base` reference (8B alone,
25.5 GiB tier) is added before the three prefill schemes. **Prediction:** `pf_base` equals
`adjust_notenant` within noise (no prefix is evicted at 25.5 GiB). The bring-up test on GPU 1 is
repeated with a 2 GiB tier.
**Bring-up outcome (01:56, GPU 1, 2 GiB tier):** the 3B fp8 engine comes up beside kvcached
(`vllm_c`, 4.6 GiB on the GPU incl. its first 0.5 GiB of KV), serves prefill-only requests at
71 ms median for 3072-token prompts (6.1k prompt tok/s in a 2 req/s burst, 19/19 ok), and takes
kvcached limit changes (1 → 6 GiB) live. `/sleep`, `/wake_up`, `/is_sleeping` returned 404: they are
vLLM dev-mode routes (`VLLM_SERVER_DEV_MODE=1`); re-testing with it set. The 3B's fixed footprint
for the VRAM scheduler is measured by NVML at launch in the runner (expected ≈ 4.1 GiB + context).
**Sleep re-test (01:58, `VLLM_SERVER_DEV_MODE=1`):** `/sleep?level=1` takes the 3B from 4945 to
1377 MiB on the GPU (weights to host; what stays is the CUDA context and kvcached's segment),
`/wake_up` returns in 0.30 s and the engine serves again. The prefill schemes therefore run with the
sleep tier enabled (`PF_SLEEP=1`). **Predictions for `pf_mps` / `pf_mps20` / `pf_split` (plan P3):**
decode +25–40 % at 1–2 req/s with shared SMs; under +10 % with the 20 % cap or the disjoint
partition while the tenant keeps ≥ 60 % of its prefill throughput; the tenant sleeps in the 3 and
4 req/s phases and wakes within a few seconds of the budget returning.

## D15. `static_split` at 500 W ran with the idle gate on and the tenant starved (02:00)

The round-1 runner starts the static-split tenant with `--gate`; in round 1 that gate happened not
to bite (9.6–12.8k tok/s), in round 2 it does (0.1k tok/s, like the gated adaptive scheme), so the
run measured "static split + gate", not a static partition with a working tenant. Kept as
`static_split_gate`; `static_split` is re-run without the gate (MPS co-run, like `adjust_nogate`)
at the head of chain 2. **Prediction:** tenant 9–12k tok/s in every phase; decode TPOT +20–30 % at
0.5–2 req/s and seconds of queueing at 3–4 req/s (its 6.7 GiB cannot hold the in-flight set), i.e.
the round-1 picture.

## D16. Chain 1 at 500 W: the tenant stopped for good after two phases in every adaptive scheme (03:15)

**Observation.** In `adjust_nogate`, `adjust_mps20`, `adjust`, `green_split`, `tpot_slo5` and
`tpot_slo15` the tenant ran in the first two phases (9.8k / 1.1k tok/s in `adjust_nogate`) and then
stayed paused (0 tok/s) for the remaining 480 s, while round 1 had it running in every 0.5–2 req/s
phase. The VRAM-scheduler log shows why: 5285 of 6800 ticks are `grow_wait`. The demand model asked
for 12.65 GiB at the 3 req/s peak; with the tenant already paused only ≈12.4 GiB could ever be
mapped (kvcached mapped + NVML free − safety), so the grow stopped at 12.0 GiB. When the demand
fell back to 2–4 GiB the requested limit (`want_limit`) stayed at 12.65 > 12.0, the grow branch kept
winning (`grow_wait`), the shrink branch (guarded by `want_limit <= limit`) never ran, and the tenant
budget published for the pending grow (0 GiB) was never revised. The reference run
`adjust_notenant` is unaffected (no tenant, so the grow never had to wait: 146 grows, 29 shrinks).

**Fix (controller.py).** A pending grow is abandoned as soon as demand is back under the granted
limit: `want_limit` is reset to the limit, the tenant's budget for the granted limit is re-published
once kvcached's mapped bytes are under it (action `grow_abandoned`). The round-1 scheduler had no
`want_limit` and so no such state to get stuck in; the bug came with the free-memory cap (D11).

**Decision.** The six adaptive runs are archived under `results/_round2a_sticky_want/` (phase,
scheduler and tenant CSVs only) and re-run with the fix, then chain 2 as planned. Chain 2 had also
not started: its waiter polled `pgrep -f logs/chain1.sh`, which matched the waiter's own command
line. **Predictions for the re-run:** the tenant runs in every 0.5–2 req/s phase as in round 1
(un-gated MPS 8–12k tok/s at 1 req/s, 3–5k at 2 req/s, ≈0 at 3–4 req/s); the gated `adjust` still
starves; `tpot_slo5` holds decode within ≈ +5 % of the reference at 1–2 req/s with the tenant at
1–3k tok/s, `tpot_slo15` within ≈ +15 % with the tenant at 3–6k tok/s; `grow_abandoned` appears
after each peak and `grow_wait` runs are seconds, not minutes.
