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

**Outcome (03:27, `adjust_nogate` re-run):** as predicted. Tenant 9.8k / 3.4k / 0 / 12.6k / 0 /
14.5k / 4.3k / 0.4k / 1.2k / 11.5k / 12.8k tok/s across the eleven phases (round 1 at 600 W: 11.0k /
4.9k / 0.2k / 12.7k / 0 / 16.3k / 8.4k / 0.4k / 1.8k / 11.9k / 14.9k); decode TPOT p50 within 0.3 ms of
round 1 in every phase. Scheduler log: 27 `grow_abandoned`, `grow_wait` runs of 2–7 s except one
of 60 s, which is the whole 4 req/s phase (demand 12.65 GiB above the ≈12.4 GiB that can be
mapped with the tenant paused: a true capacity limit, not the bug). Chain continues.

## D17. P1.1 outcome — throughput-feedback schemes `tpot_slo5` / `tpot_slo15` (04:30)

D11 predicted that a tokens/s cap driven by the decode-TPOT error would hold the slowdown at the
target with the tenant at the throughput Figure 2's line implies (≈ 2 k tok/s for 5 %, ≈ 6 k for 15 %).
**Outcome.** `tpot_slo15`: decode +0–14 % vs the reference in every phase (+5/+6/+10/+5/+10/+14/+14 %
in the 0.5–2 req/s phases), tenant 1.3–5.3 k tok/s there, cap median 1.1 k, interquartile 0.5–3.3 k;
on the Figure-2 line (5 k tok/s ↔ +12 %). The SLO is met and the tenant keeps a third to a half of
the un-gated throughput. `tpot_slo5`: decode +2–7 %, tenant 0.4–3.8 k tok/s but mostly at the
500 tok/s floor (cap at the floor for most ticks, sleep 2.8 s per step): a 5 % target is inside the
noise of a 10-s TPOT window (the reference itself moves ±4 % between repeats of the same phase),
so the loop sees a violation most of the time and throttles. Lesson for the design: the feedback
target must sit above the TPOT measurement noise, or the window must be longer (then the loop is
slower than the 30–60 s phases). Figures: `tpot_slo*_timeline.png`.

## D18. P1.2 shape probe and P2 `green_dyn2` outcomes (05:05)

**Probe** (engine at a static 3 GiB KV limit, tenant at a fixed 7.5 GiB budget and a 4 k tok/s cap,
one 60-s window per tenant shape; `results/probe_probe.md/.png`): decode slowdown per 1 k tenant
tok/s is **1.3 % for k=4, 2.6 % for k=8, 5.9 % with activation offload, 8–9 % with 12–20 streamed
layers**. Smaller micro-batches re-read the weights more often per token, and the offload shapes
add host↔GPU copies: the cost of a tenant token is set by the memory traffic it causes, not by the
token itself. NVML's DRAM-busy figure drops for the offload shapes (52–60 % vs 67–79 %) while
their slowdown rises, so the busy counter alone is not the predictor; traffic per token is. This
is the quantitative basis for the "bandwidth-aware tenant shape" proposal (plan P1.2): under a
tokens/s cap the tenant should run its largest-batch shape that fits the budget, never a streaming
shape at high rate.

**`green_dyn2`** (prebuilt green contexts of 128/96/64/32/16 SMs, SM count from the arrival rate,
stream switch only): **the tenant survived the whole run** (898 steps, 174 context switches,
reserved memory ≤ 8.7 GiB under a ≤ 9.9 GiB budget — none of the round-1 allocator leak), tenant
8.3 / 2.8 / 0 / 9.7 / 0 / 11.9 / 3.3 / 0.4 / 0.8 / 8.8 / 10.3 k tok/s, decode **+10–15 %** in the
0.5–2 req/s phases against +26–31 % for the un-gated MPS scheme at similar throughput, and +15–28 %
for the fixed 42-SM split at similar throughput. Fewer SMs for the tenant at a given tokens/s
means fewer concurrent weight re-reads: the same mechanism as the probe. Prediction in the plan was
"survives; slowdown like `green_split`"; the outcome is better than that on the decode side.
Un-gated **`static_split`** (no gate, 6.7 GiB KV): tenant 11–13 k tok/s throughout, decode +30–40 %
at 0.5–2 req/s and queues of 8–26 s at 3–4 req/s and after (the round-1 picture, as D15 predicted).

## D19. P3 outcome — the prefill tenant (`pf_base`, `pf_mps`, `pf_mps20`, `pf_split`, 05:03–05:56)

**Predictions (D14):** `pf_base` = `adjust_notenant` within noise; decode +25–40 % at 1–2 req/s
with shared SMs, under +10 % with the 20 % cap or the partition while the tenant keeps ≥ 60 % of
its prefill throughput; the tenant asleep in the 3–4 req/s phases, awake within seconds after.
**Outcome.** `pf_base` matches the main baseline (TPOT within 0.2 ms except 4 req/s: 27.5 vs 26.1 ms,
queue identical) — the smaller host tier costs nothing. The three prefill schemes are
indistinguishable from each other: decode **+4–8 % at 0.5–1 req/s, 0 % at ≥ 2 req/s**, the tenant
delivering **6–9 k prompt tok/s at 0.5–1 req/s and nothing at ≥ 2 req/s**, asleep 100 % of those
phases, 0 failed requests. The compute mechanism did not matter because compute never became the
limit: the tenant's **4.5 GiB fixed footprint** (fp8 weights, CUDA context, kvcached segment,
measured by NVML at launch) leaves a pool of 9.4 GiB for KV + tenant KV, and the 8B's demand
exceeds that from 2 req/s on, so the adapter holds requests and the engine sleeps. While it
sleeps its real footprint is 1.4 GiB, but the scheduler's fixed allowance stays at 4.5 GiB: 3 GiB
are idle in every ≥ 2 req/s phase. The adapter-side latency numbers (median 45–190 s in the
sleeping phases) are queueing at the proxy, not prefill time (71 ms for 3072 tokens when awake).
One artefact: in `pf_split` the budget came back briefly inside the 4 req/s phase, the tenant woke
(2 k tok/s) and the engine's queue doubled (6.7 s vs 3.4 s) — the wake margin (0.5 GiB) is too
small for a tenant whose wake costs 3.5 GiB. **Conclusions:** a prefill engine is a workable
tenant only when its weights fit in what the decode node leaves at low load; its decode cost per
prompt token (+6 % for 6–7 k tok/s ≈ 1 % per 1 k) is below the fine-tune tenant's (2.2 % per 1 k),
consistent with prefill being compute-bound (weights read once per 8 k-token batch). Next steps
recorded in the report: credit the sleeping footprint to the pool, wake margin ≥ the wake cost,
a 16–32-SM green context for the prefill engine.

---

# Night of Nov 16: overnight plan 3 (C1 and C2)

Plan and predictions: [`../PLAN_overnight3.md`](../PLAN_overnight3.md), last written Nov 15 23:27 and not edited
since; the predictions below are copied from it. Results in `../results_o3/`, logs in `../logs_o3/`.

## D20. Start, conditions, and what differs from the plan (01:55–02:12)

- The machine had been rebooted and both cards were free. Jiaxuan set both to 500 W and allowed
  both cards. **Deviation:** the plan was written for one card in series; the real-engine block (R4, R5, R7)
  ran on GPU 0 and the synthetic block (R0–R3, R6) on GPU 1 at the same time, as on Nov 9. The
  baseline of the night was measured while GPU 1 was busy and again at the end.
- **Deviation, order of R4:** the two controls ran right after the baseline, because the C1 and C2
  runs need R2's table and R1's slice length from the other card.
- Preparation code written between 01:55 and 02:06 and checked on the CPU (`../test_o3_cpu.py`:
  the GB/s conversion, block pacing against a simulated clock, the slow loop replayed on the Nov 9
  TPOT log, the chunked loss against the usual loss and gradients), then a 4-minute smoke run on GPU 1.
- **Deviations in the preparation:** (1) kernel launches per second are not measured with the
  profiler; (2) the existing feedback loop already uses a 10-s median of TPOT, so the slow loop
  differs from it by the ±10 % step, the GB/s unit, the floor and ceiling, and by publishing the
  ceiling while no request is in the engine (the loop's own value is kept for the next arrival);
  (3) under block pacing a late slice restarts the schedule, lost time is not made up; (4) R6 has no
  register-only kernel (that needs a hand-written CUDA kernel): matrix products of 128, 512 and
  2304 squared (0.2, 3 and 64 MB working sets) and the 16-KB "tiny" kernel instead.
- **A fault of the night (02:33):** the real-engine launcher cleaned up "every tenant process" at
  the start of each scheme, a line written when only one card was in use. It stopped the tenant of
  R1 on the other card (R1 failed after 3 of 54 points) and cut one R2 run short (top 8 layers:
  35-s window instead of 43 s, still valid). Fixed at 02:43 (the cleanup now spares the synthetic
  runs' tenants); R1 was rerun in full after R3 (02:49–03:14). No real-engine run was affected.

## D21. R0 outcome — sanity and calibration (02:06–02:12)

**Prediction:** within 3 % of Nov 9 (11.7–11.8 / 13.1–13.4 / 15.4–15.8 ms; 3.5 k / 6.5 k / 11.6 k
tok/s; 1 % of the counter = 17.8 GB/s); per-step timing within 2 % of grouped timing; the hook
alone: no basis. **Outcome: met.** Decode alone on the full device 11.69 / 13.44 / 15.85 ms
(Nov 9: 11.68 / 13.39 / 15.80). Per-step timing (a CUDA event after each queued step) and grouped
wall-clock timing agree to 0.05 %. Tenant alone 3,493 / 6,518 / 11,582 tok/s on 8 / 16 / 32 SMs
(Nov 9: 3,490 / 6,543 / 11,642). Counter: 1 % = 17.79 GB/s over 30 points, error median 0.5 %,
max 1.0 % (Nov 9: 17.79). **The syncs of block pacing cost the tenant under 1 %:** with a cap it
never reaches, 11,510 tok/s with a sync after every block, 11,623 after every 8, 11,637 after
every 48, against 11,582 without.

## D22. R2 outcome — bytes per token of every shape and variant (02:12–02:30)

Tenant alone on 32 SMs, about 43 s per variant (`../results_o3/r2_bytes.md`).

| Variant | Predicted (CPU count) | Measured MB per token | vs usual | tokens/s |
|---|---|---|---|---|
| usual shape, 4 micro-batches | 13.4–14.3 (Nov 9) | 13.47 | – | 11,628 |
| 1 micro-batch | about 9 % fewer | 15.58 | **+16 %** | 14,543 |
| 2 micro-batches | – | 13.48 | 0 % | 13,416 |
| 8 micro-batches | about 12 % more | 14.88 | +10 % | 9,579 |
| activation offload / + 12 layers / + 20 layers in host RAM | no basis | 15.19 / 16.57 / 17.61 | +13 / +23 / +31 % | 3,246 / 2,566 / 2,364 |
| attention `eager` | about 37 % more | 20.66 | +53 % | 10,096 |
| `torch.compile` | no basis | **7.95** | **−41 %** | 14,457 |
| output layer and loss in chunks of 16 | no basis | 42.23 | **+214 %** | 7,706 |
| both together | no basis | failed to start (a bug in the helper, fixed after) | – | – |
| gradient checkpointing alone | no basis | 13.08 | −3 % | 8,857 |
| LoRA on the top 8 layers | about 36 % fewer | 10.10 | −25 % | 17,743 |
| LoRA on the top 4 layers | about 44 % fewer | 9.24 | −31 % | 20,198 |
| forward only | about 60 % fewer | 3.73 | −72 % | 25,010 |

- **`torch.compile` is the best "same training" variant: 41 % fewer bytes per token and 24 %
  faster.** It became V1. LoRA on the top 4 layers became V2 (the count overstated the saving of
  both top-layer variants by about 12 points).
- **The chunked output layer was a design error of mine.** Each chunk reads the whole output
  matrix (272 MB) again, 64 times per micro-batch, so the bytes triple. The logits are not what
  has to be avoided; the output matrix would have to be read once for all positions.
- **The count got the micro-batch direction wrong** for 1 micro-batch (+16 % measured, −9 % counted).
- **The slow shapes move few bytes per second**: 42–49 GB/s while running, against 143–181 GB/s
  for the fast shapes on the same 32 SMs, at only 13–31 % more bytes per token. If they cost
  decode as much as on Nov 9, bytes do not explain it (R5).
- On the full device: usual 14.30 MB per token at 28.5 k tok/s, top 4 layers 10.26 at 54.2 k.

## D23. R3 outcome — fewer bytes per token next to the synthetic decode step (02:33–02:49)

Tenant on 32 SMs, decode on the other 138; two passes, which differ by at most 0.2 points
(`../results_o3/r3_table.md`).

**Prediction (the C2 result):** at equal tenant tokens/s decode's slowdown falls by the same share
as the bytes per token, from 1.2–1.7 % to about 0.7–1.1 % per 1 k tok/s; all points on one line
of slowdown against tenant GB/s within ±20 %; flat out: no basis.

| Tenant, at the usual tenant's tokens/s (8.0 k / 9.2 k / 9.4 k) | 1 request | 4 | 8 | per 1 k tok/s |
|---|---|---|---|---|
| usual tenant (p) | +8.1 % | +13.9 % | +13.7 % | 1.02 / 1.51 / 1.46 % |
| `torch.compile`, paced to the same rate (r1) | +5.3 % | +9.7 % | +9.2 % | 0.66 / 1.06 / 0.99 % |
| top 4 layers, paced to the same rate (r2) | +5.2 % | +9.7 % | +9.4 % | 0.65 / 1.06 / 1.00 % |

**Outcome: met.** Decode's slowdown at equal tenant work falls by 30–36 % with both variants
(bytes per token: −41 % and −31 %). Against tenant GB/s (tokens/s × R2's MB per token) the five
settings at one request give 7.0–8.5 % per 100 GB/s: one line within ±10 %. Flat out, the
variants run faster (10.7 k and 14.4 k tok/s at one request) and decode is +7.2 % and +9.6 %:
the compiled tenant costs a little less than the usual one although it does a third more work,
the top-4 tenant costs more because it moves more GB/s. The usual tenant cannot be paced to the
top-4 tenant's GB/s (point s): it does not reach it. One more observation: the unpaced usual
tenant gives decode a p90 of +16 to +29 % at a mean of +8 to +14 %; the compiled one +12 to +17 %
at +7 to +12 %.

## D24. R1 outcome — same tenant work, limited in nine ways (rerun 02:49–03:14)

Two passes, at most 0.2 points apart (`../results_o3/r1_table.md`). Tenant work 2.4 k / 2.8 k /
2.9 k tok/s (the rate of 8 SMs flat out) unless stated.

| Setting | Predicted mean (1 / 4 / 8 requests) | Measured mean | Measured p90 |
|---|---|---|---|
| a: 8 SMs all the time | +5.4 / +7.5 / +6.4 % (Nov 9) | +3.9 / +6.9 / +5.9 % | +5.4 / +7.9 / +8.2 % |
| c: 32 SMs, pause after each step | +2.9 / +4.6 / +4.4 % | +2.5 / +5.7 / +5.5 % | +5.8 / +11.8 / +11.0 % |
| g: 32 SMs, slices of 1 block | as c | +2.6 / +5.8 / +5.5 % | **+2.2 / +5.5 / +5.2 %** |
| h: 32 SMs, slices of 8 blocks | as c | +2.6 / +5.9 / +5.6 % | +5.5 / +10.3 / +9.1 % |
| i: 32 SMs, slices of 48 blocks | as c | +2.5 / +5.8 / +5.5 % | +5.7 / +11.8 / +11.0 % |
| f: all SMs, pause after each step | +4.5 / +6.5 / +6.2 % | +4.6 / +7.0 / +6.5 % | +12.7 / +20.0 / +19.0 % |
| b: 16 SMs all the time (4.5–5.3 k tok/s) | +6.4 / +9.1 / +9.9 % (Nov 9) | +4.7 / +9.6 / +8.7 % | +6.2 / +10.4 / +12.9 % |
| d: 32 SMs paced to b's rate | +5.4 / +8.5 / +8.2 % | +4.5 / +8.7 / +8.4 % | +6.7 / +12.2 / +13.3 % |

- **Q1, partly met.** 32 SMs with pauses is cheaper than 8 SMs all the time at every batch size
  and by more than the spread between passes, so the rule of the plan is passed; but the gain is
  0.4–1.4 points, not the 2–3 predicted. All SMs with pauses is no better than 8 SMs, as predicted.
- **Why the gain is smaller: pauses are not free at 4 and 8 requests.** While the tenant pauses,
  decode is +0.7 % at one request but +3.0 % and +2.7 % at 4 and 8 (the prediction took 0). While
  it runs: +8.1 / +14.0 / +13.5 %, as flat out. Checked afterwards in the same logs: it is not the
  power limit (the clock is 0.5 % lower than alone, the power 475 W on average), and it is not a
  recovery after each burst (the offset is the same 50 ms and 1 s after the tenant stops:
  +2.4 to +3.0 %). An idle second process does not cause it either (R6: decode next to the idle
  arithmetic job equals decode alone). The cause is open.
- **Q3, met.** Slices of one block (about 2.4 ms) bring the p90 of single decode steps down to the
  mean; slices of 8 blocks halve the gap; slices of 48 blocks change nothing. The tenant's speed
  while running falls by 5.0 % at worst with one-block slices, and its realised rate is 4.4 % under
  the cap. By the rule of the plan `c1_fine` runs with slices of one block.
- **Not predicted:** the p99 of single decode steps is +26 to +32 % in every 32-SM setting,
  whatever the pacing. The output layer, the loss and the optimizer step are outside the blocks
  and are not sliced; that is the likely source, not verified.

## D25. R6 outcome — the floor: a second job that moves no bytes (03:14–03:18)

Arithmetic-only second job, flat out, on 32 and on 8 SMs, decode on the remaining SMs at 1 and 8
requests (`../results_o3/r6_floor_summary.md`). **Prediction: none (no basis); three readings to
tell apart.** Known from Nov 9: +1.4–2.6 % (one request) from 8 SMs.

| Second job | Launches/s next to decode | Decode, 1 / 8 requests, job on 32 SMs | job on 8 SMs | Power, clock (32 SMs) |
|---|---|---|---|---|
| 16-KB add ("tiny") | 95–124 k | +4.8 / +3.8 % | +5.3 / +3.2 % | 435–451 W, clock unchanged |
| 128 × 128 product (0.2 MB) | 48–67 k | +5.1 / +4.4 % | +5.3 / +3.3 % | 441–459 W, unchanged |
| 512 × 512 product (3 MB) | 5–22 k | +4.4 / +5.0 % | +2.6 / +6.9 % | 500 W, clock −2 to −5 % |
| 2304 × 2304 product (64 MB) | 0.1–0.6 k | +5.7 / +10.2 % | +2.4 / +5.4 % | 500 W, clock −6 to −7 % |

**Outcome: all three readings hold, in different cases.**
1. **Launches.** About 100 k launches per second of a kernel that moves 16 KB slow decode by
   3–5 %, with the card under its power limit and the clock unchanged.
2. **The power limit.** Arithmetic on 32 SMs brings the card to its 500-W limit; the clock falls
   by 2–7 % (2,932 → 2,872–2,723 MHz) and decode slows by 4–10 %. On 8 SMs the limit is not
   reached and the clock does not move.
3. **The shared cache.** With 8 requests and the larger working sets the device counter reads
   6–8 % more traffic than decode's own (decode's known bytes ÷ counter = 0.92–0.95): the second
   job's working set does not stay in the cache next to decode, and it is in these cases, on
   8 SMs with no clock change and few launches, that decode loses 5–7 %.

So the floor under C2 is not one number: fewer launches (merged kernels) lower it, and so does
headroom under the power limit. The idle baseline drifted by +1 % at one request over each block.

## D26. R4 outcome — C1 and C2 on the real engine, reference schedule (02:06–04:14)

Ten runs of the 11-minute schedule on GPU 0 (`../results_o3/r4_summary.md`, `../analyze_o3.py`;
figure `figures/o3_c1_c2.png`). Order: baseline, `tpot_slo15`, `green_dyn2`, `c1_fix`, `c1_slow15`,
`c1_slow5`, `c1_fine`, `c2b`, `c2a`, baseline. No request failed in any C1 or C2 run.

**The baselines and the controls reproduce.** TPOT median 12.0 / 12.7 / 16.4 ms at 0.5 / 1 / 2
req/s (Nov 9: 12.0 / 12.6 / 16.3; predicted within 2 %). The closing baseline: same medians, mean
token gap within 0.4 % of the opening one. `tpot_slo15`: median +7 / +10 / +7 %, p90 +28 / +15 /
+8 % (Nov 9: +5 / +11 / +6 and +29 / +16 / +7). `green_dyn2`: +14 / +15 / +9 % and +16 / +13 /
+5 % (Nov 9: +14 / +15 / +9 and +15 / +16 / +9). Both within the predicted 2 points except
`green_dyn2`'s p90 at 1 and 2 req/s (3–4 points lower). **One limit:** the opening baseline ran
while GPU 1 was busy and its per-request p90 is 3 % above the closing one (14.5 / 19.6 against
14.1 / 18.9 ms at 1 / 2 req/s); the runs before 03:22 had GPU 1 busy, `c1_fine`, `c2b` and `c2a`
had it idle. Comparisons of p90 across that line carry about 3 points of doubt; the means do not.

| Scheme, 0.5 / 1 req/s | Tenant tok/s | Mean token gap | per 1 k tok/s | TPOT median | TPOT p90 |
|---|---|---|---|---|---|
| `tpot_slo15` (fast loop, all SMs) | 3,769 / 3,953 | +12.2 / +11.0 % | 3.2 / 2.8 % | +7 / +10 % | +28 / +15 % |
| `green_dyn2` (16–96 SMs by load) | 11,468 / 9,154 | +13.4 / +13.8 % | 1.2 / 1.5 % | +14 / +15 % | +16 / +13 % |
| `c1_fix` (32 SMs, 50 GB/s, pause after a step) | 3,524 / 3,670 | +5.6 / +6.3 % | 1.6 / 1.7 % | +6 / +6 % | +6 / +4 % |
| `c1_fine` (the same in 1-block slices) | 3,195 / 3,348 | +5.7 / +5.9 % | 1.8 / 1.8 % | +6 / +6 % | +6 / +4 % |
| `c1_slow15` (slow loop, +15 %) | 3,686 / 4,167 | +5.5 / +6.2 % | 1.5 / 1.5 % | +6 / +7 % | +7 / +5 % |
| `c1_slow5` (slow loop, +5 %) | 2,622 / 1,442 | +3.5 / +2.6 % | 1.3 / 1.8 % | +3 / +4 % | +4 / +0 % |
| `c2a` (`c1_fix` with the compiled tenant) | 3,524 / 3,688 | +4.3 / +4.0 % | 1.2 / 1.1 % | +5 / +4 % | +4 / +2 % |
| `c2b` (`c1_fix` with LoRA on the top 4 layers) | 3,524 / 3,670 | +5.6 / +4.8 % | 1.6 / 1.3 % | +5 / +5 % | +8 / +3 % |

Predictions against outcomes:

- **`c1_fix`, met.** Predicted: tenant 3.5–3.7 k tok/s; decode on average +5.3–6.3 % against
  +10.4 / +9.6 % for `tpot_slo15` at the same tenant work; p90 within 2 points of the median.
  Measured: 3,524 / 3,670 tok/s; +5.6 / +6.3 % against +12.2 / +11.0 %; p90 +6 / +4 % at a median
  of +6 / +6 %. At the same tenant work the byte allowance on 32 SMs halves decode's cost and
  removes the tail of the fast loop.
- **`c1_fine`, half met.** Predicted: per-request TPOT within 1 point of `c1_fix` (met: identical);
  the p90 of single token gaps down from the "while running" level to within 2 points of the
  mean (**not met**: +7.9 / +14.3 % against +8.2 / +14.7 %). On the real engine the tail of the
  token gaps does not come from the tenant's bursts at this allowance, so slicing them changes
  nothing visible, and it cost the tenant 9 % of its tokens (lost time is not made up). The
  synthetic result of R1 does not carry over.
- **`c1_slow15`, partly met.** Predicted: the allowance does not bind at 0.5 and 1 req/s, decode
  +9–13 %, tenant work 1.6–2 times that of `tpot_slo15`, p90 at 0.5 req/s under +20 %. Measured:
  the allowance did bind in all but the last phase (median 36–60 GB/s per phase, 113 in the last; range within a factor 1.4–4.1, once 8.5),
  decode +5.5 / +6.2 %, tenant work equal to `tpot_slo15` (3,686 / 4,167 tok/s), p90 +7 % at
  0.5 req/s (met). The loop is more careful than its target: its error signal (10-s median TPOT
  against the phase baseline at the estimated arrival rate) reads high whenever requests overlap.
- **`c1_slow5`, partly met.** Predicted: the allowance settles near 40–45 GB/s, decode median
  +4–6 % at both rates, range within a factor 2. Measured: the allowance sat at or near its floor
  at 1 req/s (median 10–16 GB/s per phase; 21 GB/s at 0.5 req/s; factor 1.1–3.1), decode median +3 / +4 %,
  tenant 2,622 / 1,442 tok/s. The target is held at both rates, which the fast loop did not do
  (+11.2 % and +2.8 % on Nov 9), at the price of tenant work.
- **`c2a`, met at 1 req/s, just above at 0.5.** Predicted: same tokens/s as `c1_fix`, decode's
  mean slowdown down from +5.3–6.3 % to about +3.0–4.0 %, p90 within 2 points of the median.
  Measured: +4.3 / +4.0 % against +5.6 / +6.3 % (−23 % and −37 %), p90 +4 / +2 %. The compiled
  tenant computes 42–44 % of the time against 54–62 %. Two restrictions I added on the night, after
  a test on GPU 1 (03:18–03:22) showed a 46-s stall when a shape is first compiled in mid-run: the
  compiled tenant compiles its two fitting shapes at start (34 s) and uses only those, and it does
  not run the shapes with offload. So at 2 req/s it pauses where `c1_fix` runs a slow shape:
  1,502 against 1,809 tok/s, and decode +2.5 % against +7.3 %.
- **`c2b`, not met at 0.5 req/s, half met at 1 req/s.** Same tenant tokens/s as `c1_fix`; decode
  +5.6 / +4.8 % against +5.6 / +6.3 %. The saving seen on the synthetic step (R3: −30 to −36 %)
  shows only at 1 req/s (−24 %).

**The load served within the SLO** (my proposed targets: TPOT ≤ 20 ms and TTFT ≤ 500 ms per
request; requests/s within both over the whole schedule): baseline 1.07; `tpot_slo15` 0.97;
`green_dyn2` 1.03; `c1_fix` 1.01; `c1_fine` 1.02; `c1_slow15` 1.01; `c1_slow5` 1.02; `c2a` 1.03;
`c2b` 0.98. All requests meet the targets at 0.5 and 1 req/s in every scheme; the differences
come from 2 req/s (baseline 91 %; `tpot_slo15` 78 %; `c1_fix` 82 %; `c2a` and `c1_slow5` 88 %) and
3 req/s, where the engine alone already misses them for two thirds of the requests.

**What this decides.** C1 is kept in its first method with a fixed or slowly moving allowance;
the second method is not worth its cost on this engine. C2 holds on the real engine for the
compiled tenant; for the top-4-layers tenant the gain is small at these rates. C1 + C2
(`c2a`) costs decode 1.1–1.2 % per 1 k tenant tok/s against 1.2–1.5 % for `green_dyn2` and
2.8–3.2 % for `tpot_slo15`, with the smallest tail of all schemes that train.

## D27. R5 outcome — every shape under the same allowance, real engine at 1 req/s (04:14–04:43)

Tenant on a 32-SM green context with the 50 GB/s allowance, engine KV limit 3 GiB, 120 s per
shape, two passes in opposite order, three no-tenant windows (mean token gap 12.53 / 12.36 /
12.70 ms). `../results_o3/probe2_summary.md`, `../analyze_probe2.py`.

**Prediction:** while the tenant runs, decode +8–12 % with every shape; the fastest shape computes
50–65 % of the time, the three slow ones all the time and below the allowance; decode on average
+5–6 % with the fastest shape and +8–12 % with the slow ones at less than half the tokens.
Whether a slow shape's GB/s predicts its cost: no basis.

| Shape (both passes) | MB per token (R2) | Tenant tok/s | Share computing | GB/s while computing | Decode, mean token gap | per 1 k tok/s | per 100 GB/s |
|---|---|---|---|---|---|---|---|
| 4 micro-batches | 13.5 | 3,701 | 62 % | 80 | +6.0 % | 1.6 % | 12 % |
| 8 micro-batches | 14.9 | 3,321 | 75 % | 66 | +7.5 % | 2.3 % | 15 % |
| + activations in host RAM | 15.2 | 2,086 | 100 % | 32 | +10.0 % | 4.8 % | 32 % |
| + 12 layers in host RAM | 16.6 | 1,818 | 100 % | 30 | +14.0 % | 7.7 % | 46 % |
| + 20 layers in host RAM | 17.6 | 1,670 | 100 % | 29 | +13.0 % | 7.8 % | 44 % |

**Outcome: the shares and the fastest shape as predicted; the slow shapes a little worse
(+10–14 %), and bytes do not explain them.** The slow shapes move 29–32 GB/s, four tenths of the
fast shapes' traffic, and slow decode by more: three to four times the cost per byte. A byte
allowance therefore does not charge them correctly; they stay under any allowance a fast shape
would be given. What they do besides moving bytes on the GPU: copies between host RAM and GPU
memory for every block, and many small kernels; which of these decode pays for is not measured.
Next to the engine they also run a third slower than alone (2.1 k / 1.8 k / 1.7 k against 3.2 k /
2.6 k / 2.4 k tok/s in R2). The two passes differ by up to 4.5 points for one slow shape
(+11.7 and +16.2 %), so the order among the three slow shapes is not resolved.
**Decision:** the shape rule inside C2 is "next to decode, do not run a shape with offload; pause
instead" (R4's `c2a` did this at 2 req/s: decode +2.5 % against +7.3 %, tenant tokens −17 %).

## D28. R7 outcome — prefill tenant on scheduler-driven green contexts: not obtained (04:43–05:40)

**Prediction (Nov 10):** decode's cost does not change, because this tenant is limited by GPU
memory; its own prefill slows down. **Outcome: no result; the mechanism faults.**

- `pf_mps` (reference, 04:43–04:57) ran normally.
- `pf_green_dyn` (04:57–05:10): the tenant engine built its five green contexts, ran its start-up
  steps on the default stream, and at the first step of the load moved to the 128-SM context.
  At that moment (05:00:01) the driver logged Xid 31 (MMU fault); the fault took the decode engine
  down too (both are clients of one MPS server), and every request of the run failed.
- My first reading was that CUDA graphs captured on the default stream cannot be replayed on a
  green context. I repeated both arms with the tenant engine in `--enforce-eager`
  (`pf_mps_eager` 05:12–05:25 ran normally; `pf_green_dyn_eager` 05:25–05:40). **The fault came
  back at the first switch** (54 of 588 requests succeeded), so CUDA graphs are not the cause, or
  not the only one.
- What is known: on Nov 9 an engine that was on one green context from its first step ran without
  fault (`pf_split`); the fine-tune tenant switches between prebuilt green contexts without fault.
  What differs here is an engine that leaves the stream it started on. Untested next step: put the
  engine on the largest prebuilt context from its very first step and switch only among green
  contexts.
- No other run of the night was affected; GPU 0 was released after each failed run.
