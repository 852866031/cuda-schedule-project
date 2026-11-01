# Decision log — another model's prefill on the decode GPU

Run-level decisions for the eighth study (`scripts/inf_pf_coloc/`), in order, with the
reason and — for experiments — the prediction written **before** the run and the outcome
after. Study-level predictions are in `PLAN_PF_COLOC.md`. Times are local.

## A. Design decisions (2026-10-13, before any run)

| decision | why | who |
|---|---|---|
| Big model = Llama-3-8B split (prefill GPU0, decode GPU1, KV offloaded); small = Qwen2.5-3B, not offloaded, prefill on GPU1 | the experiment as specified | **user** |
| Three placements: P1 prefill only / P2 prefill + KV to DRAM / P3 prefill → DRAM → decode on GPU0 | as specified | **user** |
| Record TTFT, throughput, and TPOT only where there is a decode | as specified | **user** |
| 3B workload = fresh 6144-token prompt per request | a session-reuse workload would hit the prefix cache and prefill only 128 tokens — no compute to add | **user** (asked) |
| 3B at 2 QPS first; QPS sweep afterwards, confirmed before running | | **user** (asked) |
| 8B decode budget 22 GiB | b24 leaves 6.8 GiB on GPU1 (a 3B needs 7.7); b20 already misses 2 QPS (208 tok/s); b22 held it (250 tok/s) | Claude |
| 8B prefill util 0.80 → 0.74 in every cell | the 3B decode of P3 needs 7.7 GiB on GPU0 (24.8 + 7.7 > 31.8); one 8B config for all cells means one baseline | Claude |
| MPS in every cell, baseline included | mandatory for co-resident engines (studies 5–7); baseline under MPS removes the MPS-client offset the fifth study could not attribute | Claude |
| `OMP_NUM_THREADS=8` in every cell | up to 4 engines; rule is ≤ cores ÷ engines; 2/4/8 were identical in the multi-model study | Claude |
| 3B prefill engine: whole-prompt prefill (`--no-enable-chunked-prefill`, batched 8192), util 0.22, `max-num-seqs 16` | mirrors the 8B's prefill engine; sizing verified by a smoke launch | Claude |
| 3B store capped at 4 GiB, oldest-first; late reads counted | the stock LMCache store has no evictor — 300 fresh prompts would write 65 GiB | Claude |
| 3B launches only after the 8B warmup has filled its store; RAM gate on the measured headroom (floor 4 GB, ≤ 3 GB swap allowance) | host RAM is the feasibility risk; projecting from an empty store would hide it | Claude |
| Baselines: 8B alone, and the 3B alone in each placement | every colocated number needs its own solo reference | Claude |
| Raw records are never overwritten (driver refuses an existing name) | repo rule | Claude |

## B. Log

*(appended as it happens)*
- **10-13, smoke (3B alone, 30 requests, `pfc_s1_smoke`, `pfc_s3_smoke`) — sizing only.**
  Plumbing works end to end (store cap, router, decode retrieval: 0 late reads, decode
  engine prefilled 133 ms/request = last-token recompute, not a full prefill). Sizing:
  3B prefill engine 7.5–7.6 GiB on GPU1, KV grant 0.33 GiB (P1) / 0.46 GiB (P3) — one
  6144-token prompt (0.21 GiB) at a time, which is all whole-prompt prefill at batched
  8192 admits anyway; 3B decode on GPU0 7.8 GiB, grant 0.94 GiB. Host RSS: prefill engine
  3.6 GB (no connector) / 4.5 GB (LMCache), decode engine 4.4 GB, store = its cap.
  **Seen already, before the real runs:** the prefill step is 204 ms without a connector
  and **501 ms when the KV is written to the store** — storing costs more than the
  prefill itself (PLAN predicted < 20%). At 2 QPS that puts P2/P3 at the edge of the
  prefill engine's capacity even alone; the 300-request solo cells will show whether it
  queues.
- **Decision: 3B store cap 4 → 2 GiB for all real cells.** The store sat at its cap the
  whole run (it only evicts at the cap), so the cap is pure host RAM, and host RAM is what
  decides whether P2/P3 fit beside the 8B. 2 GiB = 9 prompts in transit; a late read is
  counted and invalidates the cell.
- **Round 1 queued:** `base s1 s2 s3 p1 p2 p3` at 3B 2 QPS, 300 requests each,
  `OMP_NUM_THREADS=8`. Predictions: `PLAN_PF_COLOC.md`. Added prediction for s2/s3 (from
  the smoke): alone at 2 QPS the 3B prefill engine runs at ρ ≈ 1 and queues (TTFT p50 in
  seconds, growing through the window).
- **10-13 15:25–15:57 — round 1 outcome** (`pfc_{base,s1,s2,s3,p1,p2,p3}`, 3B at 2 QPS).
  No stall in any cell; zero failed requests in six of seven.

  | cell | 8B TTFT p50 | 8B TPOT p50 | 8B tok/s | 8B decode queue | 3B TTFT p50 | 3B TPOT | 3B throughput | GPU1 SM-active / occupancy |
  |---|---|---|---|---|---|---|---|---|
  | base | 176 ms | 31.6 ms | 250.8 | 1.07 s | — | — | — | 0.52 / 0.031 |
  | s1 | — | — | — | — | 210 ms | n/a | 2.14 req/s | 0.38 / 0.055 |
  | s2 | — | — | — | — | 576 ms | n/a | 2.14 req/s | 0.39 / 0.057 |
  | s3 | — | — | — | — | 574 ms | 12.0 ms | 2.11 req/s, 271 tok/s | 0.39 / 0.057 |
  | p1 | 195 ms | 274 ms | 202.2 | 26.4 s | 231 ms | n/a | 2.14 req/s | 0.66 / 0.064 |
  | p2 | 185 ms | 263 ms | 204.7 | 24.9 s | 853 ms | n/a | 2.13 req/s | 0.66 / 0.065 |
  | p3 | *killed by mem_guard 68 s into the window — not a result* | | | | | | | |

  Against the predictions:
  - **B0 — partly wrong.** TPOT (31.6 ms) and throughput (250.8) reproduce b22; TTFT p50
    is 176 ms, not ~110. Not yet attributed (prefill util 0.74 vs 0.80, or MPS).
  - **S — s1 right (210 ms, step 235 ms); s2/s3 as the smoke warned:** writing the KV
    doubles the prefill step (235 → 462 ms) and the engine queues (142 ms mean) alone.
    s3's 3B TPOT is 12 ms, not 7–8.
  - **P1 — direction right, size wrong.** GPU1 SM-active 0.52 → 0.66 and occupancy
    0.031 → 0.064 (predicted > 0.9 / > 0.25: far too high). The 8B does not slow per
    token (ITL p50 14.4 → 14.5 ms); it **saturates**: each request stays on the decode
    engine longer (decode phase 2.8 → 3.7 s, KV load 286 → 574 ms), the 5.77 GiB grant
    holds ~7 requests, and the scheduler queue grows through the whole window (mean
    26 s; per-request TPOT 85 → 337 ms from the first to the last sixth). Throughput
    202 tok/s (predicted 230–245). The 3B is nearly untouched (231 vs 210 ms) —
    predicted 0.8–2 s, wrong.
  - **P2 — 8B side right, 3B side wrong.** The 8B is the same as P1 (205 tok/s, 24.9 s
    queue; TTFT 185 ms, not higher than P1). The 3B's TTFT is 853 ms vs 576 alone (+48%,
    predicted +10–20% over P1): its store step slows to 576 ms and at 2 QPS it queues.
  - **P3 — refused by RAM, as predicted,** but by the guard rather than the gate: 6.0 GB
    available after launch, 4.0 GB 68 s into the window → mem_guard killed the engines
    (218 + 156 requests failed by the kill). The gate's 3 GB swap allowance was too
    generous. P3 does not fit this box beside the 24 GiB 8B store.
  - **Confound in p1:** 0.9 GB swapped in during the first ~24 s of its window (pages
    pushed out during earlier cells). The queue keeps growing after the swap-in stops,
    so swap is not the cause — rerun queued to confirm (`pfc_p1_r2`).
- **16:04 — `pfc_p1_r2` outcome: reproduces.** 8B 202.5 tok/s, TPOT p50 276 ms, TTFT
  183 ms; 3B TTFT 229 ms, 2.14 req/s; GPU1 SM-active 0.66 / occupancy 0.064. Both p1 runs
  had swap-in in the window (0.9 and 1.7 GB), but **p2 had almost none (1,346 pages) and
  shows the same 8B saturation** (204.7 tok/s, 24.9 s queue), so swap is not the cause.
  Why the p1 launches swap and p2's does not is not explained (8B engine RSS reads 7 GB
  in the p1 launches vs 10 GB elsewhere).

## C. Round 2 (requested by the user after round 1)

- **16:25 — Correction to the round-1 reading of P2 (the user asked whether writing to DRAM
  should slow TTFT at all, and whether the implementation is at fault).** Not a harness
  bug, and my wording was wrong. What the engine log shows per request in `pfc_s2`:
  forward pass ~205 ms, then LMCache's GPU→host copy, **55 ms** (`offload_time`; the push
  to the store server is asynchronous, `put_time` 0.3 ms). vLLM calls
  `kv_connector.wait_for_save()` inside `execute_model`, *before* the step's outputs are
  returned (`kv_connector_model_runner_mixin.py`), so the first token waits for that
  copy and the engine is blocked for it. So the store adds ~55 ms of service time per
  request alone (96 ms beside the 8B), not "doubles the step": vLLM's
  `request_prefill_time` (462 ms) includes time a scheduled request spends behind the
  previous one. The rest of the 210 → 576 ms TTFT is queueing — whole-prompt prefill
  admits one request per step, and 2 QPS × ~0.3 s is ρ ≈ 0.6. *Check built into R2b:*
  at 0.5 QPS the s2 TTFT should fall to ≈ forward + copy ≈ 270–300 ms.
  The 24 "Failed to allocate memory block" warnings per request are the 1 GiB L1 being
  permanently full (each chunk evicts an old one); they cost no measurable time.
  Returning token #1 before the copy would need a patch to the connector (defer the
  save past the step) — not done; the 8B's prefill engine has the same property.
- **Round 2 queued (`queue_round2.sh`), by the user's instruction:** R2a baseline at
  prefill util 0.80; R2b 3B rate sweep 0.5 / 1 QPS for P1 and P2 (alone and colocated;
  75 / 150 requests so the 3B window stays ~150 s); R2c P3 with the 8B's L1 pools at
  4 GiB (own baseline `pfc_base_l1x4`).
  *Predictions.* **R2a:** TTFT p50 back to ~110–125 ms (the smaller prefill grant is the
  cause; MPS is not). **R2b:** the 8B is at its knee alone, so any 3B rate costs it
  something: at 0.5 QPS 8B ≥ 240 tok/s with decode queue 2–4 s; at 1 QPS ~225–235
  tok/s, queue growing (≥ 8 s). P1 ≈ P2 on the 8B side. 3B: s1 ~205 ms at both rates;
  s2 ~270–300 ms at 0.5 QPS, ~350 at 1; colocated +10–25%. GPU1 SM-active ~0.56 / 0.61.
  **R2c:** base_l1x4 ≈ base (250 tok/s, TPOT ~32 ms; more L1 warnings); p3 fits (≥ 6 GB
  available throughout) and the 8B saturates as in p1/p2 (~200 tok/s); 8B TTFT p50
  250–350 ms (GPU0 now shared with a decode); 3B TTFT ~0.9–1.1 s, TPOT 12–14 ms,
  ~256 tok/s. If available RAM still falls under the 4 GB floor, P3 is recorded as not
  fitting.
- **18:11–19:03 — round 2 outcome.** No stall, no failed request in any of the 11 cells.

  | 3B rate (realised req/s) | 8B alone | + P1: 8B tok/s / decode queue / TPOT p50 | + P2: 8B tok/s / decode queue / TPOT p50 | 3B TTFT p50: s1 → p1 | 3B TTFT p50: s2 → p2 | GPU1 SM-active / occupancy (p1) |
  |---|---|---|---|---|---|---|
  | 0 | 250.8 / 1.1 s / 31.6 ms | | | | | 0.52 / 0.031 |
  | 0.5 (0.59) | | 247.5 / 2.4 s / 45 ms | 245.0 / 3.2 s / 51 ms | 209 → 224 ms | 266 → 322 ms | 0.57 / 0.041 |
  | 1 (1.12) | | 232.5 / 8.4 s / 93 ms | 228.6 / 9.5 s / 102 ms | 209 → 225 ms | 305 → 460 ms | 0.60 / 0.050 |
  | 2 (2.14) | | 202.2 / 26.4 s / 274 ms | 204.7 / 24.9 s / 263 ms | 210 → 231 ms | 576 → 853 ms | 0.66 / 0.064 |

  - **R2a — right, more than predicted.** 8B alone at prefill util 0.80: TTFT p50 **77 ms**
    (176 at 0.74); TPOT 31.6 ms and 250.9 tok/s unchanged. The baseline's TTFT is the
    smaller prefill grant (4.78 vs 6.66 GiB: fewer hot prefixes resident, more reloads),
    not MPS. All colocated cells use 0.74, so comparisons inside this study hold.
  - **R2b — right.** The 8B has no headroom at b22: 0.5 QPS of 3B prefill already costs
    1% throughput and doubles its queue; 1 QPS costs 7% (queue 8–9 s, growing); 2 QPS 19%.
    P1 ≈ P2 on the 8B side at every rate. In P1 the 3B is flat at ~225 ms at every rate
    (+7% over alone). **s2 at 0.5 QPS = 266 ms = forward + the 55 ms copy** — confirms
    the reading of P2 above; the rest at higher rates is queueing.
    Each +1 req/s of 3B prefill adds ~0.07 SM-active on GPU1 while it is serving the 8B
    (0.19 when alone): the gain is partly cancelled by the 8B doing less.
  - **R2c — P3 fits with 4 GiB L1 pools, but the 8B config is worse and the cell is
    swap-backed.** `pfc_base_l1x4`: TTFT p50 257 ms (176 with 6 GiB), TPOT 35.7 ms,
    247.9 tok/s, queue 1.5 s — predicted "≈ base", wrong on TTFT. `pfc_p3_l1x4`: zero
    failures, 0 late store reads; 8B 194.5 tok/s, TTFT 296 ms (p95 1.5 s), decode queue
    31 s; 3B TTFT 873 ms, TPOT 11.95 ms, 270 tok/s; GPU0 SM-active 0.03 → 0.39, GPU1 0.64.
    **Confound:** 2.8 GB swapped in and 2.1 GB out during the window, although 12–16 GB
    read as "available". The kernel swaps anonymous pages while keeping ~12 GB of page
    cache (model files); no sudo to change swappiness or drop caches. The same thing
    explains the earlier p1 launches whose 8B engines read 7 GB RSS instead of 10.
    P3's 8B numbers are therefore an upper bound on the cost, not a clean measurement.

## D. Asynchronous KV save (requested by the user, 19:10)

- **Decision (user): patch the connector so token #1 is returned before the GPU→host
  copy.** `scripts/inf_pf_coloc/async_save_patch/sitecustomize.py`, loaded only into the
  3B prefill engine (`--sm-async-save`, cells get an `_async` name). Mechanism: LMCache
  0.4.4's `get_finished()` returns `(None, None)` — it never uses vLLM's delayed-free
  path. The patch hands the copy to one background thread, has `request_finished`
  return delay-free while a copy is in flight, and reports `finished_sending` from
  `get_finished` when it is done. Known limit: a request preempted mid-copy is not
  protected (preemptions recorded; expected 0).
  **P3 hazard, checked not assumed:** the router sends the decode leg as soon as token
  #1 arrives; with the copy now after the token the decode may look the KV up before it
  is stored and silently re-prefill on GPU0. The driver now counts, per P3 cell, how
  many decode requests found their whole prompt in LMCache.
- **Plan and predictions.** (1) smoke `s2`, `s3` async alone — correctness: 0 errors, 0
  preemptions, and in s3 the decode still gets full hits (or the miss count says the race
  is real). (2) `s2 p2` async at 2 QPS, then `s3` if (1) is clean.
  *Predictions:* s2 async at 2 QPS: TTFT p50 576 → ~300–350 ms (the copy leaves the
  token's path; but it now overlaps the next forward, so the forward slows a little and
  not all of the queueing goes away). p2 async: 853 → ~450–550 ms; 8B unchanged (~205
  tok/s). s3 async: without a router wait, a large share of decode requests miss.
- **19:20–19:45 — patch smokes (3B alone, 30 requests).** (1) First version: token #1
  leaves early (s2 TTFT p50 ~250 ms vs 575 sync at 2 QPS), 0 errors — but in s3 **11 of
  30 decode requests missed the KV** (9 found nothing and re-prefilled on GPU0; 3B TPOT
  16.5 ms instead of 12): the predicted race is real. (2) Router gate on a marker file
  written when the GPU→host copy is done: no total misses, still 10 partial hits —
  LMCache pushes L1 → store server on its own loop, after the copy. (3) Marker written
  only after the connector's `put_tasks` are empty: **30 of 30 full hits**, TPOT 13.0 ms,
  no gate timeouts; TTFT p50 284 ms. Kept. The copy itself: ~60 ms when the GPU is free,
  ~230 ms when it overlaps the next request's forward (it queues behind it on the GPU);
  the store push adds ~38 ms. So the decode leg starts ~100–270 ms after token #1.
- **Content check added:** P3 cells send 8 fixed prompts through the split (greedy, 32
  tokens) and record the texts; async must equal sync (`pfc_s3_v` is the sync reference).
- **Round 3 queued (`queue_round3.sh`):** `s3_v` (sync, reference texts), `s2 s3 p2`
  async, `p3` async with the 4 GiB L1 pools. Predictions for s2/p2 above; added:
  s3 async TTFT ~280–320 ms (574 sync), TPOT ~12 ms, 30/30-style full hits, texts equal;
  p3 async 3B TTFT 873 → ~450–550 ms, 8B unchanged from `p3_l1x4` (~195 tok/s).
- **19:20–20:10 — round 3 outcome (async save).** No stall. 3B at 2 QPS, 300 requests:

  | cell | save | 3B TTFT p50 / p95 | 3B TPOT p50 | 3B throughput | decode requests with the whole KV | 8B tok/s / TTFT p50 |
  |---|---|---|---|---|---|---|
  | s2 | sync | 576 / 1441 ms | n/a | 2.14 req/s | — | — |
  | s2 | async | **234** / 774 ms | n/a | 2.14 req/s | — | — |
  | s3 | sync (`_v`, `_v2`) | 575, 582 / ~1500 ms | 11.8, 12.0 ms | 270 tok/s | 222, 216 of 300 | — |
  | s3 | async (+`_v2`) | **236, 238** / 775 ms | 12.2 ms | 271 tok/s | 300, 300 of 300 | — |
  | p2 | sync | 853 / 2062 ms | n/a | 2.13 req/s | — | 204.7 / 185 ms |
  | p2 | async (`_r2`) | **278** / 1122 ms | n/a | 2.14 req/s | — | 200.5 / 233 ms |
  | p3, L1 4 GiB | sync | 873 / 2082 ms | 11.95 ms | 270 tok/s | not recorded | 194.5 / 296 ms |
  | p3, L1 4 GiB | async | **287** / 1102 ms | 13.7 ms | 271 tok/s | 300 of 300 | 198.2 / 274 ms |

  - **Predictions: better than predicted.** s2 async 234 ms (predicted 300–350), p2 async
    278 (450–550), s3 async 237 (280–320), p3 async 287 (450–550). TTFT is back within
    ~25 ms of prefill-only (210 ms); the copy no longer sits in front of the token or of
    the next request's admission. 8B side unchanged, as predicted (still past its knee).
  - **Not predicted: the SYNC split was already racing.** In sync s3, 78–84 of 300 decode
    requests found only part of their prompt in the store and re-prefilled the rest on
    GPU0 — LMCache's L1 → store push is asynchronous even upstream, and the router sends
    the decode leg at once. The rounds-1/2 sync P3 numbers (s3, p3_l1x4) include that
    recompute. The async cells gate the decode leg on the push and get 300 of 300.
  - **Cost of async:** the copy takes ~230–275 ms instead of 58 (it queues behind the next
    forward on the GPU), backlog ≤ 2, 0 errors, 0 preemptions. The decode leg starts
    ~250 ms after token #1.
  - **KV content check.** 8 fixed prompts, greedy, 32 tokens: texts identical between
    sync, async and a connector-free reference (`pfc_s1_v2`: one engine prefills and
    decodes, no LMCache). The texts are degenerate (random-token prompts), so token
    logprobs were compared too: max |Δlogprob| vs the reference is 0.05–0.20 for sync and
    0.05–0.42 for async, per prompt — same order for both, i.e. the split path's own
    numerics (different engine and GPU), not something the async copy adds. Evidence
    that the stored KV is right, not a bit-exact proof.
  - **Incidents:** `pfc_p2_async` failed at launch — the 3B store could not bind :8301
    ("Address already in use", 3 attempts in ~12 s; nothing was listening and the same
    port worked in the cells before and after). Cause not found; the launcher now retries
    for ~45 s and the rerun (`_r2`) is the reported cell. `pfc_s3_v`: 1 of 300 requests
    failed with a client-side payload error (recorded; `_v2` had none).
    `pfc_p2_async_r2` and both p3 cells are swap-backed (1.1–2.8 GB swapped in).

## E. Write-up (2026-10-17)

- MPS verified live (user asked): a short `p1` smoke showed all three engines as MPS
  clients (`nvidia-smi` type `M+C`) under one `nvidia-cuda-mps-server`; not in the summary.
- Report `reports/report_colocation_prefill.md`; figures `pfc_layout`, `pfc_rate`,
  `pfc_saturation`, `pfc_async` (`scripts/plots/plot_pfc_{layout,results}.py`); README
  row and CLAUDE.md layout entry added. **Open:** the 8B was at its knee in every cell
  (prefill-as-neighbour cost not separated from no-headroom cost); P3 is swap-backed;
  most cells n=1; the one unexplained store-port bind failure.
- **10-17 — Decision (user): P3 is reported with the async-save patch only.** The stock
  P3 runs re-prefilled part of the prompt on GPU0 for 26–28% of decode requests, so they
  do not measure the placement; they stay only as the before/after comparison in report
  §5. Report §6 and `pfc_p3.png` show `pfc_base_l1x4`, `pfc_s3_async_v2`,
  `pfc_p3_async_l1x4`.
