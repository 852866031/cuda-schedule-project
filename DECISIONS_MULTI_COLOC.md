# Decision log — scaling N colocated inference models

Every non-trivial decision taken during the seventh study (`scripts/inf_multi_coloc/`,
`reports/report_colocation_multi.md`), in order, with the reason and — for experiments —
the prediction written **before** the run and the outcome written after. Companion to
`PLAN_MULTI_COLOC.md` (pre-registration) — that file holds the study-level predictions;
this one holds the run-level decisions. Times are local (2026-10-01/02).

## A. Decisions taken during the day (retro-recorded 02:05)

| when | decision | why | who |
|---|---|---|---|
| 10-01 17:30 | All N models on GPU1, one LMCache store each, GPU0 only for temp prefills | isolate the shared-GPU effect; no shared cache state between models | Claude |
| 17:30 | Bring engines up one at a time; assert identical KV grants | concurrent profiling passes corrupt vLLM's KV sizing (sixth study) | Claude |
| 17:30 | One launch serves all four cells (dfits/doff/ffits/foff) | engine flags identical across cells; launch dominates wall-clock | Claude |
| 18:50 | `ok` requires zero client failures; L1-pool warnings counted | the first N=4 offload collapse was mis-scored ok=True | Claude |
| 19:00 | Rerun N=4 doff ×2 (+2 GiB L1) before believing it | user's standard: rerun suspicious points | Claude |
| 19:35 | Per-second host monitor by process role; archive engine logs per launch | engine logs were overwritten per launch → wrong "0 warnings at N≤3" claim | Claude |
| 20:10 | Test `OMP_NUM_THREADS=4` at N=4 doff | trace showed EngineCores (not stores) pinning 32 cores | Claude |
| 20:00 | Medium cohort = Qwen2.5-3B | 6–8 GiB class, allows N=1–4; Qwen3-4B weights alone 7.5 GiB | **user** |
| 20:30 | Medium util 0.22 (not 0.24) | 0.24 fits only 3 models (8.24 GiB each) | Claude |
| 21:00 | Enlarge medium KV grant via max_num_seqs 16 / batched 1024 (0.59 → 0.94 GiB), keep thin runs as `_thin` | user asked whether the grant could be larger; probe showed it is overhead-bound | **user** (option B) |
| 21:15 | Medium sweep with the thread cap; uncapped cells kept as `_nocap` | | **user** |
| 21:40 | No gate / priority arm; MPS default equal share for all models | co-located models are equal priority | **user** |
| 21:40 | Prefill-only + external scheduler not run | | **user** |
| 21:45 | No no-MPS arm (cost carried over from the sixth study) | | **user** |
| 22:00 | Fits and offload as separate figures; panel (c) adds a GPU-memory band | | **user** |
| 23:00 | Decode-only TTFT reported N/A (KV-fetch + queue time, not a TTFT) | | **user** |
| 23:05 | Complete capped offload curves N=1–6 | user asked to see the capped curve to N=6 | **user** |
| 23:50 | Stop offload at N=6 | host already swap-backed (4.8/7.6 GiB swap); N=8 would be refused by the RAM gate | Claude |
| 00:40 | Ring a cell as failed only if >1% of requests failed | a lone client-side broken pipe (1/2,400) is not a collapse | Claude |

## B. Overnight plan (written 02:05, before any of it ran)

The user handed over the night: "check what needs supplementing or verifying, run
experiments, put conclusions in the report, keep every decision trace." Open questions in
the report, ranked by how much a claim depends on them:

| # | open question | report claim at stake | experiment | est. GPU time |
|---|---|---|---|---|
| E1 | Why do medium N=4 offload KV loads get 5× slower (401 vs 85 ms)? cap or contention? | §4 medium offload queueing | medium N=4 doff, **stock threads** | 15 min |
| E2 | Is the collapse really OpenMP spin-wait? (the one inferred link in §3) | §3 mechanism | stock small N=4 doff again with a **per-thread /proc sampler** (no ptrace/perf available: ptrace_scope=1, perf_event_paranoid=4, no sudo) | 15 min |
| E3 | Does the cap prevent the N=8 resident spin storm, or was the clean capped run luck? (stock stormed 1 of 2) | §3.2 rule "cap is general" | small N=8 dfits: stock ×2, capped ×2 | 40 min |
| E4 | What does sharing cost in **capacity**? (all results so far are at 2 QPS/model; the sixth study showed the single point flatters ~3×) | §2/§4 headline | per-model QPS sweep at fixed N, capped, decode-only fits: small N=1,4,8 × {3,4,6,8}; medium N=1,2,4 × {2.5,3,4} | ~2 h |
| E5 | Write §4 medium, §5 ceilings, §6 takeaways; figures; this log | — | — | — |

Order: E1 → E2 → E3 → E4 (cheap, claim-closing first; E4 longest last). Strictly serial,
all through the launch scripts (mem_guard + heartbeat + RAM gate).

### Predictions (before running)

- **E1.** Most likely the uncapped medium N=4 doff **collapses** like the small cohort did
  (same trigger: 4 engines × 32-thread pools, CPU copy work on every request). If it
  instead runs with *faster* loads than capped (< 401 ms), the cap has a real cost under
  offload and §4 must say so. If it runs with the same ~400 ms loads, the slowdown is
  contention, not the cap.
- **E2.** During the stock collapse each EngineCore shows **many busy threads** (≫ 4; on the
  order of 8 each, 32 cores / 4 engines), CPU almost all **user** time, and context
  switches dominated by **involuntary** ones (threads preempted while spinning, not
  blocking). Before the collapse and in a capped run: ≤ ~4 busy threads per engine, low
  CPU. If instead busy threads are few and system time dominates, it is not spin-wait
  (e.g. lock/syscall contention) and §3 must drop the spin-wait reading.
- **E3.** Stock storms in ≥ 1 of 2 further runs; capped in 0 of 2. If capped storms, the
  cap is not a general fix at N=8.
- **E4.** Saturation (achieved < 95% of offered, or failures) per model:
  small N=1 > 8 QPS (no knee in range), N=4 ≈ 6, N=8 ≈ 3;
  medium N=1 ≈ 4–5, N=2 ≈ 3, N=4 ≈ 2.5 (already at 0.92 SM-active at 2 QPS).
  I.e. aggregate capacity grows sub-linearly in N and flattens once GPU1 saturates —
  the 2-QPS results (everyone at offered rate) hide this.

## C. Overnight log

*(appended as it happens)*
- **02:04** — Queue started (`scripts/inf_multi_coloc/queue_night.sh`): E1 → E2 → E3 → E4.
  Added `thread_sampler.py` (per-thread /proc counters for every EngineCore, every 2 s,
  started by the driver on every launch) and a `--qps` list in the driver (capacity sweep
  in one launch; `_q<qps>` names; skip higher QPS once >50% of requests fail).
- **02:14 — E1 outcome: prediction confirmed (collapse).** Medium N=4 decode-only offload
  with **stock** threads: 180 of 1,200 requests timed out, throughput ~290 tok/s (28% of
  offered), host CPU 85%, GPU1 SM-active 0.29, mean KV load **5.6 s** and mean queue wait
  **204 s** — versus capped (OMP=4): 0 failures, 96% throughput, CPU 17%, load 401 ms,
  queue 4.7 s. So (a) the oversubscription collapse is not small-model-specific, and
  (b) the cap is not what makes medium N=4 loads slow — uncapped they are 14× slower.
  Raw: `mc_medium_doff_n4_mps_nocap`.
- **02:16 — Decision: add E1b**, a cap dose-response at medium N=4 doff (OMP=2, OMP=8),
  queued after the main queue (`queue_night2.sh`). Why: E1 shows the cap is necessary but
  not whether 4 threads is too tight — whether some of the 85 → 401 ms load growth is
  thread-starved copies rather than contention. *Prediction:* OMP=2 slower loads than 401
  ms (if copies are thread-bound) or the same (if contention-bound); OMP=8 (32 threads on
  32 cores) either faster loads or a tip back into collapse. I lean "contention-bound,
  OMP=2 ≈ OMP=4; OMP=8 near the cliff".
- **02:26 — E2 outcome: prediction confirmed (spin-wait signature).** Stock small N=4 doff
  collapsed again (4th reproduction; host CPU 97%, GPU1 0.03, onset 14 s into the window).
  Per-thread /proc counters, summed over the 4 EngineCores (540 threads total):
  before onset 0.5 busy threads, 5.3 user cores, 276 involuntary / 10,437 voluntary
  context switches per s; during the collapse **26.8 busy threads, 31.8 user cores, 0.0
  system cores, 11,542 involuntary (42×) / 4,303 voluntary per s**. Threads burning
  user-mode CPU with no syscall time, preempted while runnable rather than blocking — the
  spin-wait signature. Limit: this identifies the behaviour, not the code (no stack
  samples possible without ptrace/perf). Raw: `mc_small_doff_n4_mps_thr`,
  `output/gpumon/mc_small_n4_mps_thr_threads.csv`.
- **02:55 — E3 outcome: prediction confirmed.** Small N=8 decode-only fits, per-run storm =
  any 2-s sample with EngineCores > 20 cores (thread sampler) / tail blow-up. Stock: **2 of
  4** stormed (r2: ~20 s, e2e p95 8.4 s; r4: ~14 s at t≈83–96 s, e2e p95 4.4 s, peak 32.5
  cores); first run and r3 clean (peak 10.9 cores). Capped: **0 of 3** (first, r2, r3;
  peak 8.8 cores; e2e p95 0.93 s, TPOT p95 7.2 ms every time). Median cost identical
  (TPOT p50 5.60 ms). Strength: if the cap did nothing (storm rate ~50%), 0/3 has p≈0.125
  — suggestive, not conclusive alone; with E2's mechanism and N=4 (stock 4/4 collapse,
  capped 0/1) it is consistent. Raw: `mc_small_dfits_n8_mps_{r3,r4,omp4_r2,omp4_r3}`.
- **03:12 — E4 partial (small N=1, N=4): prediction wrong for N=4.** Both keep up (≥95%
  delivered, 0 failures) at every swept rate to **8 QPS/model** — N=4 is 32 QPS total.
  Small N=4 TPOT 2.6 → 3.11 → 3.62 → 4.23 → 4.56 ms (2→8 QPS), worst e2e p95 0.53 →
  0.64 s. I had predicted saturation near 6 QPS. Why it was wrong: decode cost is nearly
  flat in batch size for a 0.5B, so more load = bigger batches, not more steps.
  **Two report corrections follow:** (1) §2's "knee at N≈6–8" is a *latency* knee at 2
  QPS, not a capacity limit; (2) GPU1 SM-active *falls* 0.78 → 0.68 from 4 to 8 QPS while
  throughput doubles — it measures how often SMs are busy, not how full, so "SM-active
  plateau = GPU saturated" must be qualified. Pending N=8; if N=8 also keeps up at 8 QPS
  the sweep did not reach saturation and will be extended upward.
- **03:23 — E4 small N=8: also keeps up at 8 QPS/model** (64 QPS total, ~8.2k tok/s; TPOT
  5.6 → 9.3 ms, worst e2e p95 0.93 → 1.28 s, lowest per-model delivered 98.7 → 96.5%,
  0 failures). The small sweep never saturated.
- **03:23 — Decision: add E4b**, extending the small capacity sweep upward until each N
  saturates (`queue_night3.sh`, after E1b): N=8 at 12/16, N=4 at 12/16/24, N=1 at 16/32
  QPS/model (skip higher QPS once >50% fail). Why: the report should state a measured
  capacity per N, not "≥ 8". *Prediction:* N=8 saturates between 8 and 12; N=4 between
  12 and 24; N=1 above 16.
- **03:37 — E4 medium N=1, N=2: predictions wrong (no saturation by 4 QPS).** TPOT is
  nearly flat in load: N=1 6.78 → 6.91 ms, N=2 9.84 → 10.16 ms from 2 to 4 QPS/model;
  lowest delivered ≥ 97.9%, 0 failures. A 3B step is dominated by reading 5.8 GB of
  weights, which costs the same for batch 2 or 8 — batched requests are nearly free per
  step. Note for the extension: medium engines run max_num_seqs=16 (the setting that
  bought the 0.94 GiB grant), so at ~10+ QPS/model the sequence cap, not the GPU, may
  bind. Extension rates decided after N=4.
- **03:45 — E4 medium N=4: no saturation by 4 QPS** (TPOT 17.3 → 18.6 ms, lowest delivered
  96.5%, 0 failures). E4 main queue finished 03:45.
- **03:45 — Decision: add E4c**, medium sweep extension (`queue_night4.sh`, after E4b):
  N=4 at 6/8, N=2 at 6/8/12, N=1 at 8/12/16 QPS/model. *Prediction:* the max_num_seqs=16
  cap binds first, where QPS × e2e > 16 — N=4 near 6.4, N=2 near 10, N=1 near 16 QPS.
  If instead throughput falls short with fewer than 16 running, the GPU binds first.
