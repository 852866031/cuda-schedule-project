# Decision log — 8B-decode vs 3B-prefill priority on the shared GPU

Run-level decisions for the ninth study (`scripts/inf_pf_prio/`), in order, with the
reason and — for experiments — the prediction written **before** the run and the outcome
after. Study-level predictions are in `PLAN_PF_PRIO.md`. Times are local.

## A. Design (2026-10-17, before any run)

| decision | why | who |
|---|---|---|
| Separate study and report; the reference workload may change | | **user** |
| 8B decode budget 24 GiB | give the 8B headroom (b22 was at its knee) | **user** |
| 3B workload reduced to fit GPU1 beside b24 | | **user** |
| P2 only | P2 is the core placement | **user** |
| Arms: 8B priority vs 3B priority, each by MPS and by gating; plus fair | | **user** |
| MPS method = MPS client priority (`CUDA_MPS_CLIENT_PRIORITY`), not SM caps | | **user** |
| No green contexts | not exposed by torch 2.9.1; cross-process use unverified | **user** |
| 3B: 2048-token fresh prompts, eager, util ~0.19, max-num-seqs 4, batched 2048, max-model-len 2064 | must fit ~6.7 GiB; whole-prompt prefill needs batched ≥ max-model-len | Claude |
| 3B uses the async-save patch in every cell | the eighth study showed the stock synchronous save is an artefact | Claude |
| 8B prefill util back to 0.80 | P3 is not run; 0.74 cost the 8B ~100 ms of TTFT | Claude |
| Controls: 8B alone, 8B alone eager, 3B alone per rate | gate3b forces the 8B decode eager; that cost must be separable | Claude |
| GPU1 tensor-pipe active (DCGM 1004) added to telemetry | SM-active does not track compute load | Claude |
| Engine patches combined in `site/sitecustomize.py` (async save + busy-window signal) | Python imports only one `sitecustomize` | Claude |

## B. Log

- **10-17 18:57 — first sizing launch failed at config** (3B alone, smoke): vLLM refuses
  `max_num_batched_tokens` (2048) < `max_model_len` (8192) without chunked prefill. No
  measurement. Fixed by setting the 3B's max-model-len to prompt + 16. The user asked for
  the plan before further runs; nothing else launched.
- **10-17 — user approved the plan** (2048-token prompts, 1/2/4 req/s, ~2.2 h). Next: sizing smoke, then gate smokes, then the queue.
- **10-17 19:05–19:30 — sizing (3B alone, smokes).** max-model-len 2064 also needs batched
  ≥ 2064 (fixed: batched = max-model-len). util 0.19: no memory left for KV (refused).
  util 0.205: KV 0.42 GiB, **GPU1 7.29 GiB**; util 0.21: KV 0.58 GiB, 7.45 GiB. ~0.7 GiB
  sits outside vLLM's budget (CUDA/MPS context). 3B alone: TTFT p50 69.5 ms, p95 107 ms
  at 2 req/s (2048-token prompt + async save). **It does not fit b24:** the 8B decode at
  b24 uses 25.05 GiB, leaving 6.8 GiB; the smallest 3B with a usable KV grant (~util
  0.195) needs ~6.8–6.9 GiB. Options put to the user: 8B at b23.5 (frees ~0.5 GiB; 3B at
  util 0.20, ~7.0 GiB, KV ~0.26 GiB) or the 3B in fp8 (weights ~3 GiB).
- **10-17 — Decision (user): 8B decode budget 24 → 23.5 GiB; 3B util 0.20** (~7.0 GiB on
  GPU1, KV ~0.26 GiB). fp8 for the 3B was discussed and not chosen. Next: a colocated
  smoke at fair to confirm the fit, then gate smokes (gate8b, gate3b), then the queue.
  The 8B-alone cell at b23.5 is the check that the 8B still has headroom.
- **10-17 19:40 — colocated fair smoke failed at the 3B launch** (`pfp_p2_fair_q2_smoke`, no
  measurement). The 8B came up at b23.5 (decode KV grant 7.27 GiB, GPU1 24.6 GiB). vLLM
  refuses to start an engine unless free memory ≥ util × total: 6.19 GiB free < 0.20 ×
  31.36 = 6.27 GiB. So at b23.5 a bf16 3B is capped at util ≤ 0.197, where its KV grant
  is ~0 (util 0.19 already had none). Also found: the launch line carried
  `--quantization fp8` — `pfp_launch.sh`/`pfp_sweep.py` now contain an `SM_QUANT` /
  `--sm-quant` knob defaulting to fp8 that this log did not record; asked the user.
- **10-17 — Decision (user): 8B decode b23 with a bf16 3B (util 0.205); quick feasibility smokes only, then a plan before any large run.** The fp8 knob is left in place, unused (`--sm-quant ''`).
- **10-17 19:45–19:57 — feasibility at b23 (smokes, 30 requests each; numbers are plumbing
  checks, not results — e.g. the 8B's tok/s is window-limited at 30 requests).**
  8B decode at b23: KV grant 6.77 GiB. With the bf16 3B at util 0.205 (KV 0.42 GiB) GPU1
  holds 31.15 GiB of 31.84 — it fits, ~0.7 GiB spare. No stall, no failure, async save 0
  errors, tensor-active telemetry recorded (GPU1 0.077 at fair, 2 req/s).
  `fair`: 3B TTFT p50 78 ms. `gate8b`: busy page live on the 8B decode, CUPTI gate active
  on the 3B (log lines confirm); 3B TTFT p50 97 ms (p95 906 ms) — but this launch swapped
  1.4 GB in. `gate3b`: busy page on the 3B, gate + eager on the 8B decode; 3B TTFT 77 ms,
  8B TPOT p50 31.8 vs 29.2 at fair. All arms work end to end. Plan sent to the user; no
  large run until approved.
- **10-17 — user: run the 8B-alone checkpoint first (`pfp_base_fair_b23`), nothing else yet.** Prediction: ~250 tok/s, TPOT ~30 ms, decode queue < 0.5 s, TTFT ~80–100 ms; if the queue exceeds ~0.5 s, b23 is at the knee and the plan stops for the user.
- **20:10 — `pfp_base_fair_b23` outcome (8B alone, b23, 300 requests): passes, narrowly.**
  254.3 tok/s (full offered rate), TTFT p50 88 ms, TPOT p50 27.7 ms (ITL p50 14.8 ms),
  decode-engine queue wait **0.45 s** mean (threshold set beforehand: 0.5 s; b22 had
  1.07 s). Per request on the decode engine: KV load 299 ms + decode 2.80 s. GPU1:
  SM-active 0.54, tensor-active 0.062, DRAM-active 0.48. No swap, no stall. Prediction
  right on throughput/TPOT/TTFT; the queue is closer to the limit than predicted.
  Headroom by Little's law: grant ~8.8 requests vs ~6.2 needed — some, not much.
  Reported to the user; nothing else launched.
- **10-17 — user: try the 3B in fp8 and find the largest 8B budget that fits.** Sizing smokes (3B alone): fp8 at util 0.13 and 0.15.
- **20:40 — fp8 sizing (3B alone, smokes).** vLLM picks `CutlassFP8ScaledMMLinearKernel`
  (online fp8 works on the 5090). Weights 3.17 GiB (bf16: 5.75). util 0.13: KV 0.64 GiB,
  GPU1 4.79 GiB; util 0.15: KV 1.27 GiB, 5.44 GiB. TTFT p50 43 ms at 2 req/s (bf16 69.5:
  prefill ~1.6× faster). Arithmetic for the 8B: its decode engine occupies budget + ~1.1
  GiB (b23 → 24.1 GiB); with the fp8 3B at util 0.13 the largest budget that fits is
  ~b25.6 → smoke at b25.5 to confirm.
- **20:55 — fp8 at b25.5 failed: the 3B OOMed at startup** (`pfp_p2_fair_q2_fp8_b25.5_smoke`; 45 MiB free). Online fp8 loads the bf16 weights before converting: GPU1 peaks 6.9–9.2 GiB during the 3B's start (1-s samples) although it settles at 4.8 GiB. Added `--sm-first` (3B starts on the empty GPU1, then the 8B); the 8B's KV grant is recorded to check it is unchanged (9.27 GiB at b25.5). Smoke at b25.5 next.
- **21:05 — fp8 at b25.5 with `--sm-first`: fits.** 3B up first (KV 0.64 GiB), then the 8B
  (decode KV grant **9.27 GiB — identical** to the 8B-first launch at b25.5). GPU1 32.08 of
  32.61 GiB in use (~0.5 GiB spare). Smoke ran end to end, 0 failures, async save 0 errors,
  no stall. But the window was swap-backed (1.6 GB in, 2.4 GB out): the 3B sat idle while
  the 8B launched and warmed its 24 GiB store, and its pages were swapped out (3B RSS
  0.9 GB vs 4.1 GB normally). Smoke latencies are therefore not meaningful. Mitigation
  proposed: with `--sm-first`, run the 3B's warmup after the 8B's, right before measuring.
- **21:15 — Decision (user): run both configurations** — A: bf16 3B + 8B b23; B: fp8 3B +
  8B b25 (3B first). Rates 2/4/8 req/s. Plan and predictions: PLAN addendum; queue:
  `queue_prio.sh`. Driver change: the 3B's warmup now runs after both stacks are up,
  right before measuring (swap mitigation for `--sm-first`). Next on approval: config B's
  8B-alone checkpoint (`pfp_base_fair_fp8b25`), then the queue.
- **10-17 21:20 — user: go; ~8 h overnight; afterwards write the report, read it, and run whatever needs supplementing.** Started `queue_night.sh` (checkpoint B, then `queue_prio.sh`: A then B). If checkpoint B fails its 0.5 s queue bar, config B's part is stopped before it starts and logged.
- **00:58 — checkpoint B outcome (`pfp_base_fair_fp8b25`, 8B alone at b25): passes.** 254.3
  tok/s, TTFT p50 86 ms, TPOT p50 29.4 ms, decode queue **0.13 s** (predicted 0.1–0.25 s).
  Config B proceeds as planned. (Queue start times in the log are the box clock: the
  queue began 00:53.)
- **01:38 — progress, config A.** Done: 8B alone eager (252.9 tok/s, TPOT 31.0 vs 27.7
  with graphs, queue 0.72 s), 3B alone at 2/4/8 (TTFT p50 70/70/80 ms), fair at 2/4/8,
  mps8b at 2. No stall, no failure. Cells with > 0.5 GB swapped in, to rerun at the end:
  `pfp_p2_fair_q4_b23` (1.2 GB), `pfp_p2_mps8b_q2_b23` (1.8 GB — its 8B TTFT p50 194 ms vs
  74 at fair is likely that).
- **02:24 — progress, config A (fair, mps8b, gate8b, mps3b done; gate3b running).** No stall,
  no failed request. Early reading against the predictions (to be checked with reruns):
  **mps8b is stronger than predicted** — at 4 req/s the 8B keeps 248 tok/s, TPOT 42 ms
  (fair: 236, 70 ms); **gate8b does not protect the 8B as predicted** — 4 req/s TPOT 107 ms
  (worse than fair; swap-backed run), 8 req/s 127 ms vs fair 166, and the gated 3B
  saturates at 8 req/s (6.0 of 7.7 req/s served, TTFT p50 30 s). Swap reruns pending:
  `fair_q4` (1.2 GB), `mps8b_q2` (1.8 GB), `gate8b_q4` (2.3 GB), `mps8b_q8` (2.3 GB).
- **03:09 — config A complete (21 cells), config B started.** No stall, no failed request.
  gate3b keeps the 3B at its fair-share TTFT (79/79/107 ms) — but the 3B barely suffers
  under fair sharing anyway (80/81/114 ms vs 70/70/80 alone), so 3B priority has little to
  protect in config A. Eager-without-gate control at 8 req/s: 8B TPOT 193 ms vs fair 166,
  gate3b 225. Config B so far: 8B alone eager **swapped 7 GB in** (`pfp_base_eager8b_fp8b25`,
  TPOT 36.7, queue 1.04 s — not usable as is); fair at 2 req/s: 8B 254.4 tok/s, queue
  0.22 s; fp8 3B alone TTFT 43 ms at every rate. Rerun list now: A `fair_q4`,
  `mps8b_q2`, `gate8b_q4`, `mps8b_q8`, `gate3b_q4`; B `base_eager8b`.
- **03:55 — config B (fair, mps8b, gate8b done): the fp8 3B costs the 8B little** (fair 8 req/s: 246 tok/s, TPOT 44 ms, queue 1.3 s). Rerun list adds B `gate8b_q8` (2.1 GB). `queue_reruns.sh` written (7 cells, `_r2`), to start when `queue_prio.sh` ends; any further cell over 0.5 GB swap-in is appended then.
- **04:29 — main queue done (42 cells + checkpoint), no stall, no failed request.** Config B adds to the rerun list: `gate3b_q2` (1.6 GB), `eager8b_q8` (2.5 GB). Started `queue_reruns.sh` (9 cells).

## C. Supplements (decided 04:30 from the main-queue data, before running them)

The main queue raises two questions the report would otherwise leave as conjecture:

- **S1 — why does gate8b not protect the 8B?** In config A at 4 req/s gate8b gives the 8B
  a *worse* TPOT than fair (107 vs 70 ms; swap-backed, rerun pending), and in both configs
  it lets the 8B down while starving the 3B at 8 req/s. The gate only checks the busy flag
  when the 3B *issues* a kernel; with credits K=8 / maxpend=3 up to ~24 3B kernels (each a
  2048-token GEMM) can already be queued on the GPU when an 8B decode step starts.
  Hypothesis: that backlog, not the policy, is what the 8B waits behind.
  Test: arm `gate8bt` (K=1, maxpend=1: at most one unretired 3B kernel) — A at 4 and 8,
  B at 8 req/s. *Prediction:* 8B TPOT drops toward the mps8b level (A 4 req/s ≤ 55 ms;
  A 8 req/s ≤ 110 ms), and the 3B is starved harder (TTFT up, throughput at 8 req/s lower
  than gate8b's). If the 8B does not improve, the backlog is not the explanation.
- **S2 — is "MPS client priority protects the 8B" robust?** It is the main positive result
  and rests on single runs. Repeats: A `fair_q8` and `mps8b_q4` (`_r2`); with the swap
  reruns of `fair_q4` and `mps8b_q8` each A point at 4 and 8 req/s then has two runs.
  *Prediction:* each repeat within ±10% of the first on 8B throughput and TPOT, and the
  ordering mps8b < fair on 8B TPOT holds at both rates.
- **05:15 — swap reruns (9 cells): swap inflates the 8B's TPOT and queue.** Swap-free
  reruns vs swap-backed originals: A fair 4 req/s TPOT 59 vs 70 ms; A gate8b 4 req/s
  **64 vs 107 ms** (so gate8b ≈ fair, not worse — the round-1 "worse than fair" was swap);
  A gate3b 4 req/s 75 vs 99 ms; B 8B alone eager 28.8 vs 36.7 ms (queue 0.12 vs 1.04 s).
  Where both runs are clean they agree (B gate8b 8 req/s: 43.8 vs 44.3 ms). The two A
  mps8b points swapped again on rerun (2 req/s 2.3 GB, 8 req/s 2.5 GB); their numbers can
  only understate mps8b. Rule for the report: per cell, the run with less swap-in; cells
  still > 0.5 GB are flagged. Queued after the supplements: one more attempt at each of
  those two points (`_r3`).
- **05:49 — S1 outcome (tight gate, K=1 / maxpend=1): prediction wrong — the backlog is
  not the explanation.** A 4 req/s: 8B TPOT **78 ms** (gate8b 64, fair 59, mps8b 42), 3B
  TTFT 246 ms; A 8 req/s: 8B TPOT 199 ms, 203 tok/s (gate8b 127 / fair 164), 3B saturated
  (5.7 of 7.7 req/s, TTFT 46 s); B 8 req/s: 8B 39.8 ms / 251.9 tok/s (gate8b 43.8), 3B
  saturated (6.9 req/s, TTFT 20 s). Bounding the 3B's queued kernels to one does not give
  the 8B its standalone TPOT back and starves the 3B harder. What the gate cannot see is
  left open: it only delays kernel *launches* while the 8B is inside `execute_model`;
  copies (the 3B's KV save, the 8B's own KV loads) and the work between decode steps are
  outside it. The gate's own counters are printed at exit and were lost to the SIGKILL
  teardown.
- **05:49 — S2 outcome (repeats): prediction right.** A fair 8 req/s: TPOT 164.5 vs 166 ms,
  209.3 vs 209.3 tok/s; A mps8b 4 req/s: 42.7 vs 42.1 ms, 247.4 vs 248.0 tok/s. The
  mps8b-over-fair advantage reproduces (4 req/s: 42 vs 59–70 ms).
- **05:50 — S3 decided (before running): separate "lighter 3B" from "more 8B headroom".**
  Config A → B changes two things at once (bf16 → fp8 3B, and b23 → b25), and fair-sharing
  cost to the 8B at 8 req/s falls from −18% (queue 18 s) to −3% (1.3 s). Which one does
  it? Cross cell: the **fp8 3B beside the 8B at b23** (`--sm-first`), fair, at 4 and 8
  req/s (suffix `_fp8b23`), plus its 8B-alone reference is config A's (`pfp_base_fair_b23`,
  same 8B). *Prediction:* mostly the 3B's weight — at 8 req/s the 8B keeps ≥ 235 tok/s
  with queue ≤ 5 s (between A and B, closer to B), because the fp8 prefill is ~1.6× shorter
  and its tensor-active footprint ~half.
- **05:52 — the chained waiters for S2b/S3 never fired** (their `pgrep` patterns matched the launching shell's own command line); killed, S2b then S3 started directly in sequence. ~5 min of idle GPU lost.
- **06:00 — S2b outcome.** A mps8b 8 req/s, swap-free at the third attempt (`_r3`): 8B
  **227.4 tok/s, TPOT 88 ms**, queue 8.3 s (fair 209.3, 166 ms, 18.2 s); 3B TTFT 182 ms.
  The swap-backed runs (106 ms) had understated mps8b, as expected. A mps8b 2 req/s `_r3`
  did not run: the 3B store could not bind :8301 for ~45 s ("Address already in use",
  nothing listening — the same unexplained failure as in the eighth study). Fix after S3:
  the launcher falls back to the next free port. S3 is running.
- **06:12 — S3 outcome (fp8 3B beside the 8B at b23, fair): both factors matter, roughly
  equally — prediction ("mostly the 3B's weight") half right.** 4 req/s: 8B 247.9 tok/s,
  TPOT 41 ms, queue 1.7 s — between A (239.6 / 59 / 4.2) and B (253.7 / 34 / 0.30).
  8 req/s (swap-backed, 2.7 GB; rerun queued): 232.9 / 86 / 7.3 s — between A (209.3 /
  166 / 18.2) and B (246.1 / 44 / 1.3). Lighter 3B closes about half the A→B gap, the
  2 GiB of 8B headroom the other half.
- **06:13 — launcher: a store that cannot bind its port now falls back to the next free
  port** (from the 3rd attempt; logged). Last runs queued: S3 8 req/s rerun (`_r2`) and
  A mps8b 2 req/s (`_r3`).
  (The failed launch's raw record `pfp_p2_mps8b_q2_b23_r3.json` — error only, no measurement — was removed so the third attempt could take its name.)
- **06:24 — last cells, both clean.** Cross cell 8 req/s rerun (`_fp8b23_r2`, no swap):
  8B 233.7 tok/s, TPOT 68 ms, queue 5.6 s (the swap-backed first run: 86 ms, 7.3 s) — still
  between A and B, the "about half each" reading holds. A mps8b 2 req/s third attempt
  (no swap, bound :8301 normally): 8B 252.6 tok/s, TPOT 33 ms (fair 40), 3B TTFT 97 ms.
- **06:25 — after reading the draft report: one more rerun.** The only swap-backed cell left
  that the report leans on is the tight gate at 8 req/s in config A (4.2 GB); its 4 req/s
  point is clean and carries the conclusion, but the table shows both. Rerun `_r2`.
  *Prediction:* 8B TPOT lower than the swap-backed 199 ms but not below gate8b's 127 ms;
  3B still saturated.
- **06:35 — report read end to end; corrections applied** (mps8b 8 req/s figure in §4 text,
  the 3B-priority summary, takeaway 3 now cites §3.1). **One gap to fill: every cell runs
  the 8B at 2 QPS**, so "MPS priority protects the 8B" is shown as latency at one offered
  load, not as capacity. **S5 (decided now, before running):** config A, 3B at 4 req/s,
  8B offered **2.5 QPS** under fair and mps8b, plus the 8B alone at 2.5 and 3 QPS
  (suffix `_b23_big2.5` / `_big3`). *Predictions:* 8B alone keeps up at 2.5 (≈ 318 tok/s,
  queue < 2 s) and saturates at 3 (queue growing; its KV grant holds ~8.8 requests, ~3.1 s
  each → ~2.8 QPS); with the 3B at 4 req/s, fair saturates at 2.5 (throughput < 290 tok/s,
  queue > 15 s) while mps8b stays within ~5% of the 8B alone.
- **06:52 — gate8bt 8 req/s rerun:** swapped again (2.5 GB; 4.2 GB first), 8B TPOT 160 ms
  (first 199), 3B saturated (5.9 req/s, TTFT 31 s). Prediction right (lower than 199, not
  below gate8b's 127). Still †.
- **06:52 — S5 outcome (8B load raised, config A).** **Prediction wrong on the baseline:**
  the 8B alone at b23 does not keep up at 2.5 QPS — 280.8 tok/s of ~318 offered, queue
  6.5 s; at 3 QPS 286.0 tok/s (swap-backed, 4.6 GB) — its capacity is ~285 tok/s
  (≈ 2.2 QPS), lower than the ~2.8 QPS Little's-law estimate (the per-request residence
  grows with the batch). With the 3B at 4 req/s and the 8B offered 2.5 QPS: fair
  **240.7 tok/s** (TPOT 158 ms, queue 17.5 s), mps8b **254.5 tok/s** (120 ms, 12.8 s).
  So MPS priority keeps ~1/3 of the capacity fair sharing gives up (281 → 241 vs → 254),
  for +20 ms of 3B TTFT (81 → 100 ms). Prediction "mps8b within ~5% of alone" wrong (−9%).
- **06:58 — S5 repeats (decided before running):** the §4.1 capacity claim rests on one run
  per arm; repeat A fair and mps8b at 8B 2.5 QPS / 3B 4 req/s (`_big2.5_r2`).
  *Prediction:* each within ±3% of the first on 8B throughput; mps8b − fair ≈ +14 tok/s.
- **07:04 — S5 repeats.** mps8b: 256.0 vs 254.5 tok/s, TPOT 116 vs 120 ms — agrees. fair:
  the repeat swapped **10 GB** in (214.7 tok/s, 339 ms) — invalid by the swap rule; the
  clean first run (240.7) stands. One more fair attempt (`_r3`).
- **07:10 — fair `_big2.5_r3` (clean): 246.2 tok/s, TPOT 144 ms, queue 15.6 s** — within 2.3%
  of the first clean run (240.7). S5 repeat prediction right for both arms (±3%); the gap
  is ~+11 tok/s for mps8b (predicted +14). Report §4.1 shows both runs per arm.
- **07:12 — experiments closed.** Machine idle. Final: report
  `reports/report_colocation_prio.md`, figures `pfp_prio_{A,B}_{8b,3b}.png`,
  `pfp_configs.png`; PLAN outcomes annotated; README / CLAUDE.md entries added. Not
  committed (the user commits on request).
- **10-19 — report renamed the configurations (user):** A = bf16 3B + 8B 23 GiB (unchanged), **B = fp8 3B + 8B 23 GiB** (the cross cell / S3 here), **C = fp8 3B + 8B 25 GiB** ("config B" in this log and the plan). Figures `pfp_prio_B_*` became `pfp_prio_C_*`; data names unchanged (`_b23`, `_fp8b23`, `_fp8b25`).
- **10-19 — report takeaways rewritten after discussion with the user** (no new run). New
  analysis from the existing raw records: the 8B's residence on its decode engine (KV load
  + decode) and offered rate × residence ÷ KV slots (6400 tokens × 0.125 MiB per request;
  ~8.7 slots at 23 GiB, ~11.2 at 25 GiB). Every non-gated cell falls on one curve of the
  8B's queue wait against that ratio (`figures/pfp_slots.png`). Corrections made: the
  8B's *median* ITL is unchanged but its mean per-token time grows (2.8 → 4.1 s per
  128-token decode at A, 8 req/s); fp8 cuts the residence growth by about a third
  (+54% → +34%), and the 3× shorter queue is the cliff amplifying it; tighter gating helps
  the 8B in C only by throttling the 3B. Renamed the supplement arm "tighter gating".
- **10-19 — report restructured (user):** the all-arms slot figure moved from §3 to a new §6 after the priority results; §3.1 keeps the slot-utilization explanation with fair-sharing cells only; takeaways are §7. x-axis renamed "KV slot utilization".
- **10-19 — report §3.1: where each input of the slot utilization comes from** (user). Table of sources: offered rate (workload), residence = vLLM `request_prefill_time` + `request_decode_time` on the 8B decode engine (Δsum/Δcount over the window; definitions checked in vllm/v1/metrics/stats.py: prefill = first token − scheduled, decode = last − first token, queue = scheduled − queued), KV per request (arithmetic), KV grant (engine startup log), queue wait (`request_queue_time`). Simplifications stated.
