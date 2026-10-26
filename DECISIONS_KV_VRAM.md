# Decision log — KV slots on the decode GPU

Each entry: what was decided and why, the prediction written **before** the run, and
the outcome added after. Plan: [PLAN_KV_VRAM.md](PLAN_KV_VRAM.md).

## D0 (2026-10-25) — branch setup

- Stripped the branch to the 8B split stack (`scripts/inf_pf_prio/`), `scripts/common/`
  and the ninth study's results; 1391 tracked files removed with `git rm` (other
  branches keep them). Kept `output/calibration_pcie.json` as a prior reference.
- Verified from the archived decode logs: 23 GiB → 55,488 KV tokens (8 whole
  6400-token requests), 25 GiB → 71,872 (11).
- No MPS daemon in this study: GPU1 has one tenant.

## D1 (2026-10-25) — one launch per cell, not one launch per budget

The plan proposed running every QPS cell of a budget against one launch. Reading the
stack before writing the driver found two carry-overs that a fresh launch does not have:
(1) the prefill engine (APC on, ~9 GiB grant on GPU0) keeps the previous cell's *full
prompts*, suffixes included, so a repeated request is a full prefill hit and reaches the
decode engine earlier than in a fresh launch; (2) both engines' 6 GiB LMCache L1 pools
keep the previous cell's hottest chunks, so the decode-side KV load would come from L1
more often than after a fresh warm-up. Both shift decode arrival timing or load time,
i.e. exactly the quantities measured. Decision: one launch per cell, as in the ninth
study (~4 min extra per cell, ~1.5 h over the grid). The plan's drift check is dropped.

## D2 (2026-10-25) — what the client's TPOT contains

With `--forward-first-token`, gap #1 (token 1 → 2) holds the decode engine's queue wait
and KV load. Client TPOT (mean of all gaps) therefore rises with queueing. The client
now also reports `tpot2` (gaps 2..N only, the decode-only per-token time) and `gap1`.
Reports quote tpot2 for "per-token decode speed" and TPOT only as the end-user number.

## E0.2 — PCIe calibration (`scripts/kv_vram/calib_pcie.py`, torch 2.8 dev, .venv)

- **Prediction:** pinned H2D 13–14.5 GB/s at ≥ 64 MiB; pageable ≤ 8 GB/s.
- **Outcome:** pinned H2D **14.47 GB/s** on both GPUs from 16 MiB up (14.2 at 1 MiB),
  D2H 14.3–14.5; link Gen4 ×8 under load (GPU1 idles at Gen1). Pageable reaches
  **14.2–14.3 GB/s** at ≥ 64 MiB (9.2–9.6 at 1 MiB) — the pageable prediction was wrong:
  on this box the staging copy is not the bottleneck for large transfers. One request's
  KV (800 MiB) takes 58 ms at link speed; the measured ~400 ms load is ~7× slower than
  PCIe. `output/kv_vram/pcie_gpu{0,1}.json`.

## E3(b) — streaming micro-benchmark (`scripts/kv_vram/bench_stream.py`, GPU1, .venv)

- **Prediction** (plan, restated with the measured 14.47 GB/s): resident step 12–25 ms
  over B = 1–16; copy floor = B × 0.78 GiB ÷ 14.47 GB/s ≈ **58 ms × B**; streamed
  step ≈ max(copy, compute) for d ≥ 1 and ≈ copy + compute for d = 0; d > 1 gains
  nothing over d = 1.
- **Outcome:** confirmed. Resident 11.1–18.0 ms (B = 1 → 16). Copy floor 58.05 ms × B
  (927.7 ms at B = 16, i.e. exactly 14.47 GB/s). Streamed with d ≥ 1 = copy floor
  + 0.5 ms at every B, identical for d = 1, 2, 4, 8 (the compute is fully hidden behind
  the copy already at d = 1, and deeper prefetch cannot make PCIe faster); d = 0 = copy
  + compute (70 ms at B = 1, 946 ms at B = 16). Streaming costs 4.9× the resident step at
  B = 1 and 52× at B = 16. Corollary for partial residency: the compute that can hide
  copies is 12–18 ms per step, so per step at most ~12 ms ÷ 58 ms ≈ 20% of ONE request's
  layers (≈ 6 of 32) can stream for free at B = 1, and < 1 layer at B = 16.
  `output/kv_vram/bench_stream.json`.

## E0.1 — grant probe (`kv_sweep.py --probe --budgets 22 23 24 25 26 28`)

- **Prediction** (plan): grant = budget − 16.2 GiB ± 0.1; c = 7 / 8 / 9 / 11 / 12 / 15.
  The 23 and 25 GiB grants must reproduce the ninth study's 55,488 / 71,872 tokens
  exactly (same flags; MPS off is not expected to change the profile run).

## D3 (2026-10-25) — independent code review before the first launch

A read-only review of `scripts/kv_vram/` (against vLLM 0.15.1 / LMCache 0.4.4 source)
confirmed the hook points: the decode engine runs `AsyncScheduler` (inherits the four
patched methods), `step_with_batch_queue` calls them as assumed, all attributes exist,
the EngineCore is *spawned* and inherits PYTHONPATH, metric names are right, baseline
engine flags are byte-identical to `pfp_launch.sh big`. Fixed before running:

1. Trace tail could be lost (buffer flushed only by later events; `pkill -9` 3 s after
   SIGTERM skips atexit) → flush on every `fin`.
2. `fin.free` was read before the blocks were released → read after.
3. A failed trace write would raise into the engine loop → every recording step in
   try/except.
4. Idle polling (1 ms) would write ~2000 records/s during a stall → idle iterations are
   counted in `skip` lines.
5. Traced and untraced cells shared a name (the second would be skipped as "exists") →
   `_tr` suffix.
6. A request whose decode leg failed still counted ok (token #1 comes from prefill) →
   ok requires exactly 128 streamed tokens; short streams are failures (`n_short`).
7. Client timeout restored to 300 s (pfp_sweep's value) for comparable failure counts.
8. Environment difference noted, not changed: the ninth study's decode engine always had
   `PYTHONPATH=site` with inert patches; here the baseline has none and the trace arm adds
   it (E1 validation compares exactly this).

Also added: the decode prefix-hit fraction (local + external hits ÷ prompt tokens,
expected 6144/6272 = 0.980) is recorded and a cell is ok only if ≥ 0.95; stale
`kvv_*.log` files are deleted before each launch; the attention backend is recorded.

Interpretation notes taken from the review (affect E1/E4, not the baseline):
- `load.dt` is host wall time of `start_load_kv`; it can include waiting for the
  previous step's kernels (async scheduling), and undercounts under `use_layerwise`.
- The decode engine's `first` token is the regenerated token #1 that the router drops;
  the client's `gap1` ends at decode token #2.
- `decode_load_ms` (vLLM's prefill time on the decode engine) also contains the
  128-token suffix recompute.
- `fp8` KV will likely move the engines off FLASH_ATTN (its fp8 path is FA3/Hopper);
  the backend is now recorded and the fp8 smoke needs a bf16 control on the same
  backend.
- LMCache 0.4.4 `enable_async_loading` makes the *lookup* asynchronous; the copy stays
  in `start_load_kv`. E4 will test what it actually changes.
- **Outcome:** confirmed exactly. Grant = budget − 16.23 GiB at every budget (FLASH_ATTN
  backend, MPS off; 23/25 GiB reproduce 55,488 / 71,872 tokens):

  | budget GiB | 22 | 23 | 24 | 25 | 26 | 28 |
  |---|---|---|---|---|---|---|
  | KV tokens | 47,296 | 55,488 | 63,680 | 71,872 | 80,064 | 96,432 |
  | blocks | 2,956 | 3,468 | 3,980 | 4,492 | 5,004 | 6,027 |
  | blocks ÷ 400 | 7.39 | 8.67 | 9.95 | 11.23 | 12.51 | 15.07 |
  | c (whole requests) | 7 | 8 | 9 | 11 | 12 | 15 |

  GPU1 read 18 MiB after every stop. Note for E1: at 24 GiB the grant is 20 blocks
  short of 10 requests. Admission needs only the prompt's 392 blocks, so a 10th request
  can be admitted while the running ones are still short of 400 blocks and then force a
  preemption as they grow — watch `pre` events and `vllm:num_preemptions` at 24 GiB.
  `output/kv_vram/probe_base.json`.

## E0.3 + E1 validation — 24 GiB, 2 QPS, trace off (`kv_base_b24_q2`) then on (`…_tr`)

- **Prediction (E0.3, no-MPS baseline):** TPOT p50 within ±3 ms of the ninth study's
  8B-alone 27.7–29.4 ms; throughput ≈ 254 tok/s (= 2 QPS × 128); 0 failed; decode
  queue wait between the b23 (454 ms) and b25 (126 ms) values.
- **Prediction (E1 overhead):** trace on vs off: tpot2 p50 within 1 ms, throughput
  within 1%, queue wait within run-to-run noise (±15 ms p50 TTFT-scale; queue means can
  move more — judged against a rerun if they differ by > 30%).
- **Prediction (stalls, hypothesis 2):** steps that admit a request take ≥ 150 ms vs
  ~15 ms otherwise.

## D4 (2026-10-25) — streamed chunks are not tokens; ok rule and TPOT fixed

The D3 rule "ok = exactly 128 chunks" failed 5 requests of `kv_base_b24_q2` (126/129
chunks) and 19 of `kv_base_b24_q2_tr` (120/121 chunks, all in the first 10 requests of
the run). The trace shows all 300 requests generated 128 tokens on the decode engine, and
the engine's generation-token counter rose by exactly 300 × 128 = 38,400 in both cells.
vLLM's output collector merges pending deltas into one chunk whenever the API server
lags (`v1/engine/output_processor.py:45`), and the router's duplicate-token drop keys
on non-empty text. Consequences and fixes:

- Per request, ok = token #1 arrived and at least one decode chunk followed (catches a
  failed decode leg). Per cell, ok additionally requires decode generation tokens ==
  requests × 128 (`tokens_complete`).
- Client TPOT was the mean gap per *chunk*, i.e. inflated whenever chunks merged; it is
  now the request's time span over 127 gaps (tpot2: gaps 2..N over 126). Output tok/s
  is now ok requests × 128 (it summed chunks). ITL percentiles stay per chunk (a merged
  chunk's gap spans several tokens); the count of mismatched requests is recorded.
- The ninth study used the chunk-based TPOT; differences are < 1 ms where chunks match
  (295/300 requests here), so earlier numbers stand within noise.
- Both E1 raws were rescored from their stored records (`scripts/kv_vram/rescore.py`,
  field `rescored = D4`; the original summary is kept as `summary_strict`).

## E0.3 + E1 — first outcome (one run each)

| cell | tok/s | TPOT p50 | tpot2 p50 / mean | decode queue | KV load | preemptions | swap-in |
|---|---|---|---|---|---|---|---|
| trace off `kv_base_b24_q2` | 254.3 | 30.1 ms | 25.3 / 24.2 ms | 200 ms | 316 ms | 12 | 12 MB |
| trace on `kv_base_b24_q2_tr` | 254.4 | 28.4 ms | 24.5 / 23.2 ms | 153 ms | 300 ms | 9 | 102 MB |

- E0.3 confirmed: TPOT p50 30.1 ms (within ±3 ms of 27.7–29.4), 254 tok/s, 0 failed,
  queue 200 ms between b23's 454 and b25's 126.
- E1: every difference favours the traced run, i.e. no overhead visible, but the cells
  differ by 3 preemptions and 47 ms of queue — that is run-to-run variance from
  admission timing, larger than the plan's 1 ms test can resolve with one run each.
  Decision: one repeat of each (`_r2`) before accepting.
- New: **24 GiB preempts** (9 and 12 per run). Running reached 10 (KV 99–100%) — the
  scheduler admits a 10th request on its 392 prompt blocks while the 9 running ones are
  short of their 400, then preempts when they grow.

## E1 — outcome with repeats (accepted)

| cell | tpot2 p50 | decode queue | preemptions | tok/s | swap-in |
|---|---|---|---|---|---|
| off `kv_base_b24_q2` | 25.30 ms | 200 ms | 12 | 254.3 | 12 MB |
| off `kv_base_b24_q2_r2` | 24.78 ms | 154 ms | 10 | 254.4 | 97 MB |
| on `kv_base_b24_q2_tr` | 24.47 ms | 153 ms | 9 | 254.4 | 102 MB |
| on `kv_base_b24_q2_tr_r2` | 24.48 ms | 157 ms | 11 | 254.4 | 2 MB |

- **Overhead prediction confirmed:** trace on − off = −0.56 ms tpot2 (mean of two each;
  off varies by 0.5 ms run to run), throughput identical, queue wait inside the off
  runs' own spread (154–200 ms). The trace is accepted as behaviour-neutral; E2 runs
  with it on everywhere.
- **Stall prediction (hypothesis 2) confirmed** (`kv_base_b24_q2_tr`, from the trace):
  - Steps that overlap a KV load: p50 173 ms, mean 200 ms; 97% of their wall time is
    the synchronous `start_load_kv` call. Steps without a load: p50 14.1 ms, mean 15.0,
    p99 34 ms; by batch size p50 10.7 ms (1 running) → 16.2 ms (10).
  - Load calls: p50 171 ms, mean 198 ms, p95 300 ms for one 6144-token request
    (0.75 GiB) → ~4.4 GB/s, a third of the 14.47 GB/s link (Python fallback copy).
    272 calls loaded 1 request, 21 loaded 2, 4 loaded 3.
  - The whole batch waits: 51 s of the 150 s of decode wall time (34%) is load stall,
    which delays the other running requests by 269 request-seconds in total, ≈ 0.9 s per
    request. That is ~27% of a request's 3.26 s residence (sched → finished).
  - Per request: decode queue p50 0, mean 153 ms, p95 923 ms; load (sched → first token)
    p50 266 ms; decode (first → last) p50 3.11 s.

## E2 — baseline grid (trace on)

Budgets 24 / 26 / 28 GiB × QPS 1, 1.5, 2, 2.5, 3, 3.5, 4, plus 2.25 / 2.75 at 24 GiB
(23 cells; `kv_base_b24_q2_tr` reused). One launch per cell. Predictions as in the plan
(capacity 24: 2.0–2.5, 26: 2.5–3.0, 28: 3.0–3.5 QPS), refined with E1: at 24 GiB
residence S ≈ 3.3 s, so c ÷ S = 9 ÷ 3.3 ≈ 2.7 req/s is the saturation throughput;
**capacity (queue < 50 ms) at 24 GiB ≈ 2.0 QPS** (2 QPS already reads 153 ms mean, so
the knee is at or just below 2). Preemptions grow with QPS at 24 GiB (fractional slot)
and stay ~0 at 26 / 28 GiB (12.51 / 15.07: the slack is ≥ 0.5 request).

### E2 interim (24 GiB row done) — effective slots, and a prediction for 26 / 28 GiB

| QPS | 1 | 1.5 | 2 | 2.25 | 2.5 | 2.75 | 3 | 3.5 | 4 |
|---|---|---|---|---|---|---|---|---|---|
| decode queue (ms) | 0 | 7 | 153 | 941 | 4,680 | 8,913 | 10,977 | 18,409 | 21,413 |
| replay c = 9 | 0 | 23 | 381 | 2,612 | 9,830 | 14,282 | 16,503 | 25,099 | 28,060 |
| replay c = 10 | 0 | 7 | 153 | 932 | 4,591 | 8,759 | 10,856 | 18,216 | 21,250 |
| preemptions | 0 | 1 | 9 | 14 | 35 | 33 | 34 | 33 | 34 |
| tok/s | 128 | 191 | 254 | 282 | 289 | 292 | 303 | 298 | 305 |

- Capacity (queue < 50 ms) at 24 GiB is **between 1.5 and 2 QPS** — below the plan's
  2.0–2.5 (wrong by about one grid step). Saturation throughput ≈ 2.3 req/s.
- A FCFS replay with **10** servers reproduces the measured mean wait within 2% at
  every rate; 9 servers overpredict 1.4–3×. The replay uses each request's measured
  residence, so it is partly circular; what it shows is that waiting is fully explained
  by "integer slots, FCFS" and that the effective slot count at 24 GiB is 10.
- Why 10: admission needs the prompt's 392 blocks; a running request holds 392 → 400.
  3,980 − 9 × 392 = 452 ≥ 392, so a 10th request is admitted while the nine are young,
  and when all ten grow (10 × 400 = 4,000 > 3,980) one is preempted (and later reloads).
- **Prediction (written before the 26/28 GiB cells finish):** 26 GiB: 5,004 − 12 × 392
  = 300 < 392 → effective c = 12, no preemptions at any rate. 28 GiB: 6,027 − 15 × 392
  = 147 → c = 15, no preemptions. Replay with c (not c + 1) should match there.
- At ≥ 3.5 QPS tpot2 *drops* (25.7 → 18.6 ms) while the queue grows: under a deep queue,
  requests finish and are admitted in cohorts — one load call brings in up to 10
  requests (104 calls vs 191 at 3 QPS) — so each request is interrupted by fewer
  separate load stalls during its decode. Decode time = 128 steps + (number of load
  stalls met while resident) × stall length; the stall count depends on the admission
  pattern.

### E2 — `kv_base_b28_q1.5_tr`: one failed request (router keep-alive race)

Request #23 got token #1, then the router's decode-leg POST hit a broken pipe on send:
aiohttp reused a pooled keep-alive connection to the decode server that the server had
already closed (`lmc_proxy.py:94`, ClientOSError in `protocol.read`). The request never
reached the decode engine (299 × 128 decode tokens). Router-level transient, not an engine
or KV-path failure — the CLAUDE.md "lone broken pipe" case. Decision: keep the router
unchanged (no mid-study setup change), rerun the cell as `_r2` after the grid, report the
clean run; the failed run stays in raw.

### E2 — 26 GiB row: effective-slot prediction confirmed

Replay with c = 12 matches the measured mean wait to < 0.1% at every rate (56.2 vs 56.2
ms at 2 QPS, 1,054 vs 1,054 ms at 2.5, 6.21 vs 6.21 s at 3, 16.25 vs 16.25 s at 4);
max Running = 12; zero preemptions at every rate. c + 1 underpredicts (18 ms at 2 QPS).

### E2 — why more slots buy less than proportional throughput (decode-time split)

Per request, decode time (first → last token) split by the trace into executed steps
that overlap no KV load and steps stalled by a load (any request's):

| cell | mean running | step (no load) | decode | = steps | + load stalls |
|---|---|---|---|---|---|
| 24 GiB, 1 QPS | 3.3 | 12.2 ms | 1.90 s | 1.57 s | 0.34 s (1.8 × 185 ms) |
| 24 GiB, 2 QPS | 7.5 | 15.0 ms | 2.96 s | 1.87 s | 1.09 s (5.5 × 198 ms) |
| 26 GiB, 2.5 QPS | 10.9 | 17.4 ms | 3.88 s | 2.04 s | 1.84 s (8.7 × 211 ms) |
| 26 GiB, 3 QPS | 11.6 | 17.9 ms | 3.93 s | 2.05 s | 1.86 s (8.6 × 217 ms) |
| 28 GiB, 2.5 QPS | 11.8 | 17.6 ms | 3.93 s | 2.11 s | 1.80 s (8.6 × 209 ms) |
| 28 GiB, 3 QPS | 14.3 | 19.4 ms | 4.80 s | 2.22 s | 2.56 s (12.1 × 211 ms) |

Of the 2.9 s growth from the lightest to the heaviest cell, the larger batch makes the
steps themselves slower by 0.65 s (decode is weight-read bound: +0.5 ms per extra
request on a ~11 ms base), and load stalls add 2.2 s: every admission stops the whole
batch for ~210 ms, and a request resident among more slots sits through more
admissions (1.8 → 12.1). More slots therefore raise c but also raise S, mostly through
stalls — the reason 26 → 28 GiB (+25% slots) barely moves saturation throughput.
Methods that remove the stall (async / layerwise / faster copy) attack the 2.2 s;
methods that add slots attack the queue but increase the stall count.

## E2 — outcome (23 cells + 1 rerun, all ok; `output/summary_kv.csv`, traces in `output/kv_vram/trace/`)

Decode queue wait, mean per request (ms); throughput in tok/s in brackets:

| QPS | 1 | 1.5 | 2 | 2.25 | 2.5 | 2.75 | 3 | 3.5 | 4 |
|---|---|---|---|---|---|---|---|---|---|
| 24 GiB (c_eff = 10) | 0 (128) | 8 (191) | 153 (254) | 941 (282) | 4,680 (289) | 8,913 (292) | 10,977 (303) | 18,409 (298) | 21,413 (305) |
| 26 GiB (c = 12) | 0 (128) | 0 (191) | 56 (254) | – | 1,054 (309) | – | 6,210 (330) | 12,224 (335) | 16,252 (340) |
| 28 GiB (c = 15) | 0 (128) | 0 (191) | 10 (254) | – | 235 (316) | – | 4,511 (340) | 9,858 (348) | 14,294 (351) |

- **Capacity (queue < 50 ms, 0 failed):** 24 GiB between 1.5 and 2 req/s; 26 GiB at
  ~2 (56 ms, just over); 28 GiB between 2 and 2.5. The plan predicted 2.0–2.5 /
  2.5–3.0 / 3.0–3.5: **all three about one grid step too high**. Saturation throughput
  ≈ 2.3 / 2.65 / 2.7 req/s — +4 GiB (+50% slots) buys +17% throughput.
- **Shape as predicted:** flat below the knee, roughly linear growth of the queue above
  it; tpot2 is 14.5–14.7 ms at 1 req/s at every budget (VRAM is irrelevant below the
  knee) and **rises** with budget at equal high load (39 ms at 28 GiB / 3 req/s vs
  25.7 at 24 GiB) — bigger batches and more load stalls per residence (see the
  decode-time split above).
- **M/G/c:** the FCFS replay with the effective slot count (10 / 12 / 15) matches the
  measured mean wait to < 2% at 24 GiB and < 0.1% at 26 / 28 GiB at every rate; with
  c ± 1 it is off by 1.5–3×. Integer slots, FCFS, and the measured residence fully
  explain the waiting. Lee–Longton (M/G/c with the measured S mean and CV²) is within
  ±10% only near ρ ≈ 0.9 (26 GiB / 2.5: 1,037 vs 1,054 ms); at ρ < 0.7 it underpredicts
  up to 7× (28 GiB / 2: 1.4 vs 10 ms) because the fixed 300-request window is far from
  steady state at small waits — the prediction "within a factor 2 below ρ = 0.9" holds
  only above ρ ≈ 0.7.
- **Preemptions:** 24 GiB 1–35 per cell (0 at 1 req/s), 26 / 28 GiB none at any rate —
  as predicted from the 392-block admission rule.
- **Hypothesis 1 (no sharing):** not yet evaluated from the traces (needs the
  session-per-running-set count); deferred to the E4 APC comparison where it matters.
- 28 GiB / 1.5 req/s: the first run lost one request to a router keep-alive race
  (recorded, `ok = False`); the `_r2` rerun is clean and is the reported cell.

## Check of a §4.3 reading — cohort admission (2026-10-25; dropped from the report)

*Dropped from the report on 2026-10-25: the comparison concerns the decode-only per-token
time, and the report uses the end-user TPOT (token #1 → last), under which 28 GiB is
simply better. Kept here as a record; the finding that the copy costs ~190 ms per request
regardless of grouping stands.*

**What is being checked.** §4.3 reads from the baseline traces that at 4 req/s the 24 GiB
engine decodes faster per token than the 28 GiB one (tpot2 19 vs 31 ms) because its
requests are admitted in cohorts (3.4 per load call, 1.2 stalls per residence) while at
28 GiB admissions are spread out (2.4 per call, 3.5 stalls per residence). That is an
explanation read off the data, not a controlled result.

**Design.** Same engines, same budget (28 GiB), same rate (4 req/s), same arrivals. Two
runs: *control* = as before; *batched* = the router holds each request's decode leg until
5 are ready (or 2 s), then sends them together, so the decode engine admits them in one
step and copies their KV in one `start_load_kv` call (`ROUTER_BATCH=5`, method
`cohort5`). The engines and the workload are untouched; only when the decode engine sees
a request changes. The hold time is recorded per request (`held_ms`) and is *added* to the
client's gap #1 — it must not be read as queueing.

**Prediction (before running).** Stalls per residence fall from ~3.5 to ~1.5–2 (loads
come in groups of ≤ 5 instead of 2.4), so decode time per request falls from 4.04 s to
≈ 2.35 s (own steps) + ~1.7 × 0.25 s ≈ 2.8 s, and tpot2 from 31 to ≈ 22 ms. Mean load
call duration rises (more requests per call, ~5 × 0.17 s ≈ 0.6–0.9 s). Throughput stays
at the saturation value (≈ 350 tok/s) and the decode queue wait does not rise beyond the
hold. If stalls per residence fall but tpot2 does not, the explanation is wrong.

**Outcome (`kv_cohort5_b28_q4_tr` vs `kv_base_b28_q4_tr`): the prediction failed, and
the explanation was wrong in its mechanism.** The router did batch (144 releases, 22 of
them full groups of 5, p50 hold 169 ms) and the engine did admit in bigger groups (3.4
requests per load call, as at 24 GiB, vs 2.4). Stalls per residence halved (4.0 → 2.1)
but each stall was longer (308 → 508 ms), because a call that loads five requests takes
five times as long — the copy path has no batching advantage (187–194 ms per request in
every cell). Stall time per residence barely moved (1.23 → 1.08 s), decode time 4.04 →
3.94 s, tpot2 30.9 → 32.7 ms (within noise), throughput 351 → 358 tok/s.

Corrected statement: *stall time per residence = (requests loaded while this request is
resident) × ~190 ms*, independent of how the admissions are grouped. 24 GiB is faster per
token at saturation not because its loads are grouped but because its loads happen
*between* cohorts: the pool holds almost exactly ten, eviction re-synchronises the
turnover, so a whole batch finishes and the next whole batch is loaded while almost
nobody is decoding (0.36 s of stall per residence vs 1.23 s). With 15 slots and spread-out
completions, five requests admitted together still land in the middle of ten others'
decode. §4.3 of the report corrected accordingly. The control rerun
(`kv_base_b28_q4_tr_r2`) gave tpot2 30.7 ms, decode 4.06 s, 360 tok/s — the as-is cell
repeats within 1%, so the batched cell's 32.7 ms is not an improvement.


## E4 — feasibility smokes (2026-10-25)

Setting for every smoke: 24 GiB, 2.5 req/s, trace on, one launch; `ok` = zero failed,
prefix hit ≥ 0.95, all 38,400 decode tokens, no watchdog trip. Baseline cell to beat:
`kv_base_b24_q2.5_tr` — queue 4,680 ms, TPOT 64 ms, 289 tok/s, decode-only load stall
≈ 0.9 s per residence.

### Smoke 1 — `LMCacheMPConnector` (vLLM's asynchronous load path): infeasible here

Checked before launching: lmcache 0.4.4 ships `vllm_multi_process_adapter.py` and vLLM
0.15.1 registers `LMCacheMPConnector`, but the LMCache multi-process server
(`lmcache/v1/multiprocess/server.py:66`, `gpu_context.py`) requires the compiled ops
`PageBufferShapeDesc` and `multi_layer_block_kv_transfer`. On this box `lmcache.c_ops`
does not load against torch 2.9.1 (CLAUDE.md: a source build deadlocks at engine init)
and the Python fallback `non_cuda_equivalents` does not implement either op. Not run.
The asynchronous path therefore cannot be tested without a working c_ops build.

### Smoke 2 — LMCache `use_layerwise: true` on both engines (`kv_lw_b24_q2.5_tr`)

- **Prediction:** starts and completes (risk: the decode engine runs FULL_AND_PIECEWISE
  CUDA graphs; newer vLLM forces PIECEWISE for layerwise, 0.15.1 has no such check — a
  hang would trip the stall watchdog). Loading becomes per-layer and overlaps the
  forward pass of the admitting step, but the copy path is unchanged (~190 ms per
  request), so the admit-step stall shrinks by < 30%; slot count unchanged (c_eff = 10);
  queue wait within ±20% of baseline; outputs identical.

### Smoke 3 — LMCache `enable_async_loading: true` on the decode engine (`kv_async_b24_q2.5_tr`)

- **Prediction:** starts. The option makes the store → L1 fetch asynchronous (the
  request waits in `WAITING_FOR_REMOTE_KVS` until its chunks are in the pinned L1); the
  L1 → GPU copy stays inside the admitting step. The stall shrinks by the TCP-fetch share
  of the ~190 ms per request, which this smoke measures (unknown; guess 30–50%). If the
  engine log shows no async lookups at all, the option is inert in 0.4.4's v1 adapter
  (the review found the enabling check commented out) and the result equals baseline.

**Smoke 2 outcome (`kv_lw_b24_q2.5_tr`, one run): better than predicted, but one run.**
Both engines echo `use_layerwise: True`; grant unchanged (c_eff = 10); ok (0 failed, hit
0.98, 38,400 tokens). Queue 1,409 ms (baseline 4,680), TPOT 39 ms (64), 308 tok/s (289),
TTFT unchanged. Where it comes from, per the trace: `start_load_kv` now returns in 7 ms
(the per-layer copies run inside the forward pass), so the admitting step itself shrinks
(p50 45 → 2 ms), the per-request load (admitted → first token) falls 865 → 294 ms
mean (p50 689 → 217), residence 4.10 → 3.71 s. The batch is *not* freed: the excess step
time per residence (steps above 1.5× the 16.2 ms median) is 1.40 s vs 0.88 s baseline —
the copy still blocks the forward it rides in — and decode time is 3.42 vs 3.23 s. The
residence gain is in the load phase and in preemptions: 12 vs 35 at baseline, each
avoided preemption saving a reload. Prediction "stall shrinks < 30%, queue ±20%" failed
on the queue; whether the preemption difference is layerwise's doing or run-to-run luck is
open — reruns of both (`_r2`) queued before any conclusion. The old stall attribution
(step overlaps a `start_load_kv` window) is invalid for layerwise; the excess-step metric
above is used for both.

**Smoke 3 outcome (`kv_async_b24_q2.5_tr`): infeasible — crashes LMCache 0.4.4.**
The option engages (async lookup client active, "still waiting for async lookup after
3 seconds" warnings), then the EngineCore dies on
`lmcache/v1/lookup_client/lmcache_async_lookup_client.py:175`
`KeyError: first_lookup_time[lookup_id]` — the client drops a request's bookkeeping on
its 3-s timeout path and the next lookup for the same request raises. 86 requests failed,
heartbeat tore the stack down. Not a trace artefact (the wrapper only calls through).
Not usable in 0.4.4 without patching LMCache; dropped.

**Framing for the report (2026-10-25).** The E4 load-path methods are one idea at three
depths: *overlap the KV transfer with compute*. Baseline copies everything, then computes;
`use_layerwise` interleaves per-layer copies with the admitting forward; the multi-process
connector moves the copy to another process and stream so the batch never waits. The
report must state the goal that way and then quantify, from the trace, how much overlap
each one actually achieves: admitting-step forward time vs the sum of the 32 per-layer
copies (≈ sum + compute ⇒ no overlap, copies merely relocated; ≈ max(copy, compute) ⇒
real overlap). One layerwise run so far looks like the former: excess step time per
residence did not fall; the gain was a shorter load phase and fewer preemptions.
