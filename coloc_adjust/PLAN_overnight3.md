# Overnight plan 3 — C1 (limit when the tenant uses the bandwidth) and C2 (a tenant that uses less of it)

> **Status (2026-11-16): run on the night of Nov 16, 02:06–05:45, on both cards.** Everything below
> this box is the text as written on Nov 15 at 23:27, before any run, and has not been edited.
> Outcomes against these predictions, and every deviation from this plan, are in
> [report/DECISIONS.md](report/DECISIONS.md), D20–D28; the results are in `results_o3/`.

Pre-registration document, in the format of [PLAN_overnight2.md](PLAN_overnight2.md): the plan is
not rewritten after the fact. Background: [report/REPORT.md](report/REPORT.md) sections 2.2.1,
2.2.2, 2.7 and 5; the task list is sections C, D and E of [TODO.md](TODO.md).

**The two ideas, as used in this plan** (Jiaxuan, Nov 15: C2 is about the tenant occupying less
of the bandwidth between GPU memory and the SMs, so that decode slows down less):

- **C1** limits the tenant from outside: how many bytes per second it may move, kept by when and
  how long it runs. Two methods: pause after each training step; stop inside a step in short
  slices. The tenant itself is unchanged.
- **C2** changes the tenant itself so that the same training work puts less traffic on the path
  between GPU memory and the SMs. **Its measure is decode's slowdown at equal tenant work.** Means,
  from least to most invasive: the choice among the shapes it already has; settings (micro-batch
  size, attention implementation); merging operators so that intermediate results are not written
  to GPU memory (`torch.compile`, the output layer and loss computed in small chunks); training
  fewer layers (LoRA on the top layers only, which changes the training).

One consequence shapes the runs: a tenant that moves fewer bytes per token and is left to run
flat out may simply run faster and fill the same bandwidth. C2 therefore shows as a lower decode
slowdown only when the tenant's work is held equal, and every C2 comparison below is made at
equal tenant tokens/s. The same data read the other way gives "more tenant work at the same
decode slowdown".

The reports and the review page are being rewritten to this definition (Nov 15).

## 1. What the night has to answer

| | Question | Why it is open | Runs |
|---|---|---|---|
| Q1 | C1's premise. At equal tenant work, is "a few dozen SMs with pauses" cheaper for decode than "few SMs all the time" or "all SMs with pauses"? | The cost per 1 k tenant tok/s was measured with the tenant running all the time (8 SMs: 2.2–2.7 %; 32 SMs: 1.2–1.7 %; all SMs: 1.9–2.4 %). The paced cases are computed from these, not measured. | R1, R4 |
| Q2 | C1, first method. Does a steady allowance remove the tail of the present feedback loop? | `tpot_slo15` keeps the median, but its p90 at 0.5 req/s is +29 %, as bad as no limit: the cap swings between 500 and 20,000 tok/s. A fixed cap gave p50 +5 %, p90 +6 % in one 60-s probe. | R4 |
| Q3 | C1, second method. Does stopping the tenant inside a step, in slices shorter than one decode step, remove the bursts, and what does it cost the tenant? | Never run. With pauses only after a step, one burst lasts about 0.47 s, about 30 decode steps. | R1, R4 |
| Q4 | C2, the tenant side. How far can the tenant's traffic for the same work be lowered, and by which means? | Counted on the CPU only: 17.0 MB per token, 88 % of it intermediate results; measured on the GPU for one shape (13.4–14.3 MB). | R2 |
| Q5 | C2, the decode side. At equal tenant work, does decode slow down less when the tenant moves fewer bytes? | Cost follows tokens/s within one shape; never tested across variants with different bytes per token. The slow shapes cost decode as much while they run as the fast ones, at a fraction of the tokens. An arithmetic job that moves no bytes still costs 1.4–2.6 %. | R3, R4, R5, R6 |

**How this follows the agreed order of work** ([TODO.md](TODO.md), section E):

| Agreed item | In this plan |
|---|---|
| E1.1, same traffic with two knobs | R1, widened: all SMs with pauses added (what `tpot_slo` does), and the second method |
| C1, first method | R4: `c1_fix`, `c1_slow15`, `c1_slow5` |
| A steadier cap | R4: the fixed allowance and the slow loop against `tpot_slo15` |
| C2 | Redefined on Nov 15 as "a tenant that uses less of the GPU-memory bandwidth, so decode slows less": R2, R3, R4 (`c2a`, `c2b`), R5, R6. This takes in section D of the TODO (D1–D5), which was the secondary line |
| C1, second method | R1 (g, h, i) and R4 (`c1_fine`); no longer "only if jitter matters" |
| Prefill tenant with prebuilt green contexts | R7; its bytes per token (D6) in R8 |
| Results in the proposal's metric | A reporting item for every real-engine run (section 4), no extra GPU time |

Of the four things the project's scheduler decides (KV memory budget, neighbours, admission,
transfers), this night is about **neighbours**: what a neighbour costs decode, how to limit that cost from outside (C1), and how to change the neighbour so that it costs less (C2). The KV memory budget runs in every
real-engine run as built in phase 2. Admission and transfers are not touched.

Carried over from Nov 10: the prefill tenant with prebuilt green contexts (R7).

Not in this night: hand-written merged kernels (a merged LoRA branch as in LoRAFusion, a merged
output layer and loss). They need their own correctness checks; R2 and R3 say first whether they
are worth writing. Whether LoRA on the top layers trains as well as on all layers is not answered
either: only the loss at equal tokens is logged.

## 2. Where it runs, and conditions

- **One card, serial, about 5 h 10 min.** The real-engine launchers (`run_eval6.sh`,
  `run_eval_pf.sh`, `run_probe.sh`) are written for GPU 0, the synthetic one (`run_knee_corun.sh`)
  takes `GPU=`; the runner takes the card as an argument and the launchers are parameterised
  before the night. The other card is not touched.
- The card must be free for the whole night: under 500 MiB used by other processes, no other MPS
  client. The runner checks this before every run and stops if it fails; nothing of the other
  project is touched.
- Power limit 500 W on that card, as in every earlier run (both cards stood at 600 W when this was
  written; changing it needs root). The launchers already refuse to start above 500 W.
- `HF_HOME=/mnt/storage/huggingface`.
- Host RAM: the engine's host tier is 28 GiB in `/dev/shm` as in round 2. With 38 GiB available at
  planning time (21 GiB used by other work) this is tight; if less than 34 GiB is available at
  start, the tier is set to 20 GiB. The baseline is run in the same night in either case, at the
  start and at the end of R4, so every comparison stays within the night.

**Questions for Jiaxuan before the start:** which card, and from when; whether the other
project's host RAM use will stay; whether the MPS daemon may be restarted on that card; who sets
the power limit back to 500 W.

## 3. Preparation before the night (no GPU; each item has a CPU check)

| | File | Change | Check without GPU |
|---|---|---|---|
| A | `tenant_bytes.py` (new) | One tenant variant alone on M SMs for N seconds: tokens/s, GB/s from the calibrated counter, MB per token, kernel launches per second (profiler on one step), power, loss | `--device cpu` runs the same loop and prints the table without the counter |
| B | `ft_worker.py`, C1 | A `gbps` field in the budget file next to `tok_s_cap`, converted with a bytes-per-token table per shape and variant (written by R2). **First method:** the sleep after a step, as today. **Second method:** `--pace block --slice-blocks N`: a byte bucket in the per-block hook that exists already (forward and backward pre-hook of each of the 24 layers, 192 stop points per step; today it carries the idle gate): after N blocks, wait for the queued kernels, then sleep until the bucket refills | CPU run against a fake budget file: the conversion, the sleeps, the realised rate and slice lengths printed |
| C | `ft_worker.py`, C2 | Variants: `--attn`, `--compile`, `--loss-chunk N` (output layer and loss over N tokens at a time, gradients collected at the last hidden state), `--lora-top-layers N`; `--no-slow-shapes` (the three shapes with offload count as "does not fit") | For `--loss-chunk`: loss and LoRA gradients equal to the unchunked computation on the CPU (exact comparison). For the others: one CPU step; the shape chosen along a budget ramp |
| D | `knee_corun.py` | A tenant variant, a cap and a pacing method per series; **every decode step timed on its own** (today groups of up to 20 steps are timed together, which hides a burst); the cap for a paced series taken from the unpaced series of the same run | Dry run with the decode step replaced by a sleep; R0 compares per-step with grouped timing |
| E | `controller.py` | `--gbps-fixed X`; `--slo-mode slow`: error from a 10-s median of TPOT instead of the last 2 s, at most ±10 % per 2-s period (today ×0.5 to ×1.5), floor 10 and ceiling 150 GB/s, ceiling when no request runs | Replay of the Nov 9 `tpot_slo15` TPOT log through the loop: the allowance it would have published, and its range |
| F | `run_probe.sh`, `analyze_overlap.py` | Shape probe on a 32-SM context with a GB/s allowance, 120 s per shape, two passes in opposite order; token-gap percentiles (p50, p90, p99) per tenant state in the analysis | Analysis re-run on the Nov 9 logs gives the Nov 9 tables plus the new columns |
| G | `hp_patch/sitecustomize.py` | Prebuilt green contexts on the prefill engine, switched from the budget file's `sms` field (scoped on Nov 10, about 30 lines) | Import test |
| H | `run_overnight3.sh` (new), launchers | The order of section 5, the checks of section 2, the card as an argument, results in `results_o3/`, one log line per run with start, end and exit state | `--dry-run` prints every command |

The loop parameters of E and the slice lengths of B are fixed here and are not tuned during the
night.

## 4. How a result is read

- **Synthetic runs (R0, R1, R3, R6):** the decode step of `knee_probe.py` (Llama-3-8B shapes,
  1 / 4 / 8 running requests) and the real fine-tune tenant as a second MPS client. Decode slowdown
  is against decode alone **on the same SMs in the same run**. Each set is run twice in opposite
  order; a point that differs by more than 5 % between the passes is run a third time and all
  values are reported.
- **Real-engine runs (R4, R5, R7):** R4 and R7 use the 11-minute reference schedule
  (`0:1, 90:2, 180:3, 270:1, 360:4, 420:0.5, 480:2, 510:3, 540:2, 570:1, 600:1` req/s). Reported per
  rate: per-request TPOT median and p90 against the baseline of the same night; token-gap p50,
  p90 and p99; tenant tokens/s and GB/s; the split by tenant running or not. Above 2 req/s GPU
  memory switches the tenant off in every scheme, so 0.5, 1 and 2 req/s are the rows that matter.
- **In the proposal's metric too.** Every real-engine run also reports the load served within
  the SLO: per rate, the share of requests that meet both targets, and the requests per second
  that do; next to it the tenant's tokens/s. Targets proposed here, to be changed by Jiaxuan
  before the night if wanted: TPOT at most 20 ms and TTFT at most 500 ms (the baseline's p90 TPOT
  is 13.3 / 14.1 / 18.9 ms at 0.5 / 1 / 2 req/s). This costs no GPU time; the same numbers are
  computed for the Nov 9 runs from their logs.
- **Two kinds of C2 variant are kept apart in every table.** *Same training*: the result of a
  step is unchanged (shape, micro-batches, attention implementation, `torch.compile`, chunked
  output layer); tokens/s compare directly. *Less training*: LoRA on fewer layers; tokens/s are
  tokens of a smaller job, and the loss at equal tokens is shown next to them.

## 5. Runs, in order

### R0 Sanity and calibration (10 min, synthetic)

Decode alone on 138, 162 and 170 SMs at 1 / 4 / 8 requests, timed in groups (as on Nov 9) and
per step (new); tenant alone on 8, 16, 32 SMs, with and without the block hook; counter
calibration.

- **Prediction:** within 3 % of Nov 9 (11.7–11.8 / 13.1–13.4 / 15.4–15.8 ms; 3.5 k / 6.5 k /
  11.6 k tok/s; 1 % of the counter = 17.8 GB/s). Per-step timing within 2 % of grouped timing.
  The hook alone (no pacing): no basis; this is the first number for what the second method costs.
- **If off by more than 5 %: the night stops here.** The numbers would not be comparable.

### R1 C1 on the synthetic decode step: same tenant work, limited in different ways (35 min) — Q1, Q3

Nine tenant settings next to decode, each at 1 / 4 / 8 requests, 20-s windows, two passes:

| | Tenant SMs | Pacing | Known from Nov 9 |
|---|---|---|---|
| a | 8 (decode on the other 162) | none | 2.4–2.9 k tok/s, decode +5.4 / +7.5 / +6.4 % |
| b | 16 (decode on 154) | none | 4.5–5.3 k tok/s, decode +6.4 / +9.1 / +9.9 % |
| c | 32 (decode on 138) | first method, to the rate of **a** | – |
| d | 32 | first method, to the rate of **b** | – |
| e | 32 | none | 8.0–9.4 k tok/s, decode +9.7 / +15.3 / +14.4 % |
| f | all 170, shared | first method, to the rate of **a** | unpaced: +18.8 / +31.6 / +32.9 % |
| g | 32 | second method, slices of 1 block (about 2.4 ms), to the rate of **a** | – |
| h | 32 | second method, slices of 8 blocks (about 20 ms), to the rate of **a** | – |
| i | 32 | second method, slices of 48 blocks (one micro-batch, about 120 ms), to the rate of **a** | – |

First method: after each training step (4,096 tokens, about 0.47 s on 32 SMs next to decode) the
tenant sleeps until its average rate equals the cap. Second method: the same average rate, but the
tenant stops at block boundaries inside the step. Per point: decode step mean, p90, p99, and the
mean split by tenant running or not; tenant tokens/s, and tokens/s while it runs; GB/s.

- **Prediction for the mean** (slowdown while running × share of time running; pauses cost
  nothing):

  | | 1 request | 4 requests | 8 requests |
  |---|---|---|---|
  | a, 8 SMs all the time (measured) | +5.4 % | +7.5 % | +6.4 % |
  | c, g, h, i: 32 SMs paced to the same rate | +2.9 % | +4.6 % | +4.4 % |
  | f, all SMs paced to the same rate | +4.5 % | +6.5 % | +6.2 % |
  | b, 16 SMs all the time (measured) | +6.4 % | +9.1 % | +9.9 % |
  | d, 32 SMs paced to the same rate | +5.4 % | +8.5 % | +8.2 % |

  The expected gain of **c** over **a** is 2–3 points, of **d** over **b** about 1 point; **f** is
  no better than **a**. While the tenant pauses, decode is within 2 % of alone.
- **Prediction for the tail** (single decode steps): in **c** about 30 % of decode steps fall in a
  burst, so p90 sits at the "while running" level, +10–15 %; **i** the same (a 120-ms slice still
  covers 7–10 decode steps); **h** in between; in **g** every decode step of 12–17 ms contains
  several slices, so p90 is within 2 points of the mean. The tenant's speed while running under
  **g**: no basis (192 waits for queued kernels per step).
- **Decides:**
  - C1 wins on cost if **c** is below **a** at all three batch sizes by more than the spread
    between the two passes. If not, the cost argument rests on choosing 16–32 SMs and pacing is
    kept only as the fine and steady knob.
  - The second method is worth its cost if **g** or **h** brings p90 to within 2 points of the mean
    and the tenant loses less than 10 % of its speed while running. The slice length for
    `c1_fine` in R4 is the longest of **g**, **h** that passes; if neither passes, `c1_fine` runs
    with **h** and is reported as failed on the synthetic test.

### R2 C2, tenant side: bytes per token of every shape and variant (25 min, tenant alone) — Q4

Tenant alone, one minute per variant on 32 SMs; the usual shape and the two best variants also on
the full device.

| Kind | Variants | Prediction for MB per token (usual shape = 13.4–14.3 measured, 17.0 by count) |
|---|---|---|
| Same training: shapes | 1, 2, 4 (usual), 8 micro-batches | 1: about 9 % fewer; 8: about 12 % more (by count) |
| Same training: shapes with offload | activation offload; the same with 12 and with 20 layers kept in host RAM | no basis |
| Same training: settings | attention `eager` (the opposite direction, as a check of the method) | about 37 % more (by count) |
| Same training: merged operators | `torch.compile`; output layer and loss in chunks of 16 tokens; both together | no basis. By count the elementwise multiply and add are about 37 % of the bytes and the output layer about a fifth; how much of that a compiler or a small chunk keeps out of GPU memory is what the run measures |
| Less training | LoRA on the top 8 layers; on the top 4 layers | about 36 % fewer; about 44 % fewer (by count) |
| Reference | forward only (no training) | about 60 % fewer (by count); the floor for this model without merged kernels |

Per variant: tokens/s, GB/s while running (how much of the bandwidth it occupies), MB per token
(how much it occupies for a given amount of work), kernel launches per second, loss after the
minute.

- **Decides:** the two variants for R3 and R4 (the best "same training" one and the best "less
  training" one); the bytes-per-token table that C1 needs to turn a GB/s allowance into tokens/s;
  whether hand-written merged kernels are worth writing (yes if `torch.compile` and the chunked
  output layer together save less than half of what the count says is there).

### R3 C2, decode side, on the synthetic decode step (35 min) — Q5

On 32 SMs (decode on 138), 1 / 4 / 8 requests, two passes. V1 = the best "same training" variant
of R2, V2 = the best "less training" variant.

| | Variant | Pacing |
|---|---|---|
| p | usual tenant | none |
| q1, q2 | V1, V2 | none |
| r1, r2 | V1, V2 | paced to the tokens/s of **p** |
| s | usual tenant | paced to the GB/s of **q2** |

- **Prediction** (cost proportional to bytes):
  - **The C2 result, r against p** (equal tenant tokens/s): decode's slowdown falls by the same
    share as the bytes per token. For a variant with 36–44 % fewer bytes that is from 1.2–1.7 %
    to about 0.7–1.1 % per 1 k tok/s.
  - **q against p** (both flat out): no basis. If the variant just runs faster on the same
    32 SMs, decode's slowdown while it runs does not fall; if its GB/s while running is lower
    (R2), it falls with it.
  - **s against q2** (equal GB/s): decode's slowdown is the same.
  - All points fall on one line of decode slowdown against tenant GB/s, within ±20 %.
- **Decides:** whether C2 lowers decode's slowdown at all. If **r** is not cheaper than **p**,
  bytes are not what decode pays for in this range, C2 is dropped in this form, and R6 says what
  decode does pay for.

### R4 C1 and C2 on the real engine, reference schedule (10 runs × 12.5 min ≈ 2 h 05 min) — Q1, Q2, Q3, Q5

The tenant sits on a 32-SM green context, the engine on the whole device (as in `green_dyn2`).

| # | Run | Tenant | Allowance and rule | Compared with |
|---|---|---|---|---|
| 1 | `adjust_notenant` | none | – | the baseline of the night |
| 2 | `c1_fix` | 32 SMs, usual tenant | fixed 50 GB/s (about 3.7 k tok/s), first method | `tpot_slo15`, which delivered 3.8 k / 3.6 k tok/s at 0.5 / 1 req/s |
| 3 | `c1_fine` | 32 SMs, usual tenant | the same allowance, **second method** (slice length from R1) | `c1_fix` |
| 4 | `c2a` | 32 SMs, **V1** (same training, fewer bytes) | paced to the **tokens/s** that `c1_fix` delivered | `c1_fix`: same tenant work |
| 5 | `c2b` | 32 SMs, **V2** (less training, fewest bytes) | paced to the **tokens/s** that `c1_fix` delivered | `c1_fix`: same tenant work |
| 6 | `c1_slow15` | 32 SMs, usual tenant | slow loop, target +15 % | `tpot_slo15` (same target, fast loop, all SMs) |
| 7 | `c1_slow5` | 32 SMs, usual tenant | slow loop, target +5 % | `tpot_slo5` (Nov 9) |
| 8 | `tpot_slo15` | all SMs | fast loop, +15 % | itself on Nov 9 |
| 9 | `green_dyn2` | 16–96 SMs by load | no allowance | itself on Nov 9; the best SM-only scheme so far |
| 10 | `adjust_notenant` | none | – | run 1: drift over the block |

- **Predictions:**
  - Baselines: TPOT median within 2 % of Nov 9 (12.0 / 12.6 / 16.3 ms at 0.5 / 1 / 2 req/s) and
    within 2 % of each other.
  - `c1_fix` (Q1, Q2): tenant 3.5–3.7 k tok/s at 0.5 and 1 req/s; decode on average +5.3–6.3 %
    (that rate × 1.5–1.7 % per 1 k, the two measured schemes with fewer SMs) against
    +10.4 / +9.6 % for `tpot_slo15` at the same tenant work; the per-request p90 change within
    2 points of the median change.
  - `c1_fine` (Q3): per-request TPOT median and p90 within 1 point of `c1_fix`, because a request
    lasts about 1.7 s and averages over the bursts either way. The difference is in single token
    gaps: with `c1_fix` about half the gaps fall in a burst, so the gap p90 sits at the "while
    running" level (+10–12 %); with `c1_fine` it is within 2 points of the mean. Tenant tokens/s
    equal to `c1_fix` as long as the allowance binds.
  - `c2a`, `c2b` (Q5; **C2 on the real engine**): the tenant does the same tokens/s as in
    `c1_fix` and moves fewer bytes for them, so decode's average slowdown falls by the same share
    as the bytes per token: from +5.3–6.3 % to about +3.0–4.0 % for 36–44 % fewer bytes. p90
    stays within 2 points of the median. Read the other way: at `c1_fix`'s decode slowdown the
    variant could do about 1.55–1.8 times the tokens.
  - `c1_slow15` (Q2): the allowance does not bind at 0.5 and 1 req/s, because 32 SMs next to
    decode give only about 5.7–7.5 k tok/s; decode +9–13 %, inside the target; **tenant work
    1.6–2 times that of `tpot_slo15` at the same decode cost**; p90 at 0.5 req/s under +20 %
    (now +29 %).
  - `c1_slow5` (Q2): the allowance settles near 40–45 GB/s (about 3 k tok/s); decode median
    +4–6 % at both 0.5 and 1 req/s (the fast loop gave +11.2 % and +2.8 % on average); the
    published allowance stays within a factor 2 after the first 20 s of a phase (fast loop:
    factor 40).
  - `tpot_slo15`, `green_dyn2`: within 2 points of Nov 9 on the average and on p90.
- **Decides:** which form of C1 is kept (fixed or slow loop; first or second method), whether C2
  lowers decode's slowdown on the real engine at equal tenant work, and whether C1 + C2 beat `green_dyn2`, the best scheme so far, in
  decode cost per tenant token and in p90.

### R5 C2 with the means the tenant already has: every shape under the same allowance (30 min, real engine, fixed 1 req/s) — Q5

The shape probe of Nov 9 redone inside the C1 frame: engine KV limit fixed at 3 GiB, tenant budget
fixed at 7.5 GiB, tenant on a 32-SM context with the 50 GB/s allowance. Five shapes (4 and 8
micro-batches, activation offload, 12 and 20 layers in host RAM), 120 s each, a no-tenant window
at the start, in the middle and at the end, the whole sequence twice in opposite order.

- **Prediction:** while the tenant runs, decode is +8–12 % with every shape (the 32-SM level).
  The share of time running is the allowance divided by the shape's speed (from R2): about
  50–65 % for the fastest shape, 100 % for the three slow ones, which cannot reach the allowance.
  So decode on average is +5–6 % with the fastest shape and +8–12 % with the slow ones, which
  deliver less than half the tokens.
- **The test of "cost follows bytes" where it is most likely to fail:** whether a slow shape's
  GB/s (R2) predicts its cost. No basis. On Nov 9 the slow shapes cost decode as much while they
  ran as the fast ones, at a fifth to a quarter of the tokens per second; either they move several
  times more bytes per token (R2 shows it), or something other than bytes is being paid for.
- **Decides:** the shape rule inside C2: "among the shapes that fit, the one with the fewest
  bytes per token", if bytes explain the cost; otherwise "never run a shape with offload next to
  decode", with the reason still to find (R6).

### R6 The floor under C2 (15 min, synthetic) — Q5

Known: arithmetic that moves no bytes, on 8 SMs, slows decode by 1.4–2.6 % (one request). An
arithmetic-only second job on 8 and on 32 SMs, at 1 and 8 requests: one long kernel against many
short ones; working set in registers, 1 MB, 64 MB; power, clock frequency and the bandwidth
counter logged.

- **Prediction:** no basis. Three readings to tell apart: (1) launches: one long kernel costs
  under 1 %, many short ones the 1.4–2.6 %; (2) the shared cache: the 64 MB working set costs more
  and the counter shows bytes that are not decode's; (3) the power cap: the clock frequency drops
  when the second job runs, and decode slows by the same share.
- **Decides:** how low C2 can go, and whether merged kernels (fewer launches) have a second
  benefit besides fewer bytes.

### R7 Prefill tenant with prebuilt green contexts (25 min, real engine) — carried over

The prefill scheme of Nov 9 with the 3B tenant engine on prebuilt contexts whose SM count follows
the scheduler (`pf_green_dyn`), and `pf_mps` again as its reference.

- **Prediction** (written Nov 10): decode's cost does not change, because this tenant is limited
  by GPU memory, not by SMs; its own prefill slows down (a prefill chunk runs at 49 % on 64 SMs).

### R8 If time is left (10 min)

The prefill engine's bytes per prompt token, alone: the C2 measurement for a second kind of tenant.

## 6. Timetable

| From start | Run | Idea | Depends on |
|---|---|---|---|
| 0:00 | R0 sanity | – | – |
| 0:10 | R1 pacing on the synthetic step | C1 | R0 passes |
| 0:45 | R2 bytes per token, tenant alone | C2 | – |
| 1:10 | R3 fewer bytes next to the synthetic step | C2 | R2's variants |
| 1:45 | R4 schedule on the real engine, 10 runs | C1, C2, both together | R1's slice length, R2's table and variants |
| 3:50 | R5 shapes under one allowance | C2 | R2's table |
| 4:20 | R6 floor | C2 | – |
| 4:35 | R7 prefill tenant | carried over | – |
| 5:00 | R8 | optional | – |
| 5:10 | end | | |

If the night has to be cut short: R0–R4 (3 h 50 min) run first and complete; inside R4 the order
of the table is the order of importance, except that the closing baseline always runs. R5–R8 can
move to another night.

## 7. Stop rules and what is delivered

- A failed request or a preemption in a real-engine run: that scheme is stopped and marked, the
  night continues with the next run.
- The tenant or the engine does not start within its timeout: one retry, then the run is skipped
  and marked.
- Another process appears on the card: everything stops; the state is reported in the morning.
- No parameter is changed during the night. A loop that sits at its floor or ceiling is a result,
  not a reason to re-tune.
- Morning: `report/DECISIONS.md` D20+ (prediction, outcome, what changes in the plan and why); one
  table per run with the prediction next to the outcome; the figures regenerated; the reports get
  measured results only. Nothing is pushed without a go-ahead. No raw trace above a few MB is
  committed.
