# PLAN — who gets priority on the shared GPU: the 8B decode or a 3B prefill

Pre-registration for the ninth study (`scripts/inf_pf_prio/`). Written 2026-10-17
**before any measurement**; per repo convention the predictions below are never
rewritten — outcomes are annotated underneath.

## Question

The eighth study ([report_colocation_prefill.md](reports/report_colocation_prefill.md))
put a Qwen2.5-3B prefill (P2: KV written to DRAM) on the 8B split's decode GPU. The 8B
was at its knee (decode budget 22 GiB), so every neighbour cost it capacity, and the
cost of "prefill as a neighbour" could not be separated from "no headroom". This study
gives the 8B headroom and asks: **when the 8B decode and the 3B prefill share GPU1, how
do both perform if the 8B has priority, if the 3B has priority, and if neither does —
and does it matter whether priority is enforced by MPS or by gating?**

## Design

- **8B (unchanged reference workload):** split-LMCache stack, 32 sessions × 6144 + 128
  suffix, 128 out, zipf-1.1, 2 QPS, router `--forward-first-token`, no admission cap.
  Decode budget **24 GiB** (7.77 GiB KV grant; 254 tok/s alone in the split study, not
  at the knee). Prefill util back to 0.80 (P3 is not run, GPU0 needs no room).
- **3B, P2 only, async-save patch:** prefill on GPU1, KV to its own DRAM store (≤ 2 GiB,
  oldest out). The workload is **reduced so the engine fits the ~6.7 GiB GPU1 leaves at
  b24**: a fresh **2048-token** prompt per request (was 6144), 1 output token,
  util ~0.19, `--enforce-eager`, `max-num-seqs 4`, batched 2048, max-model-len 2064 —
  confirmed by a sizing probe before anything else. 3B rate **1 / 2 / 4 req/s**,
  ~150 s windows (requests = 150 × rate).
- **Priority arms (all under MPS):**

  | arm | priority | mechanism |
  |---|---|---|
  | `fair` | none | MPS default share |
  | `mps8b` | 8B | 3B is an MPS client at priority "below normal" (`CUDA_MPS_CLIENT_PRIORITY=1`) |
  | `gate8b` | 8B | 8B decode publishes its busy window; the 3B's kernel launches wait while it is busy (CUPTI gate; 3B eager) |
  | `mps3b` | 3B | 8B decode at MPS priority "below normal" |
  | `gate3b` | 3B | 3B publishes its busy window; the 8B decode's launches wait (8B decode forced eager) |

- **Controls:** 8B alone (CUDA graphs); 8B alone with an eager decode (the gate3b
  reference: separates eager from gating); 3B alone at each rate.
- **Fixed:** `OMP_NUM_THREADS=8`; one launch per cell through `pfp_launch.sh`
  (mem_guard, heartbeat, RAM gate); raw records never overwritten.
- **Recorded:** 8B throughput, decode-engine queue wait, TPOT, ITL, TTFT; 3B TTFT,
  throughput, queue wait; GPU1 SM-active, SM occupancy, **tensor-pipe active** (new,
  DCGM 1004 — the compute-utilization metric), DRAM-active; host CPU, RAM, swap.

Size: 15 colocated cells + 5 controls, ~2.2 h of GPU time.

## Predictions

**B — baselines.** 8B alone ≈ the split study's b24 at prefill 0.80: ~254 tok/s, TPOT
~30 ms, decode queue < 0.3 s, TTFT ~80–90 ms. With an eager decode: TPOT +10–30%,
throughput unchanged (2 QPS is below capacity). 3B alone: TTFT ~70–100 ms (a
2048-token prefill + the async copy off the token path), no queueing to 4 req/s.

**F — fair.** With headroom the 8B absorbs the 3B: at 1–2 req/s its throughput stays
≥ 250 tok/s and TPOT rises ≤ 25%; at 4 req/s TPOT +30–60% and the decode queue starts
to build, but no unbounded growth. The 3B pays little (TTFT ≤ +30%). GPU1 tensor-active
rises roughly linearly with the 3B rate.

**M — MPS client priority (mps8b, mps3b).** Priority only matters when both clients
have kernels ready at the same time; decode kernels are small and short. Effect small:
mps8b within ~10% of fair on the 8B's TPOT; mps3b within ~10% of fair on the 3B's TTFT.

**G8 — gate8b.** The 8B is protected: TPOT within ~10% of alone at every rate. The 3B
only runs in the gaps between decode steps (a few ms per ~15–30 ms step), so its
prefills stretch 2–5× and TTFT rises steeply; at 4 req/s it saturates and queues.
GPU1 tensor-active lower than fair at the same offered rate.

**G3 — gate3b.** The 3B is protected (TTFT ≈ alone). The 8B decode stops for each 3B
prefill (~70 ms): at 4 req/s that is ~30% of wall-clock, so 8B TPOT rises ~1.5× over the
eager baseline and its decode queue grows at the highest rate.

**Overall.** Gating gives a strong, one-sided priority; MPS client priority a weak one.
Fair sharing with a headroom-sized 8B is a reasonable middle at ≤ 2 req/s.

## Addendum — two configurations (2026-10-17, before the full run)

Written after the feasibility smokes and the 8B-alone checkpoint at b23, **before** any
colocated cell of the full run. The original predictions above are kept as written.

| | config A | config B |
|---|---|---|
| 3B | bf16, util 0.205 (KV 0.42 GiB, 7.1 GiB on GPU1) | **fp8** (online), util 0.13 (KV 0.64 GiB, 4.8 GiB) |
| 8B decode budget | **b23** (KV grant 6.77 GiB) | **b25** (KV grant ~8.77 GiB) |
| launch order | 8B first | 3B first (fp8 load peaks at ~7–9 GiB); 3B warmup after the 8B's |
| 3B prefill alone, 2 req/s | TTFT 69.5 ms | TTFT 43 ms |
| 8B alone | measured: 254.3 tok/s, decode queue 0.45 s | checkpoint run first |

Rates change from 1/2/4 to **2/4/8 req/s**: at 2 req/s a 2048-token prefill keeps GPU1
busy only ~16% of the time, too little contention to separate the priority arms.

Per configuration: 8B alone with an eager decode; 3B alone at 2/4/8; P2 under
fair / mps8b / gate8b / mps3b / gate3b at 2/4/8; an eager-without-gate control at 8.
21 cells per configuration, ~2.4 h each, ~4.8 h in all (`scripts/inf_pf_prio/queue_prio.sh`).

**Predictions (addendum).**
- **B0.** Config B's 8B alone at b25: ~254 tok/s, decode queue 0.1–0.25 s.
- **Config A vs B, fair.** At equal 3B rate the fp8 3B costs the 8B less (shorter, lighter
  prefills; 8B with more headroom): at 8 req/s A loses ≥ 10% of 8B throughput with a
  growing queue, B ≤ 5%. GPU1 tensor-active rises with rate in both, less per request in B.
- **mps8b / mps3b.** Within ~10% of fair on the protected side in both configurations.
- **gate8b.** 8B ≈ alone at all rates; 3B TTFT 1.5–4× fair, worst at 8 req/s (A worse
  than B: longer kernels to fit into decode gaps).
- **gate3b.** 3B ≈ alone; 8B TPOT above its eager baseline by roughly the 3B's busy
  fraction (A at 8 req/s: ~60% busy → TPOT ~2×; B ~35% → ~1.5×), queue growth in A at 8.
- **eager control.** The 8B's eager decode alone costs ≤ 20% TPOT; gate3b's excess over it
  is the gate's.

---

## Outcomes

*(annotated as measured)*

- **Design change before any measurement (2026-10-17, user):** the 3B does not fit beside
  the 8B at b24 (smallest usable 3B ~6.9 GiB, 6.8 GiB free). The 8B decode budget is
  **23.5 GiB** (KV grant ~7.27 GiB) and the 3B runs at util 0.20. Predictions above stand
  as written, read against b23.5.

Outcomes of the full run (2026-10-18; details in `DECISIONS_PF_PRIO.md`, report
`reports/report_colocation_prio.md`):

- **B / B0 — right.** 8B alone: A 254.3 tok/s, TPOT 27.7 ms, queue 0.45 s; B 254.3, 29.4 ms,
  0.13 s. Eager decode: +3 ms TPOT in A, nothing in B. 3B alone: 70–80 ms (bf16), 43 ms
  (fp8) — faster than the 70–100 ms predicted.
- **F — wrong for config A, right for B.** A: fair sharing costs the 8B −6% at 4 req/s and
  −18% at 8 req/s with an unbounded queue (predicted "≤ 25% TPOT at 1–2 req/s, no
  unbounded growth"). B: −3% at 8 req/s. The cross cell attributes the gap about equally
  to the 3B's weight and the 8B's headroom (prediction "mostly the 3B" half right).
- **M — wrong: MPS client priority is strong.** mps8b keeps the 8B at 43 ms / 247 tok/s at
  4 req/s (fair 59 / 240) and 88 ms / 227 tok/s at 8 (fair 166 / 209), for +20–70 ms of
  3B TTFT. mps3b recovers 4–10 ms of the 3B's TTFT.
- **G8 — wrong: gating does not protect the 8B.** No better than fair (64 vs 59 ms at
  4 req/s), starves the 3B (saturated at 8 req/s in both configs). A tighter gate is worse
  on both sides — the launch backlog is not the cause.
- **G3 — 3B side right, 8B side roughly right.** 3B ≈ fair-share or better; the 8B pays
  beyond its eager baseline (A 8 req/s: 225 vs 193 ms; B: 58 vs 45 ms).
- **Overall — half right.** Gating is one-sided, but against the protected model here;
  MPS client priority is the effective lever, not a weak one.
