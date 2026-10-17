# PLAN — another model's prefill on the decode GPU

Pre-registration for the eighth study (`scripts/inf_pf_coloc/`). Written 2026-10-13
**before any measurement**; per repo convention the predictions below are never
rewritten — outcomes are annotated underneath.

## Question

In the disaggregated 8B (prefill on GPU0, decode on GPU1) the decode GPU's SMs are almost
empty (SM occupancy < 5%, DRAM-active 45–62%). The fifth study filled them with a
fine-tune, the sixth with a second model's *decode*. This study fills them with a second
model's **prefill** — compute-dense work, the thing decode leaves idle: **can a
Qwen2.5-3B prefill on the 8B's decode GPU raise its compute utilization, and what do both
models pay?**

## Design

- **8B incumbent (identical in every colocated cell):** split-LMCache stack, reference
  workload (32 sessions × 6144 + 128 suffix, 128 out, zipf-1.1, 2 QPS, router with
  `--forward-first-token`, no admission cap). KV offloaded: decode budget **22 GiB**
  (5.77 GiB KV grant vs a 24 GiB working set) — the smallest budget that held 2 QPS in the
  split study, and what leaves GPU1 room for a 3B (23.1 + 7.7 of 31.8 GiB).
  Prefill engine util **0.74** (0.80 in earlier studies) so the 3B's decode of P3 fits on
  GPU0; used in every cell, so there is one baseline.
- **3B tenant, never offloaded, three placements** (prefill always on GPU1):
  - **P1** prefill only, KV dropped (no connector);
  - **P2** prefill, KV written to the 3B's own DRAM store (no decode);
  - **P3** prefill on GPU1 → DRAM store → decode on GPU0 (beside the 8B prefill), own
    router with `--forward-first-token`.
- **3B workload:** a fresh 6144-token prompt per request (16-token stub + unique suffix —
  the multi-model study's prefill-only workload), 2 QPS open-loop, 300 requests;
  `max_tokens` 1 in P1/P2, 128 in P3. The stock store has no evictor, so the 3B's store is
  capped at 4 GiB, oldest-first (a late decode-leg read of an evicted chunk is counted and
  must be zero).
- **Cells:** `base` (8B alone), `s1/s2/s3` (3B alone in P1/P2/P3), `p1/p2/p3` (both).
  Then a 3B QPS sweep on the most informative placement (to be confirmed with the user).
- **Fixed for every cell:** MPS (default share), `OMP_NUM_THREADS=8` (≤ 32 cores ÷ 4
  engines), strictly one launch at a time through `pfc_launch.sh` (mem_guard, heartbeat,
  RAM gate).
- **Recorded:** both models' TTFT and throughput, TPOT where there is a decode (8B always;
  3B in P3 only), engine queue wait and prefill step time, GPU0/GPU1 SM-active,
  SM-occupancy, DRAM-active, memory; host CPU, RAM, swap.

## Resource arithmetic (before measuring)

| resource | 8B alone | + P1 | + P2 | + P3 |
|---|---|---|---|---|
| GPU1 memory | 23.1 GiB | +7.7 → 30.8 of 31.8 | same | same |
| GPU0 memory | ~23 GiB (util 0.74) | — | — | +7.7 → ~30.7 |
| host RAM | ~49 GiB (24 store + 2 × 6 pinned L1 + engines) | +~4 GB | +~5.5 + ≤4 store | +~11 + ≤4 store |
| pinned host | 12 GiB | 12 | 13 | 14 |

Host RAM is the feasibility risk: 62 GB total. P3 projects past physical RAM (the sixth
study's split-tenant scenario C was killed by the guard for the same reason). Each 3B
launch is gated on the RAM actually available after the 8B's store is full; a refused
cell is reported, not forced.

## Predictions

**B0 — baseline.** 8B alone at b22 / prefill 0.74 under MPS reproduces the split study's
b22 row within noise: TTFT p50 ~110 ms (±15), TPOT p50 ~33 ms, ~250 tok/s. The smaller
prefill grant (≈4.8 vs 6.66 GiB) may add ≤ 20 ms of TTFT.

**S — 3B alone.** One 6144-token prefill takes ~0.25–0.35 s on an idle 5090 (≈ 4 × 10¹³
FLOPs). At 2 QPS the prefill engine is 50–70% busy, so TTFT p50 ~0.4–0.6 s including
queueing. P2 adds < 20% (a 0.21 GiB store per request). P3: 3B TPOT ~7–8 ms on an
otherwise empty GPU0, 256 tok/s.

**P1 — the central cell.** GPU1 compute rises a lot: SM-occupancy from ~0.03 to > 0.25,
SM-active from ~0.5 to > 0.9. The 8B pays on the decode leg only: TPOT **+40–80%**
(33 → 46–60 ms); TTFT (a GPU0 quantity) moves < 20 ms. Because b22 sits at the 8B's
in-flight-KV knee, the longer residence time pushes it over: 8B throughput dips below
offered (~230–245 tok/s) and the decode engine's queue wait becomes non-zero. The 3B pays
too: prefill 1.3–1.6× slower than solo, which at 2 QPS is close to saturation — TTFT p50
0.8–2 s, with a visible tail.

**P2 — storing the KV.** Same GPU picture as P1 plus a host-path cost: 0.42 GiB/s of 3B
KV crosses the same host copy path the 8B decode retrieves through. 3B TTFT +10–20% over
P1; 8B TTFT +20–50 ms over P1; 8B TPOT within 10% of P1.

**P3 — decode on the 8B's prefill GPU.** If it fits in host RAM: 3B TTFT ≈ P2 (token #1
is forwarded from the prefill leg); 3B TPOT 8–10 ms; the 8B's TTFT rises 1.3–2× because
its prefill now shares GPU0 with a decode. GPU0 SM-active rises from ~0.05 to > 0.5.
Most likely outcome on this box: swap-backed, or refused by the RAM gate.

**Overall.** Prefill is a worse neighbor for decode than a capped fine-tune (+23% TPOT)
or a resident second decode (+20%): it is bursty, uncapped and compute-dense, closer to
the uncapped fine-tune (+67%). The utilization gain is real but is paid for in the 8B's
TPOT and, at b22, in its capacity.

---

## Outcomes

*(annotated as measured)*

Round 1 (2026-10-13, 3B at 2 QPS; details and the table in `DECISIONS_PF_COLOC.md`):

- **B0 — partly wrong.** TPOT 31.6 ms and 250.8 tok/s reproduce b22; TTFT p50 is 176 ms,
  not ~110 (unattributed: prefill util 0.74 or MPS). The baseline already queues 1.07 s
  on the decode engine — b22 is at the knee.
- **S — s1 right, s2/s3 wrong.** Alone: prefill 210 ms TTFT (step 235 ms). Writing the KV
  to the store doubles the step (462 ms) — not "< 20%" — and the engine queues at 2 QPS
  (TTFT 575 ms). 3B TPOT in s3 is 12 ms, not 7–8.
- **P1 — direction right, sizes wrong.** GPU1 SM-active 0.52 → 0.66, occupancy 0.031 →
  0.064 (predicted > 0.9 / > 0.25). The 8B's per-token gap does not move (ITL p50 14.4 →
  14.5 ms); it saturates instead: requests stay longer on the decode engine (3.1 → 4.3
  s), the 5.77 GiB grant holds ~7, and the queue grows without bound (26 s mean, 202
  tok/s; reproduced in a rerun). The 3B is nearly untouched (231 vs 210 ms), not 0.8–2 s.
- **P2 — 8B side right, 3B side wrong.** 8B same as P1; 3B TTFT 853 ms vs 576 alone.
- **P3 — did not fit host RAM, as predicted.** mem_guard killed it 68 s into the window.
- **Overall — wrong about the mechanism.** The cost to the 8B at b22 is capacity (it was
  already at its in-flight-KV knee), not per-token latency.

Rounds 2–3 (2026-10-13; tables in `DECISIONS_PF_COLOC.md` sections C and D):

- **Rate sweep.** The 8B has no headroom at b22: 0.5 / 1 / 2 req/s of 3B prefill cost it
  1% / 7% / 19% of throughput, all as decode-engine queueing; P1 ≈ P2 for the 8B.
- **Baseline TTFT** is the smaller prefill grant: 77 ms at prefill util 0.80, 176 at 0.74.
- **P2's 3B cost was the connector's synchronous save** (a 55 ms GPU→host copy inside
  the step, in front of token #1, plus the queueing it causes) — the "S" and "P2"
  predictions above did not anticipate it. With an async-save patch the 3B's TTFT in
  P2/P3 returns to within ~25 ms of prefill-only (576 → 234 ms alone, 853 → 278 beside
  the 8B).
- **P3** runs only with the 8B's L1 pools cut to 4 GiB, and is swap-backed: 8B ~195 tok/s,
  3B 271 tok/s at TPOT 12–14 ms, GPU0 SM-active 0.03 → 0.38.

