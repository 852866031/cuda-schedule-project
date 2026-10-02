# PLAN — scaling N colocated inference models on one GPU

Pre-registration for the seventh study (`scripts/inf_multi_coloc/`,
`reports/report_colocation_multi.md`). Written 2026-10-01 **before any scaling
measurement**; per repo convention the predictions below are never rewritten — outcomes
are annotated underneath each one.

## Question

The sixth study ([report_colocation_inf.md](reports/report_colocation_inf.md)) put *one*
small tenant beside the 8B decode and found: without MPS two decodes time-slice (~3.4×
TPOT); with MPS a resident tenant costs ~1.2×; KV oversubscription (DRAM streaming) is
the inference tenant's distinct hazard. This study asks how those effects **scale with
the number of co-resident models N**: where does per-model latency knee, where does
aggregate throughput stop growing, and which resource binds first (GPU1 compute/HBM,
GPU1 VRAM, host CPU, host RAM)?

## Design

- **All N models on GPU1**, homogeneous cohorts, each with its own LMCache DRAM store
  and its own client at 2 QPS (open loop, reference session template: 6144-token
  prefixes, 128 forced output tokens). GPU0 hosts only the temporary populate-prefills.
- **Small cohort:** Qwen2.5-0.5B, util 0.08 (≈2.5 GiB: 0.93 GiB weights + 1.04 GiB KV
  grant + activations/graphs) — the exact tenant config of the sixth study.
  fits = 8 sessions (0.56 GiB KV, resident); offload = 48 sessions (3.38 GiB, 3.2× over).
- **Medium cohort:** a ~3–4B Qwen sized empirically (decided after the small cohort).
- **Configs:** decode-only (scenario B: temp GPU0 populate, then retrieve+decode) and
  full (scenario A: 128-token suffix prefill + decode on GPU1); prefill-only with an
  external cross-model scheduler LAST.
- **Arms:** MPS (fair) is the primary; MPS + idle-window gate (model 0 priority, others
  best-effort, eager); an eager-no-gate control; no-MPS as a reference at a few N.
- N = 1, 2, 3, 4, 6, 8, … until VRAM or host RAM binds. N=1 is each cell's baseline.

## Resource arithmetic (before measuring)

| resource | per small model | ceiling |
|---|---|---|
| GPU1 VRAM | 2.5 GiB (util) + ~0.3–0.5 GiB CUDA/MPS context | **~10 models** (31.4 GiB) |
| host RAM, fits | ~4.5 GiB (API server + EngineCore + 1 GiB pinned L1 + store proc) + 0.6 GiB store | **~10–11** before the 6 GiB guard floor |
| host RAM, offload | the above + 3.4 GiB store | **~6–7** |
| pinned host | 1 GiB L1 each (pow-2 exact) | 10 GiB at N=10 — safe |
| host CPU | ~2–3 busy cores per engine (API server + EngineCore busy loop) + client | 32 cores → **pressure from N≈8** |

## Predictions

**P1 — MPS, decode-only, fits (the clean cell).** A 0.5B decode step is dominated by
fixed per-step overhead, not HBM: ~1 GB of weights per step = ~0.6 ms at 1.8 TB/s against
a ~4 ms step. So N co-resident decodes overlap almost perfectly and per-model TPOT stays
within ~1.3× of N=1 up to **N≈4**; aggregate throughput grows ~linearly (≈256·N tok/s,
the offered rate). The knee arrives at **N≈6–8**, set by **host CPU** (N engine loops +
N clients + N stores competing for cores) before GPU1 HBM (N·~1.1 GB per step-round
≈ N·0.6 ms only reaches the step time near N≈7). Signature to expect: rising TPOT with
GPU1 DRAM-active still well below saturation.

**P2 — no MPS.** Time-slicing worsens with N: per-model TPOT grows ~linearly in N
(≈3.4× at N=2 per the last study, ≥6× by N=4) and aggregate throughput stops growing
(capacity collapse) by **N≈3–4**. The MPS-vs-no-MPS gap therefore *widens* with N.

**P3 — full (prefill + decode).** The 128-token suffix prefill adds a short compute
burst per request; at 2 QPS × N it is small. Same shape as P1, TTFT rises before TPOT,
knee ~one step of N earlier than decode-only.

**P4 — offload.** Each model streams its overflow over the host copy path (GIL-bound
Python-fallback LMCache, ~5 GB/s per store process; separate processes so CPU-parallel).
TTFT degrades first and fast: knee at **N≈2–3**, collapse (multi-second TTFT) by
**N≈4**, and the run hits the **host-RAM ceiling at N≈6** before the GPU binds. Swap
will confound the top end (as in the sixth study) — recorded per run.

**P5 — gate.** As N grows the priority model (model 0) stays near its N=1 TPOT
(≤1.2×) while best-effort models absorb the slowdown; under fair MPS everyone degrades
together. Aggregate throughput at saturation is ≈ fair MPS (the gate reorders, it does
not create capacity). Gate value should grow with N.

**P6 — VRAM ceiling.** The small cohort runs out of VRAM at **N≈9–10**; the RAM
ceiling binds first for offload (N≈6), CPU pressure first for fits (N≈8).

**P7 — medium cohort.** A ~3–4B decode reads 6–8 GB of weights per step (~4 ms at
1.8 TB/s) — HBM-bound. Under MPS, N decodes compete for the same bandwidth, so TPOT
grows ~proportionally from N=2 and the knee is **N≈2–3**, bounded by VRAM at N≈3–4.

**P8 — prefill-only + external scheduler (last).** Prefill is compute-bound, so N
concurrent prefills don't overlap usefully — aggregate prefill throughput is flat in N
and TTFT grows ~N×. An external admission scheduler (round-robin / priority across
models) can't add capacity but can bound the priority model's TTFT and trade fairness
for tail latency.

---

## Outcomes

*(annotated as measured)*

- **Medium cohort = Qwen2.5-3B** (decided 2026-10-01, by the user). Qwen3-4B's 7.5 GiB of
  weights alone fill the 6–8 GiB class (N≤3, almost no KV); Qwen2.5-3B (5.8 GiB weights,
  36 KiB/token KV) fits ~7.5 GiB per model → N=1–4. Final sizing (probes): util 0.22
  (util 0.24 fits only 3: 8.24 GiB each incl. context); max_num_seqs 16 + batched 1024
  raise the KV grant 0.59 → 0.94 GiB at the same 7.8 GiB footprint; fits = 3 sessions,
  offload = 14. Medium runs use the CPU-thread cap (OMP_NUM_THREADS=4), which is
  neutral at N=1 (all four cells within 1–3% of uncapped).
- **P1 (small, fits) — partly wrong.** Near-linear to N=6 as predicted, knee at N≈6–8
  (TPOT 1.6 → 3.6 → 5.7 ms; offered rate held to N=8). But the binding resource is
  **GPU1** (SM-active plateaus ~0.8 from N=6 to N=8), not host CPU (15–30% busy).
- **P4 (offload) — knee location right, mechanism wrong.** Decode-only offload is flat to
  N=3 (~52 ms TTFT) and collapses at N=4 (TTFT 39–104 s, ~130 timeouts/model, GPU1 ~5%
  busy; 3 reproductions; a 2 GiB L1 does not help). Cause is not host-copy bandwidth:
  the per-second host trace shows the 4 **EngineCores** pinning all 32 cores (stores
  ~0) — CPU-thread oversubscription (each engine's torch/OpenMP pool is sized to the
  whole box). `OMP_NUM_THREADS=4` removes it entirely (52 ms TTFT, 0 failures, engines
  2.9 cores mean). The L1-pool warnings are present and harmless at N=3 and in the
  capped N=4 run — a red herring.
- **P2 (no MPS), P5 (gate), P8 (prefill-only + external scheduler) — not measured**, by
  the user's decision (2026-10-01): co-located models are equal priority (no gate) and
  every arm runs under MPS with the default equal share. The no-MPS cost is carried over
  from the sixth study (2 decodes: ~3.4× TPOT, −25% capacity), not re-measured vs N.
  The scheduler (`mc_scheduler.py`) and the `pfill` cell exist but were never run.
- **Thread oversubscription is not offload-only.** The stock N=8 *fits* rerun had one
  ~20 s spin storm (8 EngineCores at 31.8/32 cores, GPU1 SM-active 0.95 → 0.03, then
  self-recovered): TTFT p50 unchanged (18 ms) but p95 0.9–2.9 s. The first N=8 run had
  none. A thread-capped N=8 fits run tests whether the cap is a general rule.
