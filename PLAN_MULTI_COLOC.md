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
