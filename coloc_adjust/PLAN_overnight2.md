# Overnight plan 2 — fix the accounting, close the compute half, try the hard parts

> **Status (2026-11-09 06:15): executed.** Outcomes per section in [report/DECISIONS.md](report/DECISIONS.md) D11–D19; results in [report/REPORT.md](report/REPORT.md) / [REPORT_zh.md](report/REPORT_zh.md). Deviations: a scheduler bug (stale pending grow, D16) forced a re-run of the six adaptive schemes; chain 2 was started by hand (its waiter matched its own command line); `static_split` was re-run without the gate (D15).

Pre-registration document (plans are never rewritten after the fact; outcomes go to
`report/DECISIONS.md` D11+ with a prediction written before each run). Written 2026-11-09 after
the first prototype round (`report/REPORT.md`). Hardware: both GPUs idle at planning time, 53 GiB host RAM free (GPU roles: see P0.0).

Priorities, in the order they run: inference latency first, tenant progress second, everything
else third. Every run reuses the 11-minute reference trace (Figure 0 of the report) unless said
otherwise, so results are comparable with the existing tables.

## P0.0 Amendment at start (2026-11-09 22:00): 500 W cap and a full re-run of the reference set

Both GPUs are now capped at 500 W (`nvidia-smi -pl 500`, persistence on; default was 600 W).
Every number in the existing tables was measured at 600 W, so **the reference schemes are re-run
at 500 W with the fixed accounting of P0.1**, and the figures/tables are rebuilt from the new set:
`adjust_notenant` (baseline), `static_split`, `adjust` (gate), `adjust_nogate` (MPS), `adjust_mps20`,
`green_split`. `decode_alone` is dropped (identical to the baseline). The 600 W results stay in
`results_600w/` and are not mixed with the new ones. The runner (`run_eval4.sh`) refuses to start
if GPU 0's limit is above 500 W; the sampler records power, SM/DRAM utilisation and SM clock.
**Prediction:** decode alone is bandwidth-bound and draws well under 500 W, so the baseline TPOT
changes by < 3 %; the tenant's solo throughput drops 5–10 % (training is compute-heavy and will
hit the cap); the colocated runs hit the cap at 1–2 QPS, which lowers the tenant's throughput and
therefore the decode penalty a little, but the slope of Figure 2 (slowdown per tenant tok/s) stays
within 2.0–2.6 %/k.

GPU roles, corrected: the engine, tenant and MPS daemon run on **GPU 0** (`CUDA_VISIBLE_DEVICES=0`
in the runner); the kvcached + LMCache work of P4 therefore runs on **GPU 1**.

## P0. Must-fix (GPU 0, ~1.5 h)

### P0.1 VRAM-scheduler accounting

Found on 2026-11-06: the budget formula `total − 17.7 − 1 − limit` over-promises by 1–2 GiB at
the top of the range: the engine's non-KV footprint is 18.3 GiB by NVML (not 17.7), the tenant's
CUDA context (~0.8 GiB) is outside `torch.cuda.memory_reserved`, and the paused tenant still
holds 1.2 GiB (0.6 gate scheme, 1.9 green split). Nothing collided because KV in use stayed
≥ 2.1 GiB below the tenant; a collision would surface as kvcached scheduling misses and vLLM
preemptions, not a crash.

Change (`controller.py`):
- `engine_fixed` = NVML used − kvcached used, measured by the scheduler over the first 5 s after
  the engine is up (replaces the constant 17.7).
- `tenant_fixed` = NVML delta when the tenant process has initialised and before it takes any
  budget (context + allocator floor), published once by the tenant in its state file; default 2.0
  GiB if absent.
- budget = total − engine_fixed − tenant_fixed − limit; the 1 GiB margin stays as a separate term.
- `limit_max` = total − engine_fixed − margin − tenant's **current reported footprint**, not its
  budget, so the limit can only be promised against memory the tenant has actually released;
  growth beyond that waits for the tenant (it releases within 50 ms mid-step, ~0.3 s for a pause).

Run: part of the P0.0 re-run set (all six reference schemes use the fixed accounting).
**Prediction:** max limit ≈ 11.0 GiB instead of 12.65; queue wait at 4 QPS rises from ~3.3 s to
4–5 s (less KV for the spike); every other phase within noise of the old run; "tenant holds more
than its budget" episodes shrink from 193 s to < 10 s total and never exceed the release lag.

### P0.2 Figures

- `analyze_coloc.py`: tenant memory drawn with last-known value (step hold), not linear
  interpolation across pauses; regenerate the 11 timelines.
- Transfer figure (`<scheme>_xfer.png`) redrawn in the report style (three rows: copies as spans
  by source, GiB/s by source, NVML PCIe) and embedded in both reports under §3.
- `sampler.py`: add NVML `utilization.memory` (memory-controller busy %) to every sample; add a
  row "DRAM controller busy %" to the timeline. This is the direct test of the report's headline
  ("the contention is DRAM bandwidth"), which so far rests on inference.
  **Prediction:** inference alone at 1 QPS keeps the controller ~45–55 % busy; with the tenant
  running at ~10k tok/s it reads > 85 % and the decode slowdown tracks it; the disjoint-SM scheme
  shows the same reading as the un-capped MPS scheme.

## P1. The compute half: proposals and the one we run (GPU 1, ~2 h)

The finding to design around: decode slowdown ≈ 2.3 % per 1k tenant tok/s, independent of how
the SMs are split. Three ways to act on it:

1. **Throughput feedback (run tonight).** Control what matters directly. The router already
   times every output token; it keeps a 10-s rolling TPOT median. The scheduler compares it with
   the baseline TPOT for the current arrival rate (lookup from `adjust_notenant_phases.csv`,
   interpolated in λ̂) and publishes `tok_s_cap` in the budget file: `cap = cap · (1 − g·e)` with
   `e = (TPOT/TPOT_base − 1) − SLO`, g = 0.5, updated every 2 s, clamped to [0.5k, 20k]. The
   tenant sleeps after each step so its tokens/s does not exceed the cap (the dynamic version of
   the old `--duty`). Memory ladder unchanged; MPS, no SM cap. Schemes: `tpot_slo5`, `tpot_slo15`.
   **Prediction:** at 0.5–2 QPS the decode TPOT settles at baseline + SLO ± 3 points within
   ~10 s of each phase start; tenant ≈ 2k tok/s (SLO 5 %) and ≈ 6.5k (SLO 15 %), i.e. the points
   land on Figure 2's line; 3–4 QPS identical to the baseline (tenant paused by the memory loop).
   SLO 15 % should match the MPS 20 % cap's tenant progress at a slightly lower decode cost,
   because the cap is now exact rather than whatever 20 % of the SMs happens to yield.
2. **Bandwidth-aware shape choice (probe tonight, 20 min).** The ladder saves memory by cutting
   the micro-batch, which lowers arithmetic intensity and raises bytes per token — the opposite of
   what the bandwidth finding wants. Probe: at a fixed 1 QPS, run each tenant shape alone for
   60 s under the throughput cap of P1.1 fixed at 4k tok/s and record decode TPOT. The slope
   (slowdown per tok/s) per shape goes into `tiers.json`; the ladder then prefers, among shapes
   that fit the budget, the lowest slope instead of the fastest.
   **Prediction:** k=2 has the lowest slope, k=8 + activation offload the highest (its PCIe traffic
   also competes), weight streaming in between; the spread is ≥ 1.5× — enough that choosing by
   slope buys ~20–30 % more tenant throughput at equal decode cost.
3. **Decode-step-aligned pacing (not run).** Launch tenant kernels only in the attention-heavy
   half of each decode step, where DRAM is less loaded. Needs kernel-level coordination inside the
   engine's async step loop; out of scope for a router-based design. Recorded as the option we
   are deliberately not taking.

## P2. Dynamic green context, second attempt (GPU 1, ~1.5 h)

Diagnosis to test: PyTorch's caching allocator keeps one pool per stream. The tenant created a
new green context and stream on every share change; the optimizer's AdamW states were allocated
lazily in the first step inside the first green stream's pool, so each later pool had to grow
from zero while earlier pools kept their fragments — reserved memory climbed 5 → 7.7 GiB and the
tenant OOMed to pause.

Fix (`ft_worker.py`, `green_ctx.py`):
- create the 4 contexts (128/64/32/16 SMs) and their streams once at start;
- materialise optimizer state and run one warm step on the default stream before any switch;
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
- switch only between steps, after `synchronize()` + `empty_cache()`; log reserved/allocated
  after every switch.
Fallback if reserved still grows > 0.5 GiB per switch: restart the tenant process from its DRAM
snapshot with the new SM count (measured 2–3 s), i.e. switching becomes a pause/resume.

Scheme `green_dyn2`: SM count from the arrival rate (128/64/32/16 at 0/1/2/≥3 QPS), MPS off.
**Prediction:** the tenant survives the whole run with reserved ≤ its shape's calibrated need
+ 0.3 GiB; decode cost per tenant tok/s is the same 2.3 %/k as every other scheme (the SM count
acts only as a throughput throttle); compared with P1.1 at equal tenant throughput it is no
better for decode and worse in tenant progress at 16 SMs (the throttle is coarse).

## P3. A prefill tenant: Qwen2.5-3B fp8, KV streamed out, no decode (GPU 1, ~3 h)

Why: a training tenant is bandwidth-heavy and showed that SM partitioning does not help. Prefill
is compute-bound (high arithmetic intensity), so it is the case where SM capping or a disjoint
partition should protect decode while the tenant keeps most of its throughput. It is also the
colocation the proposal cares about (a small model's prefill beside the 8B's decode).

Tenant: vLLM 0.30 + kvcached (second instance, `KVCACHED_IPC_NAME=vllm_b`) serving
`Qwen/Qwen2.5-3B` with `--quantization fp8` (3.17 GiB of weights, as in the earlier prio study),
`--kv-offloading-size 8 --kv-offloading-backend native` so every finished prefill's KV is
streamed to host RAM (the native connector is the one kvcached pins pages for), requests with
`max_tokens=1`, prefix caching off. Its workload: Poisson arrivals of 2k–4k-token prompts at a
fixed rate per run (1 req/s ≈ 3k prefill tok/s), its metrics prefill tokens/s and TTFT.
Budget adaptation (a small adapter process reading the budget file): ladder = kvcached limit for
vllm_b (bounds in-flight prompts) → stop admitting (queue at the adapter) → `--enable-sleep-mode`
level 1 (`/sleep`: weights to host, KV discarded) as the pause tier, `/wake_up` when the budget is
back. If sleep mode and kvcached conflict, the pause tier is "limit at minimum + no admission" and
the 3.2 GiB of weights count as tenant_fixed.
Schemes (MPS throughout): `pf_mps` (no cap), `pf_mps20` (tenant at 20 % of SM threads),
`pf_split` (engine on 128 SMs via the observation patch, tenant on the remaining 42 via a green
context — a vLLM worker can be put on a green stream the same way the 8B engine was).
**Predictions:** `pf_mps` costs decode +25–40 % at 1–2 QPS (prefill kernels occupy all SMs);
`pf_mps20` and `pf_split` bring that under +10 % while the tenant keeps ≥ 60 % of its prefill
throughput — the opposite of the training tenant, where the partition changed nothing. The memory
ladder works unchanged (the adapter is ~150 lines). Sleep mode wake-up takes 2–4 s (weights back
over PCIe), which the 3-s up-hold already covers.
Risk register: fp8 on sm_120 under vLLM 0.30 (the prio study did it on 0.15; fallback bf16 at
5.75 GiB), two kvcached instances on one GPU (supported by IPC name; fallback: tenant without
kvcached, fixed `--gpu-memory-utilization`, pause by sleep only), sleep mode vs kvcached (fallback
above).

## P4. kvcached + LMCache MP connector (GPU 1, in parallel, time-boxed 3 h)

Two independent defects were recorded in `kvcached_validation/REPORT.md` §4–5:

(a) **Server-side zero copy cannot register.** kvcached's `cuMemCreate` leaves
`requestedHandleTypes = 0`, so LMCache's `VmmCudaIPCWrapper` cannot export the pages. Patch
kvcached `csrc/inc/gpu_vmm.hpp` to request `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` behind an
env flag, rebuild into `.venv-main` (the upstream-main source has to be re-cloned; the old
scratchpad is gone), re-run B2b with `layout=LBNHC`.
**Prediction:** the exportability check passes and registration then fails at LMCache's
one-handle-per-tensor rule (kvcached backs a tensor with many 2 MiB pages). Making LMCache import
per-page handles is a change on their side; we stop there and record exactly where.

(b) **`engine_driven` faults under store traffic (Xid 31 read of an unmapped kvcached page), with
or without resizing.** Hypothesis: LMCache's asynchronous GPU→host gather reads a finished
request's blocks after vLLM has freed them and kvcached has unmapped or reused their pages;
kvcached only pins pages for vLLM's native `OffloadingConnector` (`offload_pins.py`). Fix: a pin
layer for the LMCache MP connector modelled on `offload_pins.py` — pin the blocks of a request
whose store is in flight at `request_finished` (the connector returns `_can_store=True` there) and
unpin when `get_finished` reports the store done. Test with the stress load that reproduced the
fault in 1.5–2.5 min (6 sessions, 6000-token first turns, +512 per turn), 5 min with resizes and
5 min without; then the 8-prompt greedy-output check.
**Prediction:** no Xid, 0 errors, outputs equal to plain engine + connector (which itself differs
from the no-connector engine on 4/8 prompts — a pre-existing LMCache property). If the fault
persists with pins in place, the read is outside the request's own blocks (LMCache's block list
overrunning), which the pins cannot fix; record and stop.

## P5. Reporting (~1 h, end of night)

- `report/DECISIONS.md` D11–D16: one entry per section above, prediction first, outcome after.
- `report/REPORT.md` and `report/REPORT_zh.md`: accounting fix and its effect; DRAM-busy figure
  beside Figure 2; throughput-feedback schemes added to Figures 1–2 and the tables; green-context
  second attempt; prefill-tenant section with its own three-scheme figure; LMCache status;
  transfer figure embedded.
- Memory notes updated (`coloc-adjust-prototype-findings`).

## Schedule (GPU 1 serial; P4 on GPU 0 from the start)

| when | what | GPU |
|---|---|---|
| 22:00–23:30 | P0.0/P0.1 code, smoke test, DRAM/power sampling; start the six reference re-runs (~80 min) | 0 |
| 22:00–01:00 | P4 (a) then (b), time-boxed | 1 |
| 00:50–02:30 | P1.1 code, `tpot_slo5`, `tpot_slo15`; P1.2 shape probe | 0 |
| 02:30–04:00 | P2 fix + `green_dyn2` | 0 |
| 04:00–07:00 | P3 bring-up, `pf_mps`, `pf_mps20`, `pf_split` | 0 |
| 07:00–08:00 | P5 | – |

Rules carried over: never edit a running bash script; never quote process-name patterns on the
launching command line; keep pinned host memory to a few GiB; one engine stack per GPU; commit
after each section; do not touch `proposal/` or `project_summary/`; do not push to Overleaf.
