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
