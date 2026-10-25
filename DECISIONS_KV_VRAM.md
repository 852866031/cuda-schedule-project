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
