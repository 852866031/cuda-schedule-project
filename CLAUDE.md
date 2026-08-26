# Working in this repo

Four completed studies of VRAM limits on a 2× RTX 5090 (32 GB, PCIe Gen4 ×8, no P2P —
GeForce) workstation with 60 GiB RAM. Plans (`PLAN*.md`) are pre-registration documents:
never rewrite their predictions, only annotate outcomes. Results live in `reports/`,
figures in `figures/`, all data under `output/` (summaries, raw JSONs, per-second GPU
telemetry in `output/gpumon/`, engine logs in `output/logs/`).

## Environments — non-negotiable details

- `.venv-matched` runs vLLM **0.15.1 (pinned)**; `.venv` is for everything else.
- Every vLLM invocation needs `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`
  (conda's libstdc++ is too old). Launch scripts set it.
- **lmcache 0.4.4 is the only release that pairs with this vLLM** (older = torch-ABI
  mismatch, 0.5.x = adapter break; its compiled ops never load against torch 2.9.1 —
  a source build deadlocks at engine init; the Python fallback is what all measurements
  used). pip drags `transformers` to 5.x when installing it — re-pin `<5` after.
- `PYTHONHASHSEED=0` is mandatory for any cross-process LMCache use.

## Safety — this machine has been frozen and rebooted by pinned memory

Pinned (page-locked) host memory is unswappable; overcommitting it does not OOM-kill, it
hard-freezes the box. The P2P connector defaults to **32 GiB pinned per instance**.
Rules: never launch vLLM stacks except through the launch scripts (they budget-check),
keep `scripts/common/mem_guard.sh` running (they start it), and remember the pool
allocator rounds sizes up to powers of two. The stall/heartbeat watchdogs exist because
every failure mode of the KV-transfer path is a silent hang, not an error.

## Layout

- `scripts/inference/simple/` — colocated study (also `--backend lmcache`).
- `scripts/inference/split_simple/` — hand-built P2P split + `p2p_patch/` (four
  connector fixes, loaded via PYTHONPATH sitecustomize) + `disagg_sweep.py`, the
  budget-sweep driver for BOTH split stacks (`--stack p2p|lmcache`).
- `scripts/inference/split_lmcache/` — shared-LMCache split (server wrapper adds
  SO_REUSEADDR; launch with `--warmup-qps 0.5` or the L1 staging pool overflows).
- `scripts/plots/` — one script per figure family; regenerate after data changes and
  **always view the PNG before finishing** (label collisions are the recurring bug).

## Measurement conventions

- The reference workload is identical across all studies (32 sessions × 6144-token fixed
  prefixes + 128 unique suffix tokens, 128 out, zipf-1.1, open-loop 2 QPS, seeded and
  fully deterministic). Don't invent variants.
- Split-system TTFT is only honest with the router's `--forward-first-token` and no
  admission cap (`--max-inflight 999`); the capped/discarding router variants were setup
  faults — do not report their numbers.
- Config names must never collide across arms/runs (name suffixes like `_lmcache`,
  `_fwd_ng`); raw files are overwritten by name, and original study data has been lost
  that way once.
- p50 TTFT run-to-run variance on flat regions is ±15 ms; throughput dips are the
  trustworthy knee signals.
