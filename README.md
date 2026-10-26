# KV slots on the decode GPU

**How do the decode engine's KV space, the number of requests it runs, and its VRAM budget
relate — and which techniques buy more requests per GiB?**

Branch `kvcache-vram_size` of the vram-limit repository, cut from `bcb0315` (the end of
the ninth study). Same machine: 2× RTX 5090 (32 GB), PCIe Gen4 ×8, no P2P, 32 cores,
60 GiB RAM. The stack is the Llama-3-8B split from the earlier studies — prefill on GPU0,
decode on GPU1, KV handed over through a shared LMCache store — with **nothing else on
GPU1**. Prefill is only the source of KV; everything measured here is decode.

| document | role |
|---|---|
| [reports/report_kv_slots.md](reports/report_kv_slots.md) | **baseline report (E0–E3)**: slots, scheduler, grid, queueing model, load stalls |
| [NOTES_KV_SLOTS.md](NOTES_KV_SLOTS.md) | starting facts, all re-derivable from the prior study's data |
| [PLAN_KV_VRAM.md](PLAN_KV_VRAM.md) | pre-registration: experiments and predictions (never rewritten, only annotated) |
| [DECISIONS_KV_VRAM.md](DECISIONS_KV_VRAM.md) | per-run decision log: prediction before, outcome after |

## What is kept from the main branch

Only what this study runs or cites. Everything else is on the `colocation` /
`vram-limit` branches.

- `scripts/common/` — `mem_guard.sh`, `gpu_monitor.py`, `calibrate_pcie.py`, `status.py`
- `scripts/inf_pf_prio/` — the newest self-contained 8B split-LMCache stack; the new
  `scripts/kv_vram/` is copied from it with the 3B neighbour removed
- the ninth study's results: [reports/report_colocation_prio.md](reports/report_colocation_prio.md),
  `PLAN_PF_PRIO.md`, `DECISIONS_PF_PRIO.md`, `figures/pfp_*.png`, `output/summary_pfp.csv`,
  `output/inf_pf_prio/`, `output/gpumon/pfp_*`, `scripts/plots/plot_pfp_results.py`
- [reports/report_split_inference.md](reports/report_split_inference.md) — where the
  split-LMCache stack came from
- `output/calibration_pcie.json` — an earlier pinned H2D/D2H calibration (torch 2.8 dev)

Untracked local data copied in: `output/raw/pfp_*.json` (80 files) and
`output/logs/pfp_archive/` (per-cell engine logs). `.venv` and `.venv-matched` are
**symlinks** to `../vram-limit/` — never delete their targets.

## ⚠️ Before running anything

Read [CLAUDE.md](CLAUDE.md). Pinned host memory overcommit hard-freezes this box; only
launch stacks through the launch scripts. This worktree and `../vram-limit` share the
GPUs and ports — only one directory may run a stack at a time (check `nvidia-smi` and
`ss -ltn` for 8000/8100/8200/8300 first).
