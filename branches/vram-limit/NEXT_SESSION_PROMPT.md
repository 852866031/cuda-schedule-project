# Next-session prompt — read the project first

You are joining a research project on **VRAM limits and GPU sharing** on a workstation
with 2× RTX 5090 (32 GB each), 32 cores and 60 GiB host RAM. Repo:
`/home/jiaxuan/Documents/Projects/vram-limit` (branch `vram-limit`).

Before I give you the next experiment, read the project so you have the context of what
has been done. **Do not run anything on the GPUs and do not change any files yet.**

## Read in this order

1. `CLAUDE.md` — environment, safety, layout and measurement conventions. These rules are
   non-negotiable; several of them exist because the machine has been frozen or data has
   been lost before.
2. `README.md` — the table at the top gives one line per completed study, with links.
3. The most recent study, in full. This is what the next experiment builds on:
   - `reports/report_colocation_multi_summary.md` — the short version, start here.
   - `reports/report_colocation_multi.md` — the full report, including Appendix A (the
     CPU-thread failure and how it was diagnosed) and Appendix B (the workload).
   - `PLAN_MULTI_COLOC.md` — what was predicted before measuring, and how each prediction
     fared.
   - `DECISIONS_MULTI_COLOC.md` — every decision taken, each experiment's prediction and
     outcome, and (section F) what is still open.
4. The two studies before it, for the background the last one assumes:
   - `reports/report_colocation_inf.md` — one second inference model beside an 8B decode
     (why MPS is mandatory; KV offload as the inference tenant's hazard).
   - `reports/report_colocation_ft.md` — a fine-tuning neighbor on the decode GPU (MPS
     caps, the idle-window gate).
5. Skim the earlier reports only as needed: `report_split_inference.md` (the split
   prefill/decode stack over LMCache), `report_lmcache_inference.md`,
   `report_simple_inference.md`, `report_finetune.md`.
6. The code of the last study: `scripts/inf_multi_coloc/` (start with `multi_sweep.py` and
   `multi_launch.sh`; the `queue_*.sh` files show the exact order of what was run) and its
   plot scripts `scripts/plots/plot_mc_*.py`.

## Where things are

- Reports in `reports/`, figures in `figures/`.
- Data in `output/`: per-cell summaries `output/summary_mc_*.csv`; raw per-cell JSON in
  `output/raw/` (gitignored); per-model client outputs in `output/inf_multi_coloc/`;
  per-second GPU, host and per-thread telemetry in `output/gpumon/`; engine logs in
  `output/logs/` (gitignored).

## When you have finished reading

Tell me, briefly:

- what the last study set out to answer, how it was set up, and its main findings;
- the rules you must follow on this machine;
- what the last study left open or deliberately did not run.

Then wait. I will describe the next experiment after that.
