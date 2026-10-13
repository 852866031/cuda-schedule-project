# Next-session prompt — after the N-models-on-one-GPU study

You are continuing a research project on **VRAM limits and GPU sharing** on a workstation
with **2× RTX 5090 (32 GB each, 31.35 GiB usable, PCIe Gen4 ×8, no P2P — GeForce), 32
logical cores, 60.5 GiB host RAM, 7.6 GiB swap**. Repo:
`/home/jiaxuan/Documents/Projects/vram-limit` (branch `vram-limit`, pushed to
`origin/vram-limit`). Read this fully before touching anything.

## 0. How to read this project (do this first)

- `CLAUDE.md` — the non-negotiable environment, safety, layout and measurement
  conventions. READ IT. It was updated at the end of the last study.
- `README.md` — one-line result per study, with links.
- `reports/` holds one write-up per completed study. The arc, in order:
  1. `report_simple_inference.md` / `report_lmcache_inference.md` — single-model serving,
     KV oversubscription, LMCache KV offload to host DRAM.
  2. `report_split_inference.md` — the split-LMCache stack (prefill GPU0 → DRAM store →
     decode GPU1).
  3. `report_finetune.md` — fine-tuning memory behavior alone.
  4. `report_colocation_ft.md` — colocate fine-tuning on the decode GPU (MPS, SM caps,
     idle-window gate).
  5. `report_colocation_inf.md` — colocate one second inference model beside an 8B decode.
  6. **`report_colocation_multi.md`** — N identical models on one GPU. THIS is the most
     recent study and what you build on. Read all of it, including Appendix A (the CPU
     thread failure) and Appendix B (the workload). A short version in supervisor-facing
     order is `report_colocation_multi_summary.md`; an HTML version for researchers is
     `colocation_multi_story.html`.
- `PLAN*.md` are pre-registration docs — never rewrite their predictions, only annotate
  outcomes. `PLAN_MULTI_COLOC.md` has the last study's predictions and how they fared.
- `DECISIONS_MULTI_COLOC.md` — the last study's decision log: every run-level decision,
  each overnight experiment's prediction written before it ran, the outcome, and (section
  F) what is still open. Keep one like it for the new study.
- Data lives under `output/`: per-cell summaries `output/summary_mc_{small,medium}*.csv`;
  raw per-cell JSON in `output/raw/` (**gitignored**); per-model client outputs in
  `output/inf_multi_coloc/`; per-second telemetry in `output/gpumon/` (`<tag>.csv` GPU via
  DCGM, `<tag>_host.csv` CPU by process role + RAM + swap, `<tag>_threads.csv` per-thread
  EngineCore counters); engine logs in `output/logs/` (archived per launch under
  `output/logs/mc_archive/`).
- Figures in `figures/`, one plotting script per figure family in `scripts/plots/`
  (`plot_mc_scaling.py`, `plot_mc_capacity.py`, `plot_mc_collapse.py`,
  `plot_mc_layout.py` for the last study).

## 1. Environment & safety — these have frozen/rebooted the box before

- Two venvs: `.venv-matched` runs **vLLM 0.15.1 (pinned)**; `.venv` for everything else.
- Every vLLM process needs `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` (launch
  scripts set it). `lmcache 0.4.4` is the ONLY version that pairs with this vLLM (Python
  fallback copy path). `PYTHONHASHSEED=0` is mandatory cross-process for LMCache.
- **More than one vLLM engine on the box: export `OMP_NUM_THREADS=4` (≤ cores ÷ engines)
  before the driver.** Each engine's torch/OpenMP pool defaults to all 32 cores (~135
  threads per EngineCore). Uncapped, 4 engines with DRAM-offloaded KV spin-wait on every
  core, the GPU idles and requests time out — a collapse that looks like a KV-path hang.
  This was the single biggest finding of the last study; do not rediscover it.
- **Pinned host memory overcommit hard-freezes the machine** (not an OOM-kill). Launch
  stacks ONLY through the launch scripts (RAM gate + `scripts/common/mem_guard.sh` +
  heartbeat). Run experiments **strictly serially** — one stack set at a time.
- **Host RAM is the ceiling for offload.** Budget ~5 GB RSS per engine (API server +
  EngineCore incl. 1 GiB pinned LMCache L1) plus the store's working set. The box was
  already swap-backed at 6 × 0.5B with 3.4 GiB stores.
- **Poll long runs for stalls**; every KV-transfer failure is a silent hang. The driver has
  a stall watchdog on the engines' generated-token counters.
- No ptrace, perf or sudo: no stack sampling. `thread_sampler.py` reads per-thread `/proc`
  counters instead.
- DCGM telemetry is **per-GPU, not per-process**. Trust client latencies and per-engine
  `/metrics` deltas for per-model claims.
- Never reuse a config or telemetry name. Raw files are overwritten by name; per-launch
  telemetry was clobbered once in the last study (the driver now appends `_L2`, `_L3`).

## 2. Key findings to carry forward (from report_colocation_multi.md)

Setup: N identical vLLM engines on GPU1 under MPS (default equal share), each with its own
LMCache DRAM store; Qwen2.5-0.5B (util 0.08 → 3.22 GiB per engine, 1.04 GiB KV grant,
N=1–8) and Qwen2.5-3B (util 0.22, `max-num-seqs 16` → 7.65 GiB, 0.94 GiB KV grant,
N=1–4); 2 QPS per model, 6144-token prefixes, 128 output tokens; four cells: decode-only /
full × KV fits / KV 3× over the grant (offload).

- **With stock settings the first failure is host CPU threads, not the GPU** (above).
  Decode-only offload collapsed at N=4 in 4 of 4 stock runs (0.5B) and 1 of 1 (3B); N=8
  resident stormed transiently in 2 of 4 stock runs, 0 of 3 capped. Measured at the thread
  level (27 threads burning 32 cores in user mode, involuntary switches ×42).
- **Under MPS, N models cost per-token latency, not capacity.** 0.5B: TPOT 1.56 → 5.70 ms
  at N=8 (3.6×), every model still serving its load; N=8 still serves 16 QPS per model
  (15.3k tok/s) with zero queueing. 3B: 6.78 → 17.31 ms at N=4 (2.55×), VRAM full at N=4.
- **The cost of N is lost batching** (the user's Hypothesis 1, measured): at 32 QPS total,
  one 0.5B engine 1.97 ms, four 4.56 ms, eight 7.78 ms per token. Applies to identical
  models only — different models could not batch together anyway.
- **Offload:** free for the 0.5B once threads are capped (TPOT on the resident curve to
  N=6; host RAM is the limit). For the 3B, requests queue on KV reloads and the queue
  grows with N (mean wait 0.08 s → 4.7 s for N=1→4): each prefix is 216 MiB and the grant
  holds ~4.
- **KV grant vs concurrency:** at a fixed footprint the KV grant trades against
  `--max-num-seqs` (3B: 64 → 0.59 GiB, 16 → 0.94 GiB). The 3B's measured capacity (12 / 8 /
  4 QPS per model at N=1/2/4) is that cap, not the GPU. Aggregate capacity still grows
  with N.
- **Measurement conventions that came out of it:** decode-only TTFT is N/A; capacity is
  judged by the engine's scheduler queue wait, not delivered/offered and not SM-active;
  a cell is ok only with zero failed requests.
- **Not established** (see DECISIONS section F): most points are n=1; capacity windows are
  short (300 requests, 10–20 s at the highest rates) and cover decode-only · fits only;
  the 0.5B capacity limit and the GPU-bound 3B capacity were not found; full vs
  decode-only under offload is confounded by session skew (zipf vs uniform); which
  resource 3B reloads queue on is unidentified.
- **Deliberately not run:** no-MPS vs N, priority/gate between models, prefill-only with
  the external scheduler (`mc_scheduler.py` + the `pfill` cell exist, untested), mixed
  model sizes.

## 3. Your experiment

> **TO BE FILLED IN BY THE USER.** State the goal, the models, the configurations, the
> sweep axis and the metrics, as the previous prompts did.

Candidate directions the last study left open, for reference only — do not start any of
them unless section 3 says so:

- **Mixed model sizes on one GPU** (e.g. 0.5B + 3B together). Planned as "later" in the
  last brief. Hypothesis 1 predicts no batching is lost here, so the colocation case is
  stronger than for identical models.
- **Prefill-only with an external cross-model scheduler** (admission / round-robin /
  priority across models). Code exists, never run; needs a single-model prefill capacity
  calibration first.
- **Closing the open items** above (zipf control, fixed-duration capacity windows,
  repeats, GPU-bound 3B capacity with `max-num-seqs 64`).
- **Priority between co-located models** (the idle-window gate from the fifth and sixth
  studies) — the user ruled this out for the last study; ask before reviving it.

## 4. Deliverables & discipline

- A pre-registration note (`PLAN_<study>.md`, predictions before measuring) and a decision
  log (`DECISIONS_<study>.md`: each experiment's prediction before it runs, its outcome
  after, including wrong predictions).
- A new self-contained script directory (copy, don't import, from
  `scripts/inf_multi_coloc/`, as each study has done).
- A report in the established style: front-loaded setup with exact configuration numbers
  (model size, VRAM per engine, KV grant), per-figure **observation lists** (bold claim +
  short gloss), a **results table under each figure**, long diagnostics in an appendix
  with a short "why" paragraph inline. Say "model", not "cohort". One condition per plot
  when a condition fails for a different reason. View every PNG before trusting it.
- Every explanation needs an experiment behind it; mark inferences as inferences; rerun
  suspicious points; state confounds.
- The user talks in Chinese; reports, figures and messages are in English unless told
  otherwise. Scope is the user's: confirm before queueing arms that were not asked for,
  and before spending more than ~15 minutes of GPU time beyond the request.
- Commit in logical chunks; push to `origin/vram-limit`. `output/raw/` and `output/logs/`
  are gitignored — commit source, figures, reports, summaries, telemetry and client
  outputs.

Housekeeping left from earlier sessions: six untracked `infc_B_fits_{eager,gate,gateq4}`
files in `output/gpumon/` and `output/inf_inf_coloc/` belong to the sixth study and were
never committed; the three in `output/gpumon/` were extended by stray monitors on
2026-10-01. Ask the user whether to commit or delete them.

Start by reading `CLAUDE.md`, `reports/report_colocation_multi.md` and
`DECISIONS_MULTI_COLOC.md`, then confirm section 3 with the user before scaffolding.
