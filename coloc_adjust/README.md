# Colocation VRAM-adjust prototype

Design: [DESIGN.md](DESIGN.md). Run log with predictions and outcomes: [report/DECISIONS.md](report/DECISIONS.md).
Reports (English and Chinese) are in [`../reports/`](../reports/): 1 KV budget and colocation schemes, 2 bandwidth interference, 3 C1/C2, 4 prefill tenant. Figures and the rendered diagrams they use are in `report/figures/`. Environment: `../kvcached_validation/.venv-main` (vLLM 0.30 +
kvcached main + peft).

| file | role |
|---|---|
| `hp_patch/sitecustomize.py` | observation only, loaded into the engine via `PYTHONPATH`: busy-window page for the gate (`COLOC_HP_SIGNAL=1`), native-offload transfer tracer (`COLOC_XFER_TRACE=1`) |
| `controller.py` | streaming HTTP router (:8001 → vLLM) that tracks every request's prompt/generated/max tokens, derives the KV block demand, sets kvcached's limit (inference first) and publishes the tenant budget to `/dev/shm/coloc_ft_budget` |
| `ft_worker.py` | tiered LoRA tenant (Qwen2.5-0.5B): micro-batching → activation offload → weight streaming → pause; DRAM snapshots; idle-window gate; transfer records |
| `layer_offload.py` | `LayerStreamer` / `OffloadManager` from the finetune study (unchanged) |
| `xfer_trace.py` | shared CSV writer for host↔GPU transfer records |
| `tiers.json` | calibrated per-config peak need (`ft_worker.py --calibrate`) |
| `green_ctx.py` | CUDA green contexts from Python (ctypes on libcuda): carve N SMs, get a stream; `python green_ctx.py 16 32 64` benchmarks |
| `diagrams/*.drawio`, `diagrams/export_png.py` | system architecture, tenant tiers, VRAM scheduler loop, compute-sharing mechanisms; the exporter renders them to `report/figures/` with headless Chrome |
| `run_eval.sh <scheme>` | one scheme end to end (MPS, engine, router, sampler, tenant, load, teardown, analysis): decode_alone, adjust_notenant, static_split, adjust, adjust_nogate, adjust_mps20 |
| `run_eval5.sh <scheme>` | round 2 (500 W, measured accounting): all schemes above plus `tpot_slo5` / `tpot_slo15` (tenant tokens/s throttled from the decode-TPOT error) and `green_dyn2` (prebuilt green contexts); stops only its own engine, so GPU-1 work can run alongside |
| `run_eval_pf.sh <scheme>` | round 2, prefill tenant: `pf_mps`, `pf_mps20`, `pf_split` — a Qwen2.5-3B fp8 vLLM instance (own kvcached instance, KV offloaded to host RAM, `max_tokens=1`) driven by `pf_tenant.py`, which turns the published budget into the 3B's kvcached limit / admission pause / sleep |
| `run_probe.sh`, `analyze_probe.py` | round 2 shape probe: each tenant shape alone at 1 req/s under the same tokens/s cap → decode slowdown per 1k tenant tok/s |
| `plot_dram.py` | DRAM-controller busy vs tenant throughput and vs decode slowdown (the direct bandwidth test) |
| `../kvcached_validation/scripts/p4_lmcache.sh` | kvcached + LMCache MP connector tests on GPU 1 in `.venv-lmc` (patched kvcached: `KVCACHED_EXPORTABLE`, `KVCACHED_LMCACHE_PINS`) |
| `results_600w/` | round-1 results (600 W power limit, design-constant accounting); `results/` holds round 2 |
| `run_eval2.sh` / `run_eval3.sh` | the same plus the green-context schemes (green32, green_dyn, engine_green) and the optimisation scheme (green_opt); separate files only because a running bash script must not be edited |
| `teardown.sh` | stop everything |
| `analyze_coloc.py`, `compare_arms.py`, `plot_arms.py`, `plot_workload.py` | per-arm phase tables + timeline/transfer figures; cross-arm table (+ dense all-arms figure `arms_all.png`); the report's Figures 1–2 (`arms.png`, `arms_tradeoff.png`); the workload-trace figure |

```bash
cd coloc_adjust
../kvcached_validation/.venv-main/bin/python ft_worker.py --calibrate --fresh   # GPU 1, writes tiers.json
for arm in decode_alone adjust_notenant static_split adjust adjust_nogate adjust_mps20; do bash run_eval.sh $arm; done
for arm in green32 green_dyn engine_green green_opt; do bash run_eval3.sh $arm; done
../kvcached_validation/.venv-main/bin/python compare_arms.py --dir results --arms decode_alone adjust_notenant static_split adjust adjust_nogate adjust_mps20 green32 green_dyn engine_green green_opt --out results/arms
../kvcached_validation/.venv-main/bin/python plot_arms.py --dir results --out results/arms
```
Do not quote the process-name patterns used by the scripts (`vllm serve`, `ft_worker.py`, …) in
the command line that launches them: `pkill -f` inside the scripts would match that shell.

## Environment traps (read before running anything)

- Model weights are only under `/mnt/storage/huggingface`; tool shells do not read `~/.profile`, so every launcher must `export HF_HOME` itself.
- Check that the GPUs are idle first (other projects use them); one MPS daemon and one pipe directory per GPU.
- Too much pinned host RAM freezes the machine; keep LMCache's pinned L1 to a few GiB. `/dev/shm` is 31 GiB, and both engines' host-RAM tiers must fit in it together.
- `stop.sh` kills every vLLM; `pgrep -f` / `pkill -f` match their own shell. A cleanup that runs `pkill -f ft_worker.py` on one card also kills a tenant on the other card (it did on Nov 16): the launchers now spare `corun2_` / `tenant_bytes` processes.
- Turn the FlashInfer sampler off (`VLLM_USE_FLASHINFER_SAMPLER=0`); the LMCache server uses port 5556; vLLM sleep needs `VLLM_SERVER_DEV_MODE=1`.
- With KV offload use kvcached main and set `KVCACHED_MAX_CACHED_TOKENS=-1`.
- The decode baseline on a "remaining" green context sits on one of two levels; compare small differences (50–100 GB/s, 1–3 %) across restarts.
- A compiled tenant (`--compile`) stalls for about 46 s inside a step when it meets a shape for the first time; it compiles the shapes that fit at start and uses only those.
- A vLLM engine that moves from the stream it started on to a green context faults (Xid 31) and takes the other MPS client down with it, with CUDA graphs on or off (reports/4, section 5).
- Do not push to Overleaf; do not touch `proposal/` or `project_summary/`.
