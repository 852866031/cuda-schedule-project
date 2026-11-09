# Colocation VRAM-adjust prototype

Design: [DESIGN.md](DESIGN.md). Run log with predictions and outcomes: [report/DECISIONS.md](report/DECISIONS.md).
Report: [report/REPORT.md](report/REPORT.md) (English) and [report/REPORT_zh.md](report/REPORT_zh.md) (Chinese, plain language); figures and the rendered diagrams in `report/figures/`. Environment: `../kvcached_validation/.venv-main` (vLLM 0.30 +
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
