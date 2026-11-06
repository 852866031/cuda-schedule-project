# Colocation VRAM-adjust prototype

Design: [DESIGN.md](DESIGN.md). Run log with predictions and outcomes: [DECISIONS.md](DECISIONS.md).
Report: [REPORT.md (English; Chinese plain-language version: REPORT_zh.md)](REPORT.md). Environment: `../kvcached_validation/.venv-main` (vLLM 0.30 +
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
| `diagrams/*.drawio` | system architecture, tenant tiers, controller loop, compute-sharing mechanisms (open in draw.io) |
| `run_eval.sh <arm>` | one arm end to end (MPS, engine, router, sampler, tenant, load, teardown, analysis): decode_alone, adjust_notenant, static_split, adjust, adjust_nogate, adjust_mps20 |
| `run_eval2.sh` / `run_eval3.sh` | the same plus the green-context arms (green32, green_dyn, engine_green) and the optimisation arm (green_opt); separate files only because a running bash script must not be edited |
| `teardown.sh` | stop everything |
| `analyze_coloc.py`, `compare_arms.py`, `plot_arms.py` | per-arm phase tables + timeline/transfer figures; cross-arm table (+ dense all-arms figure `arms_all.png`); the report's Figures 1–2 (`arms.png`, `arms_tradeoff.png`) |

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
