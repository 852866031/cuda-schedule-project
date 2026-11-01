# NanoFlow reproduction

Paper: *NanoFlow: Towards Optimal Large Language Model Serving Throughput* (arXiv:2408.12757)
Code: https://github.com/efeslab/Nanoflow (branch `Nanoflow-python`, cloned in `./Nanoflow`)

Date: 2026-08-11

## Goal

"Just run it": build NanoFlow on this machine and run its end-to-end Llama3-8B
serving pipeline, measuring offline throughput.

## Hardware gap vs paper

| | Paper | This machine |
|---|---|---|
| GPU | 8× A100 80GB SXM (NVLink) | 2× RTX 5090 32GB (sm_120) |
| Headline eval | Llama2-70B, TP=8 | infeasible (needs ~140GB+) |
| Feasible eval | Llama3-8B, 1 GPU | Llama3-8B on **GPU 1** |

The `Nanoflow-python` branch (current default) is a rewrite: GEMMs go through
torch/cuBLAS + nvmath, attention through FlashInfer, and only small custom CUDA
kernels (rmsnorm, silu, rope, embedding, sampling) are compiled. The old
CUTLASS sm_90a GEMM backend is disabled upstream, which is what makes running
on Blackwell possible at all.

## Environment

- conda env `nanoflow` (python 3.12): torch 2.7.1+cu128, flashinfer 0.2.11.post1
  (built from the repo submodule), nvmath-python, transformers; conda-forge
  `liburing`, `pybind11`, `nccl` (no sudo available, so no apt packages).
- CUDA 12.8 system toolkit; driver 580.
- mscclpp/Gurobi/nsight NOT installed — not needed for the single-GPU python path
  (mscclpp is only used by commented-out build targets; the auto-search MILP
  result file referenced by the demo is missing from the repo anyway, see below).

## Changes made (all inside ./Nanoflow, on top of origin/Nanoflow-python)

`./Nanoflow` is a vendored copy of upstream commit `f179a907` (branch
`Nanoflow-python`, including its 3rdparty submodules) with the changes below
applied directly in-tree.

1. `pybind/CMakeLists.txt`: `CMAKE_CUDA_ARCHITECTURES` 90 → **120** (RTX 5090);
   added an `-O3` flags branch for 120 (the sm_90a flags don't apply).
2. `platform_config.py`: `PLATFORM_CUDA=True` (upstream ships all-False; the
   default "cuda" algo tags require the compiled bind modules).
3. `models/llama3_FlashinferKVCache.py`: KV pool page count made configurable via
   `NANOFLOW_KV_PAGES` (upstream hardcodes 2048*26 pages ≈ 106GB(!) — sized for
   much bigger memory).
4. `entry/run_llama3_5090.py` (new, based on `run_llama3.py`):
   - weights/tokenizer from ungated `NousResearch/Meta-Llama-3-8B-Instruct`
     (identical mirror of the gated `meta-llama` repo, to which this HF account
     has no access);
   - dropped `profile_result_path="../auto_search/8B_search_result_large_btz.json"`
     — that file does not exist anywhere in the repo history, so the demo as
     shipped crashes; without it NanoFlow uses its built-in 2-way nano-batch
     split (decode + prefill) instead of the MILP-optimized schedule;
   - batch/seq shrunk for 32GB: global batch 2048, decode batch 128, seq len 512,
     KV pool 5120 pages (~10.7GB); CUDA graphs + nano-split enabled as upstream;
   - added per-cycle timing and a tokens/s summary (upstream measures nothing).

## How to run

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate nanoflow
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # liburing
cd Nanoflow/entry
CUDA_VISIBLE_DEVICES=1 python run_llama3_5090.py -l   # first run: converts+caches weights (~16GB under Nanoflow/cached_weights)
CUDA_VISIBLE_DEVICES=1 python run_llama3_5090.py      # perf run
CUDA_VISIBLE_DEVICES=1 python run_llama3_5090.py --correctness  # sanity generation
```

## Results

See [../nano-reproduce.md](../nano-reproduce.md) for the full report.
Best: 12,401 tokens/s/GPU (decode 128 / ctx 512, main-stream nanobatch-only,
CUDA graphs), vs 12,756 tok/s/GPU reported in the paper for Llama3-8B on one
A100 80GB. Green-context SM-partition overlap lost 14–32% on this
hardware/workload (see report analysis).
