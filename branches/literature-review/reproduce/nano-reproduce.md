# NanoFlow Reproduction Report

**Paper:** NanoFlow: Towards Optimal Large Language Model Serving Throughput (arXiv:2408.12757, efeslab)
**Code:** https://github.com/efeslab/Nanoflow, branch `Nanoflow-python` (default), cloned at `reproduce/nanoflow/Nanoflow`
**Date:** 2026-08-11
**Machine:** 2× RTX 5090 32GB (Blackwell, sm_120, 170 SMs), CUDA 12.8, driver 580, 60GB RAM, no sudo. All runs on **GPU 1** (GPU 0 hosts Xorg + a tally server).

## TL;DR

NanoFlow's Llama3-8B pipeline **builds and runs end-to-end on an RTX 5090** after porting
(arch flags, memory downsizing, and fixing two upstream bugs in the non-auto-search code
path). Best measured offline throughput: **12,401 tokens/s/GPU** (2048-token dense batch,
CUDA graphs, 2-way nano-batching). The paper's Figure 11 reports **12,756 tokens/s/GPU**
for the same model on a single A100 80GB — so the pipeline lands in the same ballpark on
a 32GB consumer card. The paper's headline Llama2-70B/8×A100 evaluation is infeasible here.
Notably, the paper's core technique — SM-partitioned kernel overlap — **hurt** throughput
in every configuration we tried on this hardware/workload (best overlap config: 10,723
tok/s, 14% below no-overlap); see Analysis.

## What was reproduced

The current default branch (`Nanoflow-python`) is a Python rewrite of the original C++
backend: GEMMs via torch/cuBLAS (+ nvmath for SM-count capping), attention via FlashInfer,
NCCL instead of MSCCL++, and only ~7 small custom CUDA kernels (rmsnorm, silu, rope,
embedding, sampling, io_uring weight loader). The old CUTLASS sm_90a GEMM path is disabled
upstream — this is what makes Blackwell feasible, since cuBLAS/FlashInfer ship sm_120 support.

Pipeline features exercised: nano-batched execution (2-way decode/prefill split), CUDA
graph capture, asynchronous batch scheduling, paged KV cache (FlashInfer), chunked prefill,
green-context SM partitioning (tested, see below).

Not reproduced: 70B/TP-8 evaluation (needs ~8×80GB), MILP auto-search (the search-result
file `auto_search/8B_search_result_large_btz.json` that the demo references **does not
exist anywhere in the repo or its git history** — upstream demo crashes as shipped),
KV-cache SSD offloading, online-latency traces, baseline comparisons (vLLM etc.).

## Results

Steady-state, 20 cycles, global dense batch 2048, CUDA graphs + nano-split on, fp16:

| Config | Streams | Cycle time | Throughput (tok/s/GPU) |
|---|---|---|---|
| decode 128 / ctx 512 | all ops on main stream (all 170 SMs) | 165.1 ms | **12,401** |
| decode 256 / ctx 256 | main stream | 169.6 ms | 12,078 |
| decode 128 / ctx 512 | green-ctx overlap COMP=144 / MEM=16 | 191.0 ms | 10,723 |
| decode 128 / ctx 512 | overlap COMP=152 / MEM=8 | 199.7 ms | 10,257 |
| decode 128 / ctx 512 | overlap COMP=120 / MEM=8 (H100-sized splits) | 215.2 ms | 9,518 |
| decode 128 / ctx 512 | overlap COMP=104 / MEM=24 | 242.5 ms | 8,444 |
| decode 256 / ctx 256 | overlap COMP=144 / MEM=16 | 194.1 ms | 10,550 |

Reference points from the paper:
- Fig. 11 (feasibility, input 1024 / output 512): Llama-3-8B on **one A100 80GB**:
  NanoFlow **12,756** tok/s/GPU (78.5% of optimal), vLLM 5,187 (31.9%).
- Fig. 9 (ablation, 70B/8×A100): full NanoFlow beats nanobatch-only by only ~1–2%,
  and beats non-overlap by 1.07–1.17×.

Correctness: 4-request batched generation with KV cache produces coherent, identical
continuations for identical prompts ("Hi, who are you? I'm a 25-year-old software
engineer…"), and the perf run's sampled output is coherent English about transformers.

### Caveats on comparability

- Our decode share is small (128–256 decode requests of a 2048 dense batch, ctx ≤512)
  because 32GB must hold 16GB fp16 weights + KV pool. The paper's 8B run keeps
  ~640 decode requests at ctx ~1024+ (≈2× more KV than our whole GPU). Decode-heavy
  mixes are exactly where overlap pays; we cannot reach that regime.
- Tokens/s counts all processed tokens (prefill chunk + decode) per iteration, matching
  the paper's offline-throughput accounting.
- Same model family but a per-GPU comparison across A100 80GB ↔ RTX 5090 is loose:
  the 5090 has ~2.5× A100's dense fp16 FLOPS and ~0.87× its HBM bandwidth.

## Analysis: why overlap loses here

1. **Workload**: with only 6–12% of batch tokens in decode, decode attention is a tiny
   fraction of cycle time; there is little memory-bound work to hide, but partitioning
   still taxes every GEMM.
2. **Hardware ratio**: RTX 5090 raises compute ~2.5× over A100 while lowering bandwidth;
   the compute-bound fraction of a cycle grows, shrinking overlap headroom further.
3. **Upstream H100 constants**: SM partitions are hardcoded for 132-SM H100s
   (pairs summing to 128), idling 42 of 170 SMs. After making partitions device-sized
   (pairs summing to 160), overlap improved (9,518 → 10,723) but still lost to
   letting cuBLAS/FlashInfer use the whole GPU.
This is consistent with the paper's own ablation (Fig. 9): most of NanoFlow's win over
naive serving comes from dense batching + nano-batched chunked prefill + async
scheduling + CUDA graphs — which we did reproduce — while kernel-level overlap
contributes the last few percent only in decode-heavy, memory-bound regimes.

## Porting steps (all changes inside `reproduce/nanoflow/Nanoflow`)

1. **Build for sm_120**: `pybind/CMakeLists.txt`: `CMAKE_CUDA_ARCHITECTURES` 90→120
   (+ flags branch). All 8 pybind CUDA modules compile and run unmodified on Blackwell.
2. **Env without sudo**: conda env `nanoflow` (py3.12, torch 2.7.1+cu128, flashinfer
   0.2.11.post1 built from submodule, nvmath-python, gurobipy, transformers) with
   conda-forge `liburing`/`pybind11`/`nccl` instead of apt packages; skipped mscclpp
   (only used by commented-out build targets) and Nsight. Runtime needs
   `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` for liburing.
3. **`platform_config.py`**: `PLATFORM_CUDA=True` (ships all-False, which breaks the
   default "cuda" kernel tags).
4. **Gated weights**: `meta-llama/Meta-Llama-3-8B-Instruct` is gated (no access on this
   HF account); used the identical ungated mirror `NousResearch/Meta-Llama-3-8B-Instruct`.
5. **Memory downsizing**: upstream hardcodes a **106GB** KV pool (2048×26 pages) and a
   64GB `gpu_mem` config; made pool size env-configurable (`NANOFLOW_KV_PAGES`, we used
   5120 pages ≈ 10.7GB) and shrank decode batch/context.
6. **Upstream bug**: `config_streams`/`config_algorithm` fallback path (used whenever the
   missing auto-search JSON is absent) passes a single stream/tag to nano-split ops which
   assert for per-nano-op lists — i.e. `use_nano_split=True` cannot work as shipped.
   Patched both to distribute per-nano-op, with env-selectable green-ctx placement
   (`NANOFLOW_COMP_SM`/`NANOFLOW_MEM_SM`).
7. **Upstream bug**: retired requests' KV pages are never freed in the demo loop
   (`DistKVCache.release()` exists but is never called), leaking 120 pages/cycle and
   OOMing a small pool in 6 cycles; the new entry script releases the finished prefill
   request each cycle.
8. **Split weight conversion from serving** (`-l` now exits after caching): the original
   single-process flow holds 2×16GB of weights on-GPU, which OOMs a 32GB card.
9. **Device-sized SM partitions**: green-context split lists derived from
   `multi_processor_count` instead of hardcoded H100 132/128.
10. Added timing/throughput instrumentation (`entry/run_llama3_5090.py`) — the upstream
    demo measures nothing.

## How to rerun

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate nanoflow
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd reproduce/nanoflow/Nanoflow/entry
CUDA_VISIBLE_DEVICES=1 python run_llama3_5090.py -l             # once: convert + cache weights (16GB)
CUDA_VISIBLE_DEVICES=1 python run_llama3_5090.py                # throughput (12.4k tok/s config)
CUDA_VISIBLE_DEVICES=1 python run_llama3_5090.py --correctness  # sanity generation
# knobs: NANOFLOW_DECODE_BATCH, NANOFLOW_SEQ_LEN, NANOFLOW_KV_PAGES,
#        NANOFLOW_COMP_SM / NANOFLOW_MEM_SM (green-ctx overlap; 0 = main stream)
```

## Verdict

- **Runs:** yes — end-to-end serving with nano-batching, CUDA graphs and paged KV on
  Blackwell, after ~10 porting fixes.
- **Headline claims:** not testable here (needs 8×A100 + baselines); the 8B absolute
  throughput is consistent with the paper's single-GPU 8B figure.
- **Reproduction friction:** the released demo does not run as shipped even on intended
  hardware — the referenced auto-search result file is missing and the fallback path has
  two crashing bugs, the demo leaks KV pages, and memory sizes/SM counts are hardcoded
  for 80GB/H100-class GPUs. The paper's evaluated C++ backend (`main` branch) is no
  longer the default and was not evaluated here.
