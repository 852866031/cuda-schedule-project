# CUPTI Tracing Workflow

A CUPTI-based tracing workflow for CUDA workloads. It includes a simple LLM inference app, an injection-based CUPTI tracer, and plotting scripts for hotspot and overhead analysis.

## Directory structure

```
cupti/
├── Makefile                        # top-level build and run orchestration
├── plot_hotspots.py                # visualization script (hotspot charts + overhead comparison)
├── tracer/                         # CUPTI activity tracer (see tracer/README.md)
│   ├── activity_tracer.cpp         # tracer shared library source
│   ├── example_workload.cu         # standalone CUDA workload for testing the tracer
│   └── Makefile
├── llm_app/                        # LLM inference workload
│   ├── run_llm.py                  # batched Llama-3-8B inference with manual decode loop
│   ├── generate_prompts.py         # generates the prompt dataset
│   └── prompts.jsonl               # 64 pre-generated prompts
└── output/                         # collected metrics and generated plots
    ├── llm_metrics.json
    ├── llm_trace_metrics.json
    ├── overhead_compare.csv
    ├── kernel_hotspots_global.csv
    ├── kernel_hotspots_recent_5s.csv
    └── plots/
```

## Quick start

```bash
make llm          # run the plain LLM workload, write output/llm_metrics.json
make llm-trace    # run the same workload with the CUPTI tracer attached,
                  #   write output/llm_trace_metrics.json + hotspot CSVs
make compare      # run both modes and produce output/overhead_compare.csv
make plot         # generate hotspot and overhead figures in output/plots/
```

`make all` (default) just builds the tracer library.

## Components

### Tracer (`tracer/`)

A shared library that attaches to an unmodified CUDA process via `CUDA_INJECTION64_PATH`. It records every kernel's launch count and execution time and writes two CSV files on exit:

- `kernel_hotspots_global.csv` — all-time statistics
- `kernel_hotspots_recent_5s.csv` — rolling 5-second window

See [tracer/README.md](tracer/README.md) for build instructions and implementation details.

### LLM workload (`llm_app/`)

Runs batched autoregressive inference with Llama-3-8B using HuggingFace Transformers. One prefill pass is followed by a manual token-by-token decode loop (32 new tokens, batch size 8, float16). Metrics written as JSON include total time, tokens/sec, and requests/sec.

Model: `meta-llama/Meta-Llama-3-8B`. Requires `torch`, `transformers`, `sentencepiece`, and `accelerate`.

### Plotting (`plot_hotspots.py`)

Produces two sets of figures:

- **Hotspot charts** — top-10 kernels by total duration and by launch count, for both the global and recent-5s windows. Raw C++ mangled names are mapped to readable labels (Flash Attention, CUTLASS GEMM, LayerNorm, etc.).
- **Overhead comparison** — side-by-side bar chart of execution time and tokens/sec with and without the tracer.

## Example results

From a single run on Llama-3-8B with 64 prompts (2048 tokens generated):

| Mode | Time (s) | Tokens/sec |
|------|----------|------------|
| Baseline | 5.88 | 348 |
| With tracer | 5.99 | 342 |
| Overhead | +1.9% | — |

Top kernels by GPU time: CUTLASS GEMM (~2791 ms total, 55 800 launches), Flash/MemEff Attention (~359 ms, 8192 launches).
