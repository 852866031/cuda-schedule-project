# LLM Inference Workload

This directory contains a small LLM inference workload used as the target application for later GPU tracing and profiling experiments.

## What this directory does

The workload runs inference with a small, controllable script instead of using a serving engine such as vLLM.

It:

- loads a Llama-family causal language model with `transformers`
- reads a prompt set from `prompts.jsonl`
- tokenizes prompts in batches
- runs one **prefill** pass
- runs a manual **decode loop**
- prints overall throughput statistics

The purpose is to provide a **real CUDA-based inference workload** whose kernel activity can later be observed with tools such as CUPTI, Nsight Systems, or Nsight Compute.

## Files

- `generate_prompts.py`  
  Generates a prompt set with relatively regular length and writes it to `prompts.jsonl`.

- `run_llm.py`  
  Loads the model and runs batched inference with a manually written generation loop.

- `prompts.jsonl`  
  Generated prompt set. Each line is one JSON object with an `id` and a `prompt`.

## Requirements

### Python packages

Install the required packages in your environment:

```bash
pip install torch transformers sentencepiece accelerate