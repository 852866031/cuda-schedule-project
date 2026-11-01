# Nsight Systems Inference Pipeline Example

This directory contains a small two-GPU experiment for showing why **similar GPU activity** does not necessarily imply similar **productive** or **service-level** utilization.

## What this directory does

The experiment launches two inference pipelines in parallel:

- **GPU 0** runs a more efficient pipeline  
  It batches work and executes larger, more continuous GPU regions.

- **GPU 1** runs a less efficient pipeline  
  It processes requests in a more fragmented way, with more synchronization and smaller GPU work regions.

The goal is to compare what different tools reveal:

- **NVML / `nvidia-smi`**: coarse device-level busy-time signal
- **DCGM**: aggregated hardware activity such as `SM_ACTIVE` and `SM_OCCUPANCY`
- **Nsight Systems**: full CPU–GPU timeline, including launches, copies, synchronization, and overlap

## Files

- `inf_sys.py`  
  Main two-GPU inference pipeline example.

- `nvml_run.py`  
  Runs `inf_sys.py` while sampling NVML metrics.

- `dcgm_run.py`  
  Runs `inf_sys.py` while sampling DCGM metrics.

- `plot_all.py`  
  Generates a combined figure with NVML and DCGM results.

- `Makefile`  
  Convenience commands for running the experiment and profiling it with Nsight Systems.

## Requirements

### System tools
- NVIDIA driver
- CUDA toolkit
- **Nsight Systems** (`nsys`)
- **DCGM** (`dcgmi`)

### Python packages
Install in your environment:

```bash
pip install torch matplotlib nvidia-ml-py