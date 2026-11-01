# Nsight Compute Workloads

This directory contains a small two-GPU experiment for studying why **similar aggregated GPU activity** can still correspond to **very different kernel efficiency**.

## What this directory does

The experiment launches two workloads in parallel:

- **GPU 0** runs `dense_fma_kernel`  
  A regular dense compute kernel that repeatedly updates dense vectors.

- **GPU 1** runs `bad_gather_kernel`  
  An irregular gather kernel that repeatedly follows index chains and reads from scattered memory locations.

The goal is to show:

- DCGM can report similar **SM activity / occupancy**
- but **Nsight Compute** can reveal that one kernel is much less efficient than the other

## Files

- `workload_two_gpu.py`  
  Main workload file. Launches the two kernels on two GPUs.

- `dcgm_run.py`  
  Runs the workload while collecting DCGM metrics.

- `plot_dcgm.py`  
  Plots DCGM results from CSV.

- `tuner.py`  
  Searches parameter settings to make GPU 0 and GPU 1 look more similar in DCGM.

- `Makefile`  
  Convenience targets for running DCGM and Nsight Compute experiments.

## Requirements

### System tools
- NVIDIA driver
- CUDA toolkit
- **DCGM** (`dcgmi`)
- **Nsight Compute** (`ncu`)

### Python packages
Install in your environment:

```bash
pip install torch cupy-cuda13x matplotlib