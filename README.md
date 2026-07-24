# gpu-util-demo

Small experiments for understanding what “GPU utilization” really means, and why different tools answer different questions.

This repo is organized around several measurement layers:

- **NVML / `nvidia-smi`** for coarse device-level busy-time signals
- **DCGM** for aggregated hardware telemetry such as `SM_ACTIVE` and `SM_OCCUPANCY`
- **Nsight Systems** for CPU–GPU timeline structure
- **Nsight Compute** for kernel-level efficiency and bottleneck analysis
- **CUPTI** for injection-based kernel tracing and hardware counter profiling on unmodified CUDA applications

The repository currently contains the top-level directories `.vscode`, `DCGM`, `cpu_util`, `cupti`, `nsight_compute`, `nsight_systems`, and `nvml`.

## GPU observability tools at a glance

GPU-Util only measures the fraction of time *some* kernel was resident — not how much work the GPU is doing. Each tool below answers a different question, at a different depth:

| Tool | Capabilities | Typical Use Case | Overhead |
|---|---|---|---|
| **nvidia-smi / NVML** | Reports device-level utilization, power consumption, memory usage, and clock speeds | Rapidly determine whether the GPU is active and assess its basic operating state | Negligible |
| **DCGM** | Provides hardware telemetry—including SM activity, occupancy, Tensor Core utilization, and DRAM bandwidth—across large-scale deployments | Continuous cluster monitoring, dashboarding, and automated alerting | Less than 1% |
| **Nsight Systems** | Captures CPU–GPU timelines covering kernel execution, memory transfers, and API calls | Identify performance bottlenecks such as launch gaps, stalls, and unintended serialization | Low; intended primarily for development |
| **Nsight Compute** | Collects detailed hardware performance counters and performs kernel-level roofline analysis | Diagnose kernel performance limitations and guide targeted optimization | High due to replay; intended for development |
| **CUPTI** | Provides the programmable instrumentation interface underlying NVIDIA Nsight tools, including kernel tracing, API interception, and performance-counter collection | Build customized, continuously running production instrumentation without modifying application source code | Varies by the enabled instrumentation |

Top to bottom, this is also the diagnosis path: detect (NVML) → monitor (DCGM) → locate (Nsight Systems) → explain (Nsight Compute) → automate (CUPTI).

## Repository layout

### `cpu_util/`
Contains a simple CPU utilization demo (`cpu_util_demo.py`) used as a conceptual baseline before moving to GPU metrics.  [oai_citation:1‡GitHub](https://github.com/852866031/gpu-util-demo/tree/main/cpu_util)

### `nvml/`
Contains a CUDA workload (`two_cases.cu`), a small NVML Python example (`nvml.py`), and a `Makefile`. This directory is used to demonstrate how coarse GPU-util numbers can hide very different workload structures.  [oai_citation:2‡GitHub](https://github.com/852866031/gpu-util-demo/tree/main/nvml)

### `DCGM/`
Contains `dcgm_trace.py` plus a directory README. This part of the repo shows how to install DCGM and how to trace a real workload while collecting metrics such as `SM_ACTIVE`, `SM_OCCUPANCY`, `TENSOR_ACTIVE`, and `DRAM_ACTIVE`.  [oai_citation:3‡GitHub](https://github.com/852866031/gpu-util-demo/tree/main/DCGM)

### `nsight_compute/`
Contains a two-GPU kernel experiment (`workload_two_gpu.py`), DCGM helpers, a tuner, plotting code, a `Makefile`, and a local README. The goal is to show that similar aggregated GPU activity can still correspond to very different kernel efficiency.  [oai_citation:4‡GitHub](https://github.com/852866031/gpu-util-demo/tree/main/nsight_compute)

### `nsight_systems/`
Contains a two-GPU inference-pipeline example (`inf_sys.py`), NVML/DCGM runners, combined plotting code, a `Makefile`, and a local README. The goal is to show that similar GPU activity does not necessarily imply similar productive or service-level utilization.  [oai_citation:5‡GitHub](https://github.com/852866031/gpu-util-demo/tree/main/nsight_systems)

### `cupti/`
Contains two injection-based CUPTI libraries for profiling unmodified CUDA applications, plus an LLM inference workload (Llama-3-8B) for realistic testing and plotting scripts for visualization.

- **Tracer** (`tracer/`) — a lightweight library that attaches via `CUDA_INJECTION64_PATH` and records every kernel's launch count and GPU execution time into CSV hotspot tables. Overhead is ~1-5%.
- **Auto-Profiler** (`profiler/`) — a more advanced library that cycles between tracing and profiling. It traces kernels to identify the hottest one, then uses CUPTI's Profiler API with hardware counter collection (AutoRange + KernelReplay) to profile that kernel's next launch. The kernel is transparently replayed across multiple passes to read hardware performance counters (SM utilization, occupancy, DRAM throughput). Results are written to JSON files with per-cycle metrics.
- **Plotting** (`plot_all.py`) — generates hotspot charts, overhead comparison across all three modes (baseline / tracer / profiler), and per-cycle profiling analysis with SM utilization breakdown, DRAM traffic, and automated bottleneck diagnosis.

See [cupti/README.md](cupti/README.md) for the full workflow, and [cupti/profiler/README.md](cupti/profiler/README.md) for detailed documentation on the profiler's kernel replay mechanism and metric interpretation.

## Create a conda environment

A minimal environment for the Python parts of this repo:

```
conda create -n gpu-util-demo python=3.11 -y
conda activate gpu-util-demo
pip install torch matplotlib nvidia-ml-py
```

For directories that use custom CUDA kernels from Python, also install CuPy:

```
pip install cupy-cuda13x
```

If your system is on CUDA 12 instead of CUDA 13, replace cupy-cuda13x with cupy-cuda12x.

System tools you may also need

Some subdirectories require NVIDIA system tools in addition to Python packages:
	•	DCGM (dcgmi) for the DCGM/, nsight_compute/, and nsight_systems/ monitoring scripts. The DCGM directory includes Ubuntu 24.04 installation notes.  ￼
	•	Nsight Systems (nsys) for nsight_systems/.  ￼
	•	Nsight Compute (ncu) for nsight_compute/.  ￼

On many systems, Nsight Compute also requires permission to access NVIDIA performance counters.

Suggested workflow

1. Start with the coarse demos
	•	cpu_util/ to ground the CPU-side idea of utilization
	•	nvml/ to see how one coarse GPU-util number can hide very different cases

2. Move to aggregated hardware telemetry
	•	DCGM/ to collect SM_ACTIVE, SM_OCCUPANCY, and related metrics on a real workload

3. Compare kernel efficiency
	•	nsight_compute/ to see why similar aggregated activity does not imply similar kernel productivity

4. Compare pipeline efficiency
	•	nsight_systems/ to see why similar GPU activity does not imply similar throughput or latency at the pipeline level

5. Injection-based profiling on real workloads
	•	cupti/ to trace and profile an LLM inference workload without modifying the application, and collect hardware performance counters for the hottest kernels

Examples

Run the Nsight Systems inference example directly:

```
cd nsight_systems
make run
```

Collect NVML and DCGM measurements for that example:

```
make nvml
make dcgm
make plot-all
```

Run the two-kernel Nsight Compute example:

```
cd ../nsight_compute
make run
make plot
make ncu-gather
make ncu-dense
```

Run the CUPTI tracing and profiling workflow on an LLM workload:

```
cd ../cupti
make llm-trace        # trace kernel hotspots
make llm-profile      # auto-profile the hottest kernel (requires sudo)
make plot             # generate hotspot, overhead, and profiling charts
```