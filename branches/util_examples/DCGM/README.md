# Install DCGM on Ubuntu 24.04

This guide shows how to install **NVIDIA DCGM (Data Center GPU Manager)** on **Ubuntu 24.04** using NVIDIA’s APT repository. DCGM is NVIDIA’s framework for GPU management and telemetry in workstation, server, and datacenter environments. It is typically used through the `nvidia-dcgm` service and the `dcgmi` CLI.  [oai_citation:0‡NVIDIA Docs](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html?utm_source=chatgpt.com)

## 1. Check your Ubuntu version

Confirm that the system is running Ubuntu 24.04:

```bash
. /etc/os-release
echo "$ID $VERSION_ID"

Expected output:

ubuntu 24.04

2. Enable the NVIDIA CUDA APT repository

NVIDIA’s Ubuntu installation guide recommends installing the cuda-keyring package, then running apt update. For Ubuntu 24.04, use the ubuntu2404 repository.  ￼

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

3. Check which DCGM packages are available

After enabling the repository, list the available DCGM packages:

apt-cache search datacenter-gpu-manager-4

On Ubuntu 24.04 with the NVIDIA repo enabled, you may see packages such as:
	•	datacenter-gpu-manager-4-cuda11
	•	datacenter-gpu-manager-4-cuda12
	•	datacenter-gpu-manager-4-cuda13
	•	datacenter-gpu-manager-4-cuda-all  ￼

4. Choose the package that matches your CUDA user-mode driver major version

NVIDIA’s DCGM getting-started guide recommends installing the package that matches the CUDA user-mode driver major version, which you can check with nvidia-smi -q.  ￼

Check the CUDA version:

nvidia-smi -q | grep -E 'Driver Version|CUDA Version'

If your system reports CUDA Version: 13.x, install:

sudo apt-get install -y --install-recommends datacenter-gpu-manager-4-cuda13

If your system reports CUDA Version: 12.x, install:

sudo apt-get install -y --install-recommends datacenter-gpu-manager-4-cuda12

5. Enable and start the DCGM service

After installation, enable and start the DCGM service:

sudo systemctl --now enable nvidia-dcgm
systemctl status nvidia-dcgm --no-pager

DCGM normally runs through the nv-hostengine daemon, while client tools such as dcgmi connect to it.  ￼

6. Verify that DCGM can see your GPUs

List the GPUs discovered by DCGM:

dcgmi discovery -l

If installation was successful, your GPUs should appear in the output.  ￼

7. Basic post-install checks

List the profiling metrics supported on GPU 0:

dcgmi profile -l -i 0

Monitor a few metrics in real time on GPU 0:

dcgmi dmon -e 1001,1002,1003,1004,1005 -i 0

8. Trace a real workload with dcgm_trace.py

This project also includes a Python script, DCGM/dcgm_trace.py, which automates a complete DCGM monitoring demo.

What dcgm_trace.py does

The script performs the following steps:
	1.	Checks that DCGM is running
	•	Verifies that the nvidia-dcgm service or nv-hostengine process is active.
	2.	Compiles the CUDA workload
	•	Calls the Makefile in ../nvml/ to compile the two_cases CUDA program.
	3.	Launches DCGM monitoring
	•	Starts dcgmi dmon and continuously samples the following profiling metrics for GPU 0 and GPU 1:
	•	1002 = SM_ACTIVE
	•	1003 = SM_OCCUPANCY
	•	1004 = TENSOR_ACTIVE
	•	1005 = DRAM_ACTIVE
	4.	Runs the workload
	•	Launches the compiled two_cases binary through make run.
	5.	Collects and saves the data
	•	Saves the raw DCGM output
	•	Parses the metric stream into a CSV file
	•	Computes average values for each GPU
	6.	Generates a plot
	•	Produces one figure with four subplots:
	•	SM_ACTIVE
	•	SM_OCCUPANCY
	•	TENSOR_ACTIVE
	•	DRAM_ACTIVE
	•	Each subplot includes:
	•	one curve for GPU 0
	•	one curve for GPU 1
	•	average values shown in the plot

⸻

9. File layout expected by dcgm_trace.py

The script assumes a directory structure like this:

util_examples/
├── DCGM/
│   ├── dcgm_trace.py
│   └── README.md
└── nvml/
    ├── Makefile
    ├── two_cases.cu
    └── ...

In particular:
	•	dcgm_trace.py must be inside the DCGM/ directory
	•	the CUDA workload must live in ../nvml/
	•	the nvml/Makefile must provide:
	•	compile
	•	run

⸻

10. How to run dcgm_trace.py

From the project root:

cd DCGM
python dcgm_trace.py

Or from the top-level project directory:

python DCGM/dcgm_trace.py


⸻

11. Output files

The script creates an outputs/ directory inside DCGM/ and writes:
	•	dcgm_dmon_raw.log
Raw text output from dcgmi dmon
	•	dcgm_metrics.csv
Parsed metric values with timestamps
	•	dcgm_metrics_subplots.png
A figure containing four subplots for:
	•	SM_ACTIVE
	•	SM_OCCUPANCY
	•	TENSOR_ACTIVE
	•	DRAM_ACTIVE

Example output structure:

DCGM/
├── dcgm_trace.py
├── README.md
└── outputs/
    ├── dcgm_dmon_raw.log
    ├── dcgm_metrics.csv
    └── dcgm_metrics_subplots.png


⸻

12. What the resulting plot means

The generated figure helps compare the two GPUs over time:
	•	SM_ACTIVE shows how active the SMs are over the sampling interval
	•	SM_OCCUPANCY shows how densely SMs are populated with resident warps
	•	TENSOR_ACTIVE shows Tensor Core activity
	•	DRAM_ACTIVE shows device-memory activity

Since the workload runs different launch configurations on GPU 0 and GPU 1, the plot provides a hardware-level view of how the two execution patterns differ, even when a single utilization number such as nvidia-smi may look similar.