# Reproduction Exploration: Public Code Availability

**Goal of this directory:** reproduce and run the systems from the papers covered in
[LITERATURE_REVIEW.md](../LITERATURE_REVIEW.md) (fine-grained GPU sharing and co-location
scheduling). This file records the results of a web search (2026-08-11) for public code
repositories for each paper.

## Summary Table

| Paper | Venue | Public code? | Repository | Notes |
|---|---|---|---|---|
| **Tally** — Non-Intrusive Performance Isolation for Concurrent DL Workloads | ASPLOS '25 | ✅ Yes (official) | [tally-project/tally](https://github.com/tally-project/tally) + [tally-project/tally-bench](https://github.com/tally-project/tally-bench) | Best reproduction target. MIT license, C++/CMake, Dockerfile included; `tally-bench` has scripts to reproduce the paper's results. |
| **Hummingbird** — SLO-Oriented GPU Preemption at Microsecond-scale | arXiv 2601.04071 (Jan 2026) | ❌ No | — | ~8,000 lines C++/CUDA per the paper, but no code link in the paper (v1/v2 full text checked) and no repo found. |
| **Bless** — Adaptive Bubbleless Spatial-Temporal GPU Sharing | EuroSys '25 | ❌ No | — | ~5,000 lines C++ per the paper. No repo in search results; first author Shulai Zhang's homepage (shulai.org) lists the paper but no code release. |
| **LithOS** — An Operating System for Efficient ML on GPUs | SOSP '25 | ❌ No | — | Rust prototype (CMU/Meta). Not in the SOSP '25 artifact-evaluation results list; no repo found. Its TPC-mask mechanism is deferred to an unpublished technical report, so full reproduction is likely infeasible anyway. |
| **MMK** — A Hybrid Scheduling Framework for Fine-Grained GPU Sharing | ACM TACO 2026 (DOI 10.1145/3820163) | ❌ No | — | ~3,000 lines C++/Python per the paper; no artifact or repo found. |
| **SMore** — Serverless-Based Co-Location Scheduling | IEEE TPDS 2025 | ❌ No | — | Journal paper (U. Melbourne CLOUDS lab / Buyya group); no repo found. |
| **Usher** — Holistic Interference Avoidance for Resource Optimized ML Inference | OSDI '24 | ⚠️ Partial | [ss7krd/Usher](https://github.com/ss7krd/Usher) | Author's repo, explicitly "preliminary codes": model definitions (TF 2.4.1, CUDA 10.1, Python 3.7), resource-estimation scripts (`sm_requirement.py`, `memory_requirement.py`), scheduling policies. No license, only 8 commits, sparse documentation — usable as reference, not turnkey. |
| **Orion** (reference baseline throughout the review) | EuroSys '24 | ✅ Yes (official) | [eth-easl/orion](https://github.com/eth-easl/orion) | ETH Zurich EASL. ~3,000 lines C++/CUDA integrated with PyTorch; has `INSTALL.md`, `PROFILE.md`, and artifact-evaluation docs. |

## Per-Paper Details

### Tally (ASPLOS '25) — best candidate to run
- Main repo: <https://github.com/tally-project/tally> — the CUDA-virtualization client/server,
  kernel slicing, and persistent-worker preemption ("priority scheduler"). MIT license,
  CMake build, Dockerfile, ~296 commits.
- Benchmarks: <https://github.com/tally-project/tally-bench> — scripts to reproduce the
  paper's HP latency / BE throughput experiments.
- Paper: <https://arxiv.org/abs/2410.07381>, DOI: <https://doi.org/10.1145/3669940.3707282>
- Expect to need a Linux box with an NVIDIA GPU + recent CUDA; it intercepts CUDA calls via
  its own transport layer, so containerized runs are the intended path.

### Hummingbird (arXiv 2601.04071)
- Paper: <https://arxiv.org/abs/2601.04071> (Hu, Wang, Cao, et al., Jan 2026).
- No code release: the full text (both HTML versions) mentions an ~8k-line C++/CUDA
  implementation but includes no repository URL, and web searches surface no repo.
- Reproduction options: contact authors, or re-implement the core ideas (PTX `blockIdx`-offset
  split kernels + kernel-tick launch loop) — Tally's slicing code is the closest open
  starting point.

### Bless (EuroSys '25)
- Paper PDF: <https://jamesthez.github.io/files/bless-eurosys25.pdf>,
  DOI: <https://dl.acm.org/doi/10.1145/3689031.3696070>
- No public repo found. First author Shulai Zhang (SJTU) lists the paper on
  <https://www.shulai.org/> without a code link.
- Partial reproduction is feasible with standard tooling: the key mechanism (pre-created
  `cuCtxCreate_v3` contexts with SM-affinity + MPS) uses only public CUDA APIs.

### LithOS (SOSP '25)
- Paper: <https://arxiv.org/abs/2504.15465>, DOI: <https://doi.org/10.1145/3731569.3764818>,
  slides: <https://www.eliot.so/sosp25-slides.pdf>
- No public repo; LithOS does not appear in the
  [SOSP '25 artifact evaluation results](https://sysartifacts.github.io/sosp2025/results).
- Caveat for reproduction: the per-launch TPC-mask injection relies on undocumented
  NVIDIA launch metadata (deferred to a separate tech report), so even a re-implementation
  would be blocked on that mechanism. The Prelude-based kernel atomizer, however, is
  reproducible with public APIs (wrapper kernel + block-range filtering).

### MMK (ACM TACO 2026)
- DOI: 10.1145/3820163 (ACM TACO; Jiang, Cai, Li, Wang, Ma, Guan, Buyya).
- No artifact or repo found. The building blocks (MIG partitioning, MPS percentages,
  `LD_PRELOAD` CUDA interception, random-forest predictor) are all standard, but the
  hierarchical policy would need re-implementation. Requires an A100/H100-class GPU for MIG.

### SMore (IEEE TPDS 2025)
- Paper: <https://ieeexplore.ieee.org/document/10912752/>, open PDF:
  <https://clouds.cis.unimelb.edu.au/papers/GPU-DLClusters.pdf>
- No repo found. It is a cluster-level scheduler (degradation predictor + admission +
  LS-LSTM prewarming); reproducing it requires a multi-GPU serverless testbed.

### Usher (OSDI '24)
- Paper: <https://www.usenix.org/conference/osdi24/presentation/shubha>
- Repo: <https://github.com/ss7krd/Usher> — preliminary code only (no license, 8 commits).
  Pinned to an old stack: Python 3.7, TensorFlow 2.4.1, CUDA 10.1, cuDNN 7.6.5.
  Contains the GK-estimator-style resource scripts and scheduling policies but is not a
  runnable end-to-end system.

### Orion (EuroSys '24) — baseline used across the review
- Repo: <https://github.com/eth-easl/orion> (official, ETH Zurich EASL), paper:
  <https://anakli.inf.ethz.ch/papers/orion_eurosys24.pdf>
- Includes install and profiling docs; designed around PyTorch and NVIDIA GPUs
  (artifact originally evaluated on V100/A100-era stack).

## Recommended Reproduction Plan

1. **Tally** — clone `tally-project/tally` + `tally-bench`, build via the provided Docker
   image, and run the HP/BE co-location benchmarks. This is the only fully open,
   artifact-backed system among the five core kernel-scheduling papers.
2. **Orion** — set up `eth-easl/orion` as the comparison baseline (the review repeatedly
   contrasts every system against it).
3. **Usher** — mine `ss7krd/Usher` for the resource-estimator logic if the workload-placement
   layer becomes relevant; do not expect it to run unmodified on a modern CUDA stack.
4. **Bless / LithOS mechanisms** — if needed, re-implement the reproducible pieces
   (SM-affinity contexts for Bless; Prelude-wrapper atomization for LithOS) as
   micro-prototypes; full-system reproduction is not possible without author code.
5. **Hummingbird / MMK / SMore** — no code available; consider emailing the authors
   (Hummingbird and MMK are recent enough that code may be released later).

## Tally Reproduction

Tally has been reproduced and run on this machine (2× RTX 5090). See the dedicated
report: [tally-reproduce.md](tally-reproduce.md) — environment, the four fixes needed
for Blackwell/driver-580 compatibility, the co-location experiment, and results
(HP p99 improved 8.4× over the hardware scheduler with no BE throughput loss).

## Search Notes / Sources

- Searched 2026-08-11 via web search (queries per paper: title + "github"/"code"/"artifact")
  plus targeted checks of the arXiv full texts (Hummingbird), author homepages (Bless),
  and the SOSP '25 artifact-evaluation results (LithOS).
- Key sources: [arXiv:2601.04071](https://arxiv.org/abs/2601.04071),
  [tally-project/tally](https://github.com/tally-project/tally),
  [tally-project/tally-bench](https://github.com/tally-project/tally-bench),
  [Bless PDF](https://jamesthez.github.io/files/bless-eurosys25.pdf),
  [arXiv:2504.15465](https://arxiv.org/abs/2504.15465),
  [SOSP '25 AE results](https://sysartifacts.github.io/sosp2025/results),
  [SMore @ IEEE](https://ieeexplore.ieee.org/document/10912752/),
  [Usher @ USENIX](https://www.usenix.org/conference/osdi24/presentation/shubha),
  [ss7krd/Usher](https://github.com/ss7krd/Usher),
  [eth-easl/orion](https://github.com/eth-easl/orion).
