# Tally Reproduction Report

**Paper:** [Tally: Non-Intrusive Performance Isolation for Concurrent Deep Learning Workloads](https://arxiv.org/abs/2410.07381) (ASPLOS '25, Zhao, Jayarajan, Pekhimenko)
**Code:** [tally-project/tally](https://github.com/tally-project/tally) (MIT) + [tally-project/tally-bench](https://github.com/tally-project/tally-bench)
**Date:** 2026-08-11
**Verdict:** ✅ Reproduced end-to-end on hardware two GPU generations newer than the paper's, after three source/config fixes. A micro-benchmark qualitatively confirms the paper's headline claim: co-located HP p99 latency improves **8.4×** over the hardware scheduler with **no BE throughput loss**.

## 1. Environment

| Item | Value |
|---|---|
| Host GPU | 2× NVIDIA GeForce RTX 5090 (Blackwell, **sm_120**, 32 GB); experiment uses device 0 |
| Driver | 580.126.09 (CUDA 13.0) |
| Host CUDA toolkit | 12.8 (`/usr/local/cuda-12.8`) |
| Host OS | Ubuntu 24.04, kernel 6.14 |
| Container | `wzhao18/tally:latest` — Ubuntu 20.04, CUDA 12.2, PyTorch 2.2.0a0 (source build), Tally prebuilt at `/home/tally-bench/tally` |

The paper evaluates on A100 (sm_80). Everything below is about bridging the two-generation gap to Blackwell.

## 2. Why the prebuilt container instead of a native build

Three reasons the native path was rejected:

1. **CMake arch-detection bug on Blackwell.** Both the top-level `CMakeLists.txt` and `tests/CMakeLists.txt` do
   `string(SUBSTRING ${INSTALLED_GPU_CCS_1} 0 2 CUDA_ARCH_LIST)` — compute capability
   "120" gets truncated to the invalid arch "12".
2. **Heavy, version-sensitive dependency chain**: folly, Boost (serialization/regex/stacktrace_backtrace), gflags, iceoryx, and g++-10-era C++20 code — high friction on Ubuntu 24.04 / GCC 13.
3. **Pinned CUTLASS submodule predates Blackwell**, so `tally_cutlass` would likely not compile for sm_120 anyway.

Docker Hub tags for `wzhao18/tally`: `latest` (16.4 GB compressed — Tally + toolchain + PyTorch), `base` (16.5 GB — deps only), `bench` (87.8 GB — adds all paper workloads). `latest` was used; `bench` is the one the official `tally-bench` instructions assume.

Container launch (host CUDA mounted in — see fix #2):

```bash
docker run -d --name tally-repro --gpus all --shm-size=16g \
  -v $PWD/reproduce/scripts:/workspace/scripts \
  -v /usr/local/cuda-12.8:/host-cuda:ro \
  wzhao18/tally:latest tail -f /dev/null
```

Note: the repo's own `Dockerfile` is stale — it downloads Boost from the dead
`boostorg.jfrog.io` mirror, expects a cuDNN tarball not in the repo, and doesn't build
folly — another reason to prefer the prebuilt image.

## 3. Tally's run model (for orientation)

Tally is a client/server CUDA virtualization layer:

1. **`iox-roudi`** — iceoryx shared-memory transport daemon.
2. **`tally_server`** — owns the GPU; scheduler chosen by `SCHEDULER_POLICY` ∈
   `NAIVE` | `PRIORITY` | `PROFILE` | `WORKLOAD_AGNOSTIC_SHARING` | `WORKLOAD_AWARE_SHARING`.
3. **Clients** — unmodified applications launched with `LD_PRELOAD=libtally_client.so`
   (via `scripts/start_client.sh`). Each client carries a `PRIORITY` env var; **higher
   value = higher priority** (server keeps `client_priority_map` sorted descending).

On first sight of a client fatbin, the server extracts PTX (`cuobjdump`), applies its
transformations (kernel slicing / persistent-thread-block conversion for the priority
scheduler), recompiles with `nvcc`, and caches the result in `~/.cache/tally/`.

## 4. Issues found and fixes applied

### Fix 1 — hardcoded compute-capability list (server abort)

**Symptom:** first client connection kills the server:

```
[warning] Fail to extract elf file from /tmp/tmp_99.cubin. Tried compute capabilities:
terminate called after throwing an instance of 'std::runtime_error'
  what():  get_kernel_names_and_param_sizes_from_elf file not found
```

**Root cause:** `src/tally/cuda_util.cpp:16` hardcodes
`CUDA_COMPUTE_CAPABILITIES = {"90", "86", "80"}`. `get_candidate_cuda_compute_capabilities()`
does `std::find` for the runtime-detected capability ("120") and returns the suffix of the
list from that position — not found ⇒ **empty candidate list** ⇒ no ELF/PTX can be
extracted from any client binary.

**Fix:** change the list to `{"120", "90", "89", "86", "80"}` and rebuild
(`cd /home/tally-bench/tally/build && make -j`). The extraction loop in
`include/tally/cache_util.h` tries candidates in order and falls back, so on sm_120 it
finds no sm_120 image in the (sm_80-era) fatbins and correctly falls back to the embedded
sm_80 ELF/PTX (`[warning] Fall back to use PTX code for compute capability 80`).

### Fix 2 — runtime PTX recompilation needs an sm_120-capable nvcc

**Root cause:** `get_fatbin_str_from_ptx_str()` (`src/tally/cuda_util.cpp`) shells out to
`nvcc --fatbin -gencode arch=compute_<cap>,code=sm_<cap>` where `<cap>` is the
**runtime-detected** capability — `120` here. The container's CUDA 12.2 nvcc predates
Blackwell and rejects `compute_120`, which would make every kernel transformation fail
("Fail to compile PTX.").

**Fix:** no code change — mount the host's CUDA 12.8 into the container
(`-v /usr/local/cuda-12.8:/host-cuda:ro`) and prepend `/host-cuda/bin` to `PATH` for the
server process. Transformed PTX then compiles to real sm_120 SASS. Tally's own prebuilt
libraries (compiled for sm_80 with embedded PTX) load fine on sm_120 via driver JIT, as
does the container's PyTorch (~8 s one-time JIT on first kernel, then cached).

### Fix 3 — iceoryx mempool config not loaded (client allocation failure)

**Symptom:**

```
ICEORYX error! MEPOO__MEMPOOL_GETCHUNK_CHUNK_IS_TOO_LARGE
Could not allocate Request: AllocationError::NO_MEMPOOLS_AVAILABLE
```

**Root cause:** `scripts/start_iox.sh` launches `iox-roudi` with no arguments, so iceoryx
falls back to its built-in mempool config (max chunk 4 MB). Tally's client sends ~67 MB
messages (fatbin registration). The repo ships the right config —
`config/roudi_config.toml`, with 1 GB×2 and 3 GB×2 pools — but never passes it.

**Fix:** launch as `iox-roudi -c $TALLY_HOME/config/roudi_config.toml`. The pools need
~8.5 GB of shared memory ⇒ run the container with `--shm-size=16g` (default 64 MB is far
too small).

### Fix 4 — `cuGetExportTable` abort under newer drivers (PyTorch segfault)

**Symptom:** every PyTorch client dies at `torch.cuda` initialization:

```
[warning] cuGetExportTable is a internal CUDA call. Must override from higher-level API.
 0# cuGetExportTable in libtally_client.so
 1# ... libnvidia-ml.so.1 ... nvmlInitWithFlags ... ctypes ... torch::utils::cuda_lazy_init()
Segmentation fault (core dumped)
```

**Root cause:** Tally's interposed `cuGetExportTable`
(`src/tally/preload/tally_client.cpp`) deliberately throws — export tables are an opaque
side-channel that would bypass its virtualization. That was sound on the driver
generation the authors used, but **driver 580's `libnvidia-ml` internally calls
`cuGetExportTable` during `nvmlInit`**, which PyTorch reaches via ctypes during
`torch.cuda` lazy init. The intercepted call originates from NVML (GPU monitoring), not
from the application's CUDA path — Tally intercepts that at the runtime-API layer.

**Fix:** make the stub pass the call through to the real driver
(`return lcuGetExportTable(...)` — the passthrough already existed for local mode) with a
warning instead of throwing. This is safe for NVML's monitoring queries; it does not
route any kernel launches around the scheduler.

Both source patches are in [results/tally_sm120_fixes.patch](results/tally_sm120_fixes.patch)
(applied both to the local clone at `reproduce/tally/` and inside the container, then rebuilt).

## 5. Verification steps

1. **Smoke test** ([scripts/tally_smoke_test.sh](scripts/tally_smoke_test.sh)):
   `iox-roudi` + `tally_server` (NAIVE) + the repo's `elementwise` CUDA test client
   through `LD_PRELOAD`. Passes: server extracts the fatbin, falls back to sm_80 PTX,
   transforms it, recompiles for sm_120, dispatches; client prints
   `Kernel execution time: 0.12 ms` and exits 0.
2. **Native PyTorch sanity** (no Tally): 1024² matmul on GPU works; first call ~7.8 s
   (driver JIT of the sm_80-built PyTorch), fast afterward.
3. **PyTorch through Tally** (NAIVE): initially segfaulted → fix 4 → runs cleanly.
4. **Co-location experiment** — below.

## 6. Co-location experiment

Script: [scripts/tally_colocate_test.sh](scripts/tally_colocate_test.sh); full log:
[results/tally_colocate_2026-08-11.log](results/tally_colocate_2026-08-11.log).

**Workloads** (single GPU, device 0):
- **HP client** ([scripts/hp_client.py](scripts/hp_client.py)) — simulated online
  inference: `relu(x@W1)@W2` MLP step (64×512×1024), synchronized per request, 10 ms
  think time between requests (creates the idle gaps Tally harvests), 200 requests,
  reports p50/p99/avg latency. Runs with `PRIORITY=2`.
- **BE client** ([scripts/be_client.py](scripts/be_client.py)) — throughput job:
  back-to-back 4096×4096×4096 matmuls (~6.5 ms each), reports it/s. Runs with `PRIORITY=1`.

**Protocol:** four phases — HP alone native → HP+BE native (hardware scheduler, both
processes directly on the GPU) → HP alone under Tally → HP+BE under Tally
(`SCHEDULER_POLICY=PRIORITY`). In co-located phases the BE job warms up 10–15 s before
HP measurement starts, and runs throughout.

### Results

| Configuration | HP p50 | HP p99 | HP avg | BE throughput |
|---|---|---|---|---|
| HP alone, native | 0.053 ms | 0.142 ms | 0.059 ms | — |
| HP + BE, native (hardware scheduler) | 0.231 ms | **2.103 ms** | 0.780 ms | 154.65 it/s |
| HP alone, Tally PRIORITY | 0.056 ms | 0.099 ms | 0.057 ms | — |
| HP + BE, Tally PRIORITY | 0.233 ms | **0.250 ms** | 0.229 ms | 154.57 it/s |

**Findings:**

- **Tail-latency isolation reproduced.** Under the hardware scheduler, co-location
  inflates HP p99 by ~15× (0.142 → 2.103 ms). Tally's priority scheduler holds p99 at
  0.250 ms — an **8.4× improvement** over uncontrolled sharing — and makes latency far
  more predictable (avg ≈ p50 ≈ p99).
- **BE throughput preserved** (154.6 vs 154.7 it/s). Consistent with the paper's claim
  of retaining >80% of aggregate throughput; here the HP job is light enough that BE
  loses essentially nothing.
- **Negligible mechanism overhead**: HP-alone latency through Tally's entire
  intercept/IPC/dispatch pipeline matches native (0.056 vs 0.053 ms p50).
- The server log confirms real work: `Running priority scheduler ...` and 422
  transformation-related entries (`Generating transformed code for ... sm_80.ptx`).

### Caveats

- This is a **micro-benchmark**, not the paper's evaluation. The official suite
  (`tally-bench`: BERT/ResNet/Whisper etc. co-location grids) assumes the 88 GB `bench`
  image and an A100-40GB; not attempted here.
- The BE job's matmuls dispatch to **cuBLAS library kernels**, which Tally flags
  (`Found library call from low priority job`) and schedules but cannot transform into
  preemptible form. The measured isolation therefore comes mostly from priority
  launch-gating between BE kernels (each ~6.5 ms, synchronized). Running the BE client
  with `REPLACE_CUBLAS=TRUE` (Tally's cuBLAS→CUTLASS substitution) would exercise the
  persistent-thread-block preemption path fully — a natural follow-up.
- Results are from Blackwell hardware and CUDA versions the authors never tested;
  absolute numbers are not comparable to the paper's A100 figures, only the qualitative
  behavior.
- The p99 of a 200-sample run is the ~2nd-worst sample; run-to-run variation of the
  native co-located p99 was small in practice (2.05 / 2.10 ms across two runs).

## 7. How to re-run

```bash
# one-time: container (see §2), then apply results/tally_sm120_fixes.patch inside
# /home/tally-bench/tally and `make -j` in its build/ dir
docker exec tally-repro bash /workspace/scripts/tally_smoke_test.sh      # single-client sanity
docker exec tally-repro bash /workspace/scripts/tally_colocate_test.sh   # 4-phase experiment
```

## 8. Summary of the artifact's quality

The code matches the paper's architecture (virtualization server, per-kernel
slicing/persistent-worker transformation, priority scheduler) and works, but it is
research-grade: GPU support is a hardcoded list, the Dockerfile has bit-rotted,
`start_iox.sh` omits the repo's own required config, and driver evolution broke the
`cuGetExportTable` assumption. None of these required deep changes — every fix was
small and local — which speaks well of the codebase's structure. Total effort from
clone to reproduced result: one session, four fixes.
