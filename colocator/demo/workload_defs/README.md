# workload_defs/ — one workload per file

Every non-underscore `.py` file here (except `base.py`) defines one workload
and exports it as the module-level name **`WORKLOAD`**; the registry
[../workloads.py](../workloads.py) auto-discovers them all. Add a workload =
add a file.

## Contract (`base.py` — `WorkloadBase`)

| method | role |
|---|---|
| `setup()` | allocate host/device state |
| `launch_iter()` | launch ONE iteration's GPU work, **no sync**; return the result buffer |
| `run_iter()` | (base) pipelined admission: waits for the iteration from **2 launches ago** (torch.cuda.Event), then calls `launch_iter()` — so **2 iterations are always in flight**, solo and colocated alike |
| `drain()` | (base) wait for everything in flight (loop ends, barriers) |
| `calibrate()` | pick `iters` for a ~**500 ms** timed section, measured at pipelined steady state |
| `check()` | one verified iteration via `run_iter_sync()` → bool |
| `describe()` | **HTML fragment** shown in the replay's description boxes: `summary()` + `params_html()` (kernel geometry, sizes) + pipeline + calibration info |
| `colocatable` | class attr, default `True`. Set `False` when the workload launches kernels the interceptor cannot see — `run_demo --mode colocated` refuses it and `run_pairs` excludes it from the default matrix |

The pipeline works under the colocator because `cudaEventRecord` **and**
`cudaEventRecordWithFlags` (the one torch actually calls) are interposed and
redirected onto the client's colocator stream; `event.synchronize()` then
waits on the real event. Iteration wall times measure *admission* at steady
state (≈ per-iteration GPU time), not end-to-end latency of one iteration.
Workloads that copy results out use one pinned buffer **per pipeline slot**
(`self._slot`) so two in-flight iterations never race on the same memory.

## The workloads

Two application-style clients (tiled-SGEMM extension, H2D → compute → D2H):

| file | name | character |
|---|---|---|
| `latency.py` | `latency` | 4× (SGEMM 256×1024×1024 → relu) chain + reduce; many ~µs–80 µs kernels, ~0.45 ms/iter |
| `throughput.py` | `throughput` | one 4096³ SGEMM (fills all SMs, ~16 ms) + 64 MiB H2D per iter |

Six resource probes ported from the **gpu-interfere profiler**
(`~/Documents/Projects/gpu-interfere/profiler/code/probe.cu`, kernels rebuilt
in [../../ext/csrc/probes.cu](../../ext/csrc/probes.cu)); each launches one
kernel per iteration, **one block per SM**, inner-loop sized for a
few–tens of ms per launch:

| file | name | saturates | kernel |
|---|---|---|---|
| `sleep.py` | `sleep` | SM residency only (no mem/compute) | `k_sleep`: nanosleep loop, 768 thr |
| `dram.py` | `dram` | DRAM bandwidth | `k_copy` over 512 MiB (≫ L2), 512 thr |
| `l2.py` | `l2` | L2 bandwidth | `k_copy` over 16 MiB (L2-resident), 512 thr |
| `l1.py` | `l1` | L1 / per-SM working set | `k_copy_tb`: 32 KiB region per block, 64 thr |
| `fma.py` | `fma` | FP32 FMA pipe + issue slots | `k_fma32`: 4 indep. FMA chains, 128 thr |
| `fp64.py` | `fp64` | FP64 pipe (1/64 rate on GeForce) | `k_fma64`, 128 thr |

`_probe_common.py` holds their shared base (`ProbeBase`); underscore files are
not registered as workloads.

One real-model client (needs `transformers`, weights from the HF cache):

| file | name | character |
|---|---|---|
| `llmdecode.py` | `llmdecode` | Llama-3-8B (bf16) **decode steps**: each iteration = one decode forward pass generating tokens 65/66/67 over a 64→66-entry KV cache. The 63-token prefill runs **once per machine** — its state (KV tensors + pending token, deterministic from seed+weights) persists in `~/.cache/colocator/`; later runs load it and execute **pure decode** (no prefill kernels on the timeline; delete the file or bump `SNAP_VERSION` to re-prefill). Default **3 iterations** (the workload is the 3 passes — no 500 ms calibration); `drain()` restores the in-memory cache snapshot so every loop replays the same steps; greedy argmax feeds back GPU-side (no host sync), so the depth-2 pipeline applies unchanged. `check()` = greedy decode twice from the restored snapshot → same token (also validates snapshot determinism across runs). |

**`llmdecode` is colocatable** — via cuBLAS *API* interception: its GEMVs
never reach the kernel-level interposer (cuBLAS statically links the CUDA
runtime; some paths use the driver API — measured: CUPTI sees 11 819
kernels in a solo run, the interposer only 9 689), so `libcolocator`
captures the `cublasGemmEx` / `cublasSgemm_v2` **library calls** instead and
the scheduler replays them with the stream substituted on the handle (see
`src/intercept/README.md`). Colocated `check()` (greedy decode decoded
twice must yield the same token) passes, submitted == issued, and analyze's
leak check reports zero client-launched kernels. Setup details: weights are
device-synced before prefill (transformers loads them from passthrough
worker threads), and `marks["load_done"]` lets the visualizer trim the
~1.3 s weight upload from timelines.

## Checks

Application clients verify against CPU float64 references; copy probes verify
`out == in` (GPU compare); FMA probes check finiteness (accumulating one value
20 M times in fp32 can't be compared numerically); `sleep` has nothing to
verify.
