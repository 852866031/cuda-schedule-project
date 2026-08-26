# VRAM Limit Study — Test Plan


> **This is the design as written before running anything**, kept as a record of what was
> predicted. Several predictions in it turned out wrong — where the study says otherwise,
> the results document is correct. See [reports/report_simple_inference.md](reports/report_simple_inference.md).

**Question:** an LLM serving app whose working set is ~40 GB is run on a 32 GB GPU, with
DRAM absorbing the overflow. How does serving performance degrade as we shrink the VRAM
budget the app is allowed to use?

**Case A (priority).** Weights stay resident; the *KV cache* is oversubscribed. Sweep the
VRAM budget from ~the whole GPU down to "model + 2 GB" and measure the cost of pushing an
ever-larger share of the KV working set into DRAM.

**Case B (later).** The *model* no longer fits. Sweep how much of the weights are offloaded
and measure the per-forward-pass streaming cost.

---

## 1. Testbed

### Measured hardware facts

| | value | how it constrains the design |
|---|---|---|
| GPU | 2× RTX 5090, 32607 MiB (**31.84 GiB**) each | use **GPU 0 only**; GPU 1 stays idle |
| **PCIe** | **Gen4 ×8** (`Host Max: 4`, width `8x`) | **~16 GB/s theoretical, ~12–13 GB/s real.** The GPUs support Gen5 ×16 but the host does not. This is *the* dominant cost term — measure it, don't assume it. |
| DRAM | 60 GiB total, ~55 GiB available | CPU KV pool capped at 24 GiB, pinned. Leaves ~30 GiB headroom. |
| Disk | 113 GiB free | no model downloads needed |
| vLLM | `0.15.0rc2.dev23+g5d3d6e44e` (source, cu128), V1 engine | already has every knob we need |
| torch | `2.8.0.dev20250322+cu128` (nightly) | ABI risk for external compiled pkgs — see §6 |
| nvcc | 12.8 | source-build fallback available |

### Model — Llama-3-8B-Instruct fp16 (already cached, no download)

| quantity | value |
|---|---|
| weights | 8.03 B × 2 B = **14.96 GiB** |
| KV per token | 32 layers × 8 KV heads × 128 dim × 2 (K,V) × 2 B = 131072 B = **0.125 MiB** |
| 1 GiB of KV | 8,192 tokens |
| **KV working set (target)** | **24 GiB = 196,608 tokens** |
| **total footprint** | 14.96 + 24 ≈ **39 GiB (~42 GB decimal)** ✓ the "40 GB app" |

### Mechanism

`--kv-offloading-size N --kv-offloading-backend native` (in-tree `OffloadingConnector` +
`CPUBackend`, LRU/ARC eviction, block-granular H2D/D2H on dedicated CUDA streams, keyed by
the same block hashes prefix caching uses).

> **Semantics that shape the whole design:** this is a *second-level prefix cache*, not
> paging. An actively decoding sequence still needs its blocks in VRAM. Offloading rescues
> **reuse across requests**; it does not let a single request's KV exceed the GPU. The
> workload must therefore be built around reuse, or there is nothing to measure.

LMCache is deferred to Phase 4 (§5) — its extra tiers (disk, Redis/Mooncake, CacheBlend,
compression) are confounds for a pure DRAM↔VRAM measurement.

---

## 2. Phase 0 — Calibration (do this before any sweep)

Establishes the physical constants that every later number is interpreted against.

1. **PCIe bandwidth**: pinned-memory H2D/D2H microbenchmark at 1/4/16/64/256 MiB transfer
   sizes, both directions, plus bidirectional. Confirm the link ramps Gen1→Gen4 under load
   (`nvidia-smi -q` during the run).
2. **Derive the offload cost model**:
   - `t_load(prefix) = prefix_tokens × 0.125 MiB / BW_h2d`
   - at 12 GB/s: an **8192-token prefix = 1 GiB = ~85 ms** (measured: 14.47 GB/s → 74 ms)
3. **Measure recompute cost**: prefill latency for the same 8192-token prefix (one request,
   empty cache, `--kv-offloading-size` unset).
   Expected ~0.9–1.3 s ⇒ **offload should be ~10–15× cheaper than recompute per hit.**

**This is the crux of the experiment.** The whole Case A curve is a race between
`t_load` (paid on a DRAM hit) and `t_prefill` (paid on a full miss). Phase 0 predicts the
curve; Phases 1–2 test the prediction.

Deliverable: `results/calibration.json` + the two constants quoted in every plot caption.

---

## 3. Phase 1 — Case A sweep (priority)

### The knob

`--gpu-memory-utilization = B / 31.84`, where the budget `B` covers weights + activations +
CUDA graphs + GPU KV. With ~1.3 GiB of non-weight overhead, GPU KV ≈ `B − 16.3 GiB`.

| B (GiB) | util | GPU KV (GiB) | GPU KV as % of 24 GiB working set |
|---|---|---|---|
| 30 | 0.942 | ~13.7 | 57% |
| 28 | 0.879 | ~11.7 | 49% |
| 26 | 0.817 | ~9.7 | 40% |
| 24 | 0.754 | ~7.7 | 32% |
| 22 | 0.691 | ~5.7 | 24% |
| 20 | 0.628 | ~3.7 | 15% |
| 19 | 0.597 | ~2.7 | 11% |
| **18** | 0.565 | **~1.7** | 7% ← "model + 2 GB" floor |

Actual GPU KV is read back from the vLLM startup log, not assumed. If 18 GiB fails to
allocate, the floor moves up one step.

### Arms (at every budget)

| arm | config | what it isolates |
|---|---|---|
| **A0 — no offload** | `--kv-offloading-size` unset | control. Misses are *recomputed*; scarcity also drives V1 recompute-preemption. |
| **A1 — DRAM offload** | `--kv-offloading-size 24 --kv-offloading-backend native` | misses served over PCIe instead |

The CPU pool stays **fixed at 24 GiB** while the GPU tier shrinks, so total KV capacity always
covers the working set. Only the *hot tier* size varies — that is the independent variable.
(The alternative — hold GPU+CPU constant at 24 GiB and vary the split — answers a different
question and is a possible follow-up.)

8 budgets × 2 arms = **16 server runs**, sequential on GPU 0.

### Workload — synthetic multi-session prefix reuse

Off-the-shelf datasets don't let us pin the working set to exactly 24 GiB, so the harness
generates it:

- **32 sessions × 6144-token unique prefix** = 196,608 tokens = **24.0 GiB of KV** exactly
- each request = `session_prefix + ~128 unique tokens`, **128 output tokens**
- **warmup pass** touches all 32 sessions once (populates both tiers), excluded from stats
- **300 measured requests**, Poisson arrivals at a fixed QPS (default 2.0)
- **access skew** is a first-class parameter:
  - `zipf-1.1` (**primary**) — realistic hot/cold split, produces a genuine knee
  - `uniform` (**secondary**) — LRU-thrashing worst case
- fixed seed; 1 run per point, **3 repeats at 3 key budgets** (30, 24, 18) for error bars

Skew matters: under uniform access an LRU hot tier thrashes and the curve is nearly flat and
bad everywhere; under Zipf the curve has an inflection where the hot set stops fitting. Both
are worth having, and the pair is more informative than either alone.

### Metrics

Client-side: **TTFT p50/p95/p99** (the headline — offload cost lands almost entirely here),
ITL/TPOT, end-to-end latency, output tok/s, goodput.
Server-side (`/metrics`): GPU prefix-cache hit rate, KV-connector bytes offloaded/loaded and
transfer time, **preemption count**, running/waiting queue depth.
System: `nvidia-smi` memory + PCIe sampling at 1 Hz, host RSS.

### Expected result (state it now, so the run can falsify it)

TTFT p50 roughly flat from 30→24 GiB (hot set still fits), a knee around 22–20 GiB, then a
rise toward the floor bounded by `t_load ≈ 85 ms` per 8K prefix — while **A0 rises far
steeper**, toward ~1 s recompute plus preemption thrash. If A1 does *not* stay well below A0
at small budgets, either PCIe is slower than calibrated or the hit rate collapsed; both are
diagnosable from the metrics above.

---

## 4. Phase 2 — Plot

Primary figure, `plots/case_a.png`:

- x: **VRAM budget (GiB)**, annotated with the derived GPU-KV GiB
- y1: TTFT p50 with p95 band — two lines (A0 no-offload, A1 DRAM-offload), two panels or
  line styles for zipf vs uniform
- y2 (right axis, faint): GPU prefix-cache hit rate
- vertical marker at the "model + 2 GB" floor; horizontal markers at the Phase-0 constants
  (`t_load`, `t_prefill`) so the reader sees the curve approach its physical bounds

Secondary: output tok/s vs budget; bytes-moved-over-PCIe vs budget; preemptions vs budget
(A0 only).

---

## 5. Phase 3 — Case B (weight offload), later

`--cpu-offload-gb X` streams X GiB of weights from DRAM every forward pass. Because it
applies to a model that *does* fit, we can emulate oversubscription **without downloading a
bigger model**: sweep `X ∈ {0, 2, 4, 8, 12}` on the same Llama-3-8B.

Decode becomes PCIe-bound, so expect ~linear degradation: each token costs an extra
`X GiB / 12 GB/s` ≈ **83 ms per GiB offloaded**. Cheap to run, and it makes the contrast with
Case A sharp — offloading *reuse* is nearly free, offloading *weights* is brutal, because
weights are re-read every single pass.

Phase 4 (optional): re-run Phase 1's A1 arm with `--kv-offloading-backend lmcache` as a third
curve. Needs `pip install lmcache` (0.5.3, cp312 wheel, unpinned torch dep) — see risk below.

---

## 6. Risks

| risk | mitigation |
|---|---|
| dev build of vLLM has rough edges in `--kv-offloading-size` | **Phase 1 starts with a single smoke run** at B=24 before committing to the matrix |
| 24 GiB pinned host alloc on a 60 GiB box | measured `ulimit -l` = **7.5 GiB**, well under the 24 GiB pool. CUDA `cudaHostAlloc` normally bypasses `RLIMIT_MEMLOCK` (the driver pins on the process's behalf), so this is expected to be fine — but **Phase 0 verifies it explicitly** with a 24 GiB pinned alloc before the sweep depends on it. Fallbacks: raise the limit via `ulimit -l unlimited` / limits.conf, or drop the pool to 20 GiB. |
| LMCache wheel built against release-torch ABI vs local nightly | try the wheel; on `undefined symbol`, either source-build (nvcc 12.8 present) or drop Phase 4 |
| run-to-run noise | fixed seeds, warmup excluded, 3 repeats at 3 budgets, GPU 1 idle throughout |
| server startup ~1–2 min × 16 runs | budgeted below; harness reuses a server across repeats of the same config |

Wall-clock: Phase 0 ~30 min · Phase 1 ~2–2.5 h · Phase 2 ~30 min · Phase 3 ~45 min.

---

## 7. Repo layout

```
bench/
  workload.py      # synthetic multi-session prefix generator (sessions × prefix_len → exact GiB)
  client.py        # async load generator; per-request TTFT/ITL via streaming completions
  server.py        # launch vLLM per config, wait for /health, scrape /metrics, teardown
  configs.py       # the sweep matrix
  run_sweep.py     # driver: for each config → server up → warmup → measure → JSON out
scripts/
  calibrate_pcie.py    # Phase 0
  monitor.py           # 1 Hz nvidia-smi / RSS sampler
plots/
  plot_case_a.py
results/             # raw JSON gitignored; summary.csv committed
PLAN.md
```
