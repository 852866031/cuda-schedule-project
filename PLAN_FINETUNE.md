# LoRA Finetuning under a Shrinking VRAM Budget — Test Plan


> **This is the design as written before running anything**, kept as a record of what was
> predicted. Several predictions in it turned out wrong — where the study says otherwise,
> the results document is correct. See [RESULTS_FINETUNE.md](RESULTS_FINETUNE.md).

**Question:** with the base model's weights offloaded to DRAM, how does LoRA finetuning
throughput degrade as the VRAM budget shrinks?

Companion to [PLAN.md](PLAN.md) (inference). Same machine, same model, same measured
constants — deliberately, so the two studies are directly comparable.

---

## 1. Why this is a different regime, and a better-posed question

The inference study found that offloading model **weights** is hopeless for serving: weights
are re-read every forward pass, so at decode's batch sizes the cost lands on every token.
Training inverts this, because a training step's forward pass serves `B x S` tokens instead of
`B`:

| | tokens per weight-read | verdict |
|---|---|---|
| decode | B (e.g. 4) | 297 ms/token at 4 GiB offloaded — unusable |
| training | B x S (e.g. 8192) | streaming hides behind compute — **viable** |

Streaming all 14.96 GiB costs **2.22 s per step** (fetched once for forward, again for
backward, at the measured 14.47 GB/s). Compute is `6 x 8.03e9 x T / 150 TFLOPS`:

| tokens/step | example | compute | streaming hidden? |
|---|---|---|---|
| 1,024 | B=1 x S=1024 | 0.33 s | no |
| 4,096 | B=2 x S=2048 | 1.32 s | no |
| **8,192** | **B=4 x S=2048** | **2.63 s** | **yes** |
| 32,768 | B=16 x S=2048 | 10.53 s | yes, easily |

**Break-even: ~6,900 tokens/step.** Finding where that knee really sits, and whether real
frameworks achieve it, is the point of this experiment.

Two things also make this *cleaner* to measure than the inference study:

- **No queueing.** Fixed batch, no arrivals. None of Little's law, Poisson bursts, or
  preemption that dominated the inference results.
- **Throughput is the natural metric.** No TTFT, no SLO, no open-loop artifact pinning
  throughput to the arrival rate (which flattened it at 254 tok/s in every inference config).

## 2. Testbed

Same box, same measured constants: RTX 5090 31.35 GiB usable, **PCIe Gen4 x8 = 14.47 GB/s**
measured, 60 GiB DRAM, GPU 0 only.

**Model:** Llama-3-8B-Instruct bf16, already cached. **Frozen** base weights = **14.96 GiB**
— this is the entire offloadable quantity.

**LoRA:** r=16 on `q,k,v,o` projections, all 32 layers.

| | |
|---|---|
| trainable params | **13.6 M** (0.17% of the model) |
| adapter weights | 26 MB |
| + grads + Adam fp32 (m, v, master) | **208 MB total** |

That 208 MB is the whole reason LoRA is the right choice here. A full finetune needs ~128 GB
of optimizer state against 60 GiB of DRAM — it does not fit, and 8-bit Adam only barely
rescues it at 48 GB. **LoRA removes the optimizer from the experiment entirely**, leaving a
clean two-way split of VRAM between *frozen weights* and *activations*. Frozen weights are
also read-only, so streaming never needs a write-back path.

**Gradient checkpointing: on.** Mandatory — without it activations for 8192 tokens run to tens
of GiB and there is no budget left to sweep.

**Activation memory** (computed, checkpointing on):

| batch x seq | tokens/step | boundaries | recompute | logits | total |
|---|---|---|---|---|---|
| 1 x 2048 | 2,048 | 0.50 | 0.12 | 0.49 | **1.11 GiB** |
| 2 x 2048 | 4,096 | 1.00 | 0.23 | 0.98 | **2.21 GiB** |
| **4 x 2048** | **8,192** | 2.00 | 0.47 | 1.96 | **4.43 GiB** |
| 8 x 2048 | 16,384 | 4.00 | 0.94 | 3.91 | **8.85 GiB** |

Note the logits term (vocab 128,256) is nearly as large as all 32 layers of boundary
activations. If VRAM gets tight, chunked cross-entropy is the first thing to reach for.

## 3. The knob and the sweep

**Fix** batch at **4 x 2048 = 8192 tokens/step** — just above the computed break-even, which
is the interesting place to sit. **Vary** how many of the 32 transformer layers keep their
weights resident in VRAM; the rest stream from DRAM each pass.

Layers are the natural unit: each is ~0.467 GiB, streaming is layer-wise anyway, and the knob
is exactly monotonic. Implied VRAM budget = `resident_layers x 0.467 + 4.43 (activations) +
0.21 (LoRA) + ~2.0 (workspace)`.

| resident layers | offloaded | streamed GiB | implied VRAM budget |
|---|---|---|---|
| 32 | 0 | 0.00 | ~21.6 GiB (reference: nothing streams) |
| 24 | 8 | 3.74 | ~17.9 GiB |
| 16 | 16 | 7.48 | ~14.1 GiB |
| 8 | 24 | 11.22 | ~10.4 GiB |
| 4 | 28 | 13.09 | ~8.5 GiB |
| **0** | **32** | **14.96** | **~6.6 GiB** (floor) |

**Floor = 6.63 GiB** — activations + LoRA state + workspace, with every weight streamed. That
is the hard limit: below it, activations no longer fit and it OOMs rather than degrades.

Every point also reports `torch.cuda.max_memory_allocated()`, so the implied budget is
verified, never assumed — the same discipline as parsing vLLM's real KV size in the
inference study.

## 4. Arms

| arm | what | isolates |
|---|---|---|
| **A0 — all resident** | 32 layers on GPU | the reference: no streaming at all |
| **A1 — streamed, with prefetch** | offloaded layers fetched ahead on a separate CUDA stream | can overlap hide the transfer? |
| **A2 — streamed, no prefetch** | fetch on demand, synchronous | the no-overlap lower bound |

A1 vs A2 is the heart of it. The analysis says streaming *should* be free at 8192 tokens/step
**if** the framework overlaps properly. A2 gives the pessimistic bound to measure against, and
the gap between them is exactly the value of prefetching. Reporting only A1 would leave "did
overlap work?" unanswerable.

## 5. Second sweep: the amortization curve

Fix the budget at a tight setting (0 or 4 layers resident) and vary **tokens per step**:
2048, 4096, 8192, 16384, 32768 (via batch size, or gradient accumulation for the largest).

This is the money plot: **step time vs tokens/step should be flat-then-linear**, with the knee
at the point where compute overtakes the 2.22 s of streaming. Predicted at ~6,900 tokens.
Measuring the knee empirically and comparing it to the prediction is the cleanest result this
experiment can produce.

## 6. Metrics

- **tokens/s** and **s/step** (primary)
- **MFU** = `6 x 8.03e9 x T / (step_time x 150e12)` — normalizes throughput against what the
  GPU could theoretically do, and makes the overlap question legible
- **PCIe bytes/step and achieved GB/s**, against the 14.47 GB/s ceiling
- **GPU idle fraction** — the direct evidence for whether the GPU is waiting on the link
- `torch.cuda.max_memory_allocated()` and nvidia-smi sampling
- loss curve, purely as a correctness check that offloading changes nothing numerically

## 7. Walls (computed up front, as in the inference study)

| wall | threshold | expected to bind? |
|---|---|---|
| **Activation capacity** | VRAM < 6.63 GiB at 8192 tok/step | **yes — a hard OOM floor**, not a degradation |
| **PCIe bandwidth** | streaming demand > 14.47 GB/s | **yes — the binding constraint below break-even** |
| **DRAM capacity** | 14.96 GiB of weights vs 60 GiB | no, trivially fits |
| **Compute** | — | not a wall; staying compute-bound is the *goal* |

The symmetry with the inference study is the interesting part: there, the **capacity** walls
bound and bandwidth was never touched (5.7% of the link). Here, bandwidth is the wall and
capacity is a cliff rather than a curve. Same hardware, same offloading idea, opposite
limiting resource — because training reads weights per *step* while serving reads KV per
*reuse*.

## 8. Predictions (recorded before running, so they can be wrong)

1. **A1 at 8192 tok/step: nearly flat.** Offloading all 32 layers should cost <20% throughput,
   because 2.22 s of transfer hides behind 2.63 s of compute.
2. **A2: linear.** Step time ≈ compute + transfer, so ~1.8x slower at full offload.
3. **The knee in sweep 2 lands near 6,900 tokens/step** for A1.
4. If A1 comes out linear like A2, the framework is not overlapping — a finding about the
   tooling, and worth reporting as such rather than as a property of offloading.

## 9. Tooling — the weak point, honestly

There is no clean `--cpu-offload-gb` for training. Options, with real caveats:

| tool | knob | caveat |
|---|---|---|
| **DeepSpeed ZeRO-3** + `offload_param: cpu` | `stage3_max_live_parameters`, `stage3_prefetch_bucket_size` | designed for multi-GPU sharding; works at world_size 1 but the knob is a live-parameter budget, not a layer count. Has real prefetch — **primary candidate for A1** |
| **PyTorch FSDP** `CPUOffload(offload_params=True)` | all-or-nothing in FSDP1 | too coarse for a sweep; FSDP2 is finer |
| **HF Accelerate** `cpu_offload` / `max_memory` | per-device memory cap | built for inference device_map; naive fetch-on-demand — **natural fit for A2**, the no-overlap bound |
| **custom hooks** | exact layer count | most control and cleanest instrumentation, ~200 lines: pin K layers, stream the rest on a side CUDA stream with prefetch |

**Recommendation:** DeepSpeed ZeRO-3 for A1 and Accelerate for A2, with the custom harness as
fallback if DeepSpeed's live-parameter budget doesn't map cleanly onto a layer count. Phase 0
decides this before any sweep is committed to — same as the inference study, where a smoke
test at one config caught three environment blockers before the matrix ran.

## 10. Phases

| phase | work | time |
|---|---|---|
| **0** | env check (peft/deepspeed install), one smoke step at 32 resident, verify memory accounting against `max_memory_allocated`, confirm the knob moves resident bytes | ~1 h |
| **1** | budget sweep: 6 points x 3 arms at 8192 tok/step | ~1.5 h |
| **2** | amortization sweep: 5 token counts x 2 arms at the tight budget | ~1 h |
| **3** | figures + RESULTS_FINETUNE.md | ~0.5 h |

Reuse from the inference harness: the PCIe calibration, the run-per-config driver pattern,
the stall watchdog, `rebuild_summary.py`, and `status.py` all carry over with small edits.

## 11. Risks

| risk | mitigation |
|---|---|
| DeepSpeed's knob doesn't map to layer counts | Phase 0 checks it explicitly; custom harness as fallback |
| DeepSpeed build against torch 2.9.1 / sm_120 | install into `.venv-matched`; it compiles ops JIT, nvcc 12.8 is present |
| logits memory dominates at large batch | chunked cross-entropy, or cap seq at 2048 |
| streaming makes steps so slow the sweep drags | cap at 20 measured steps/config after 3 warmup steps |
| pinned-memory pressure (15 GiB of weights pinned) | already proved a 24 GiB pinned alloc works on this box |
