# VRAM Limit

**How far can you shrink an LLM's VRAM budget before DRAM offloading stops saving you?**

Three experiments on the same machine — an RTX 5090 (32 GB) behind **PCIe Gen4 ×8** — asking that
question of inference, of finetuning, and of a disaggregated decode node. The first two are
complete, and they fail for opposite reasons.

| study | status | question | answer |
|---|---|---|---|
| **[Inference](reports/report_simple_inference.md)** | ✅ done | vLLM serving, KV cache oversubscribed, weights resident | VRAM 30 → 22 GiB (**58% less KV**) for **9% of TTFT p95**. Below that it collapses two orders of magnitude in two steps. The wall is **concurrency**, not cache capacity or bandwidth. |
| **[Finetuning](reports/report_finetune.md)** | ✅ done | LoRA r=16, base weights streamed from DRAM | **3.25 GiB freed for 1.4%** of throughput *with overlapped transfers* — **30% without**. The wall is **PCIe bandwidth**, and degradation is smooth rather than a cliff. |
| **[LMCache inference](reports/report_lmcache_inference.md)** | ✅ done | same sweep, LMCache as the DRAM tier | Same cliff at 19–20 GiB — the wall is the workload's, not the backend's. Below it: **zero preemptions, no stalls** — saturation queues instead of spiraling. |
| **[Split inference](reports/report_split_inference.md)** | ✅ done | GPU0 prefill → GPU1 decode over a shared LMCache; decode VRAM swept | TTFT ~105 ms flat until the wall at 20–22 GiB — set by in-flight KV, not the working set. Prefill GPU 97% idle; decode DRAM-bound. The hand-built transfer arm and why it was abandoned are §1. |
| **[Decode node](PLAN_DECODE.md)** | ✅ done — see [reports/report_split_inference.md](reports/report_split_inference.md) | GPU0 prefill → GPU1 decode, sweep GPU1's VRAM | measured over a shared LMCache; the plan's P2P transport arm was built, patched working, and superseded (report §1) |

Designs and predictions, written before running: [PLAN.md](PLAN.md),
[PLAN_FINETUNE.md](PLAN_FINETUNE.md), [PLAN_DECODE.md](PLAN_DECODE.md).

---

## ⚠️ Read this before running anything on this machine

**`P2pNcclConnector` pins 32 GiB of host memory *per instance* by default.** Two instances ask
for 64 GiB of unswappable memory on a 60 GiB box. That is not an OOM kill — the kernel cannot
reclaim pinned pages, so it starves the display server and the OOM killer alike. **It froze and
rebooted this workstation once.**

Always set `mem_pool_size_gb` explicitly. `scripts/inference/split_simple/disagg_launch.sh` does, prints the
budget, and refuses to start if the total is unsafe. Run `scripts/common/mem_guard.sh` alongside
anything that allocates pinned memory:

```bash
scripts/common/mem_guard.sh 12000 &   # kills vLLM if available RAM drops below 12 GB
```

---

## Picking this up

**Done:** both completed studies, their reports, figures and raw data. Nothing left to run.

**Historical — this section described phase 0 while it was blocked; the study is now
complete** (see [reports/report_split_inference.md](reports/report_split_inference.md)).
Kept because the failure mode is instructive. Two vLLM instances (prefill on GPU0, decode on
GPU1) plus a router come up cleanly, and the **NCCL handshake succeeds in both directions**.
What does not work: the decode leg never responds. The cause is located but not yet fixed —
`P2pNcclConnector`'s consumer blocks in

```python
while tensor_id not in self.recv_store:
    self.recv_store_cv.wait()          # unbounded — no timeout, no logging
```

so a key mismatch hangs the decode engine **permanently and silently**. Leading hypothesis: the
prefill and decode legs disagree on `request_id`, because vLLM can append suffixes to the id
taken from the `X-Request-Id` header. [PLAN_DECODE.md §13](PLAN_DECODE.md) has the full state
and the ordered steps to finish.

Two practical warnings for whoever continues:

- **Never probe with a bare `curl`** — that unbounded wait hangs the client indefinitely. Put a
  hard timeout on every probe.
- **One bad request wedges the whole decode engine**, not just that request. The sweep driver
  will need the stall watchdog from the inference study.

---

## Quick start

```bash
# 1. environments (see "Why two venvs" below)
python3 -m venv --system-site-packages .venv
PIP_CONFIG_FILE=/dev/null .venv/bin/pip install "transformers>=4.56,<5" peft accelerate matplotlib pandas

python3 -m venv .venv-matched
PIP_CONFIG_FILE=/dev/null .venv-matched/bin/pip install "vllm==0.15.1" pandas matplotlib

# 2. measure the machine's physical constants — everything else is read against these
.venv/bin/python scripts/common/calibrate_pcie.py

# 3. inference: validate the pipeline in ~3 min, then run the sweep (~2 h)
cd scripts/inference/simple && ../../.venv-matched/bin/python run_sweep.py --smoke
cd scripts/inference/simple && nohup ../../.venv-matched/bin/python run_sweep.py --tag main > ../output/logs/sweep.out 2>&1 &

# 4. finetuning (~10 min per arm)
.venv/bin/python scripts/finetune/finetune_sweep.py --find-batch
.venv/bin/python scripts/finetune/finetune_sweep.py --batch 2 --tag ft                        # no prefetch
.venv/bin/python scripts/finetune/finetune_sweep.py --batch 2 --prefetch --max-offload 16 --tag ft_prefetch

# 5. figures
.venv/bin/python scripts/plots/plot_workload.py
.venv/bin/python scripts/plots/plot_case_a.py
.venv/bin/python scripts/plots/plot_finetune.py
.venv/bin/python scripts/inference/simple/compute_walls.py     # wall thresholds from the calibration
```

The decode study is not runnable end to end yet; see **Picking this up** above.

**Watch a running sweep** — safe to run any time, read-only:

```bash
python3 scripts/common/status.py
```

It prints what is running, the live engine state, every completed result, and an ETA. `output/summary_*.csv` is also rewritten after **every** config, so partial results are always readable.

---

## Reproducing each result

### Inference (Case A)

The full matrix is 8 budgets × 2 arms × 2 access patterns. As run:

```bash
cd scripts/inference
../../.venv-matched/bin/python run_sweep.py --tag main                                    # everything
../.venv-matched/bin/python run_sweep.py --budgets 19 18 --skews zipf --arms offload --tag zipf_low
../.venv-matched/bin/python run_sweep.py --skews uniform --arms offload --tag uniform_off
../.venv-matched/bin/python run_sweep.py --budgets 30 --skews uniform --arms nooffload --tag uniform_ref
```

One point on its own, or a server to poke by hand:

```bash
cd scripts/inference/simple && ../../.venv-matched/bin/python run_sweep.py --budgets 24 --skews zipf --arms offload --requests 100 --tag oneoff
.venv-matched/bin/python scripts/inference/simple/server.py --util 0.7655 --kv-offload-gib 24 --hold
```

Useful flags: `--budgets --skews --arms --requests --qps --repeats --cpu-pool-gib --stall-timeout --tag --dry-run`.

### Finetuning (five offload strategies)

```bash
.venv/bin/python scripts/finetune/finetune_sweep.py --batch 2 --tag ft                                   # no prefetch, 0-32
.venv/bin/python scripts/finetune/finetune_sweep.py --batch 2 --prefetch --prefetch-depth 1 --max-offload 16 --tag ft_prefetch
.venv/bin/python scripts/finetune/finetune_sweep.py --batch 2 --prefetch --prefetch-depth 2 --max-offload 16 --tag ft_prefetch_d2
scripts/finetune/run_interleave.sh                                                                        # both interleaved arms
```

Flags: `--batch --steps --warmup --offload-step --max-offload --prefetch --prefetch-depth --pattern {tail,interleave} --tag`.

### Decode disaggregation (complete — see reports/report_split_inference.md)

```bash
scripts/common/mem_guard.sh 12000 &          # ALWAYS run this first
scripts/inference/split_simple/disagg_launch.sh 0.9568       # prefill GPU0 + decode GPU1 + router on :8000
# ... probe with a HARD TIMEOUT, never a bare curl ...
scripts/inference/split_simple/disagg_stop.sh                # tears down; also kills orphaned engine children
```

`disagg_launch.sh` takes the decode instance's `--gpu-memory-utilization` as its one argument —
that is the variable the study sweeps. Prefill is fixed at 28 GiB so it never bottlenecks.
Pinned pools: 1 GiB on prefill (never used under `PUT_ASYNC`), 24 GiB on decode (holds queued
KV). The current blocker is described in **Picking this up**.

### Regenerating derived numbers without re-running anything

Every run's per-request records are kept, so a fix to a derived column costs a rebuild, not an experiment:

```bash
cd scripts/inference/simple && ../../.venv/bin/python rebuild_summary.py --tag main
```

---

## Layout

| path | what it is |
|---|---|
| `scripts/common/` | shared: PCIe calibration, live status, memory watchdog |
| `scripts/inference/` | Case A harness: workload, client, server, sweep driver, wall calculator |
| `scripts/finetune/` | LoRA sweep and the layer streamer |
| `scripts/inference/split_simple/` | prefill/decode disaggregation: launcher, router, teardown |
| `scripts/plots/` | all figure generation |
| `output/` | `summary_*.csv`, `*_sweep.json`, `calibration_pcie.json`, plus gitignored `raw/` and `logs/` |
| `figures/` | generated PNGs |
| `reports/report_simple_inference.md`, `reports/report_finetune.md` | findings, with figures and limitations |
| `PLAN.md`, `PLAN_FINETUNE.md` | designs and predictions recorded up front |

### Scripts

Grouped by experiment; `common/` holds what all three share.

| script | role |
|---|---|
| `common/calibrate_pcie.py` | **Run this first.** Pinned/pageable H2D+D2H bandwidth, PCIe link state under load, and a 24 GiB pinned-allocation test. Derives `t_load`, the constant every later number is read against. |
| `inference/workload.py` | Synthetic multi-session workload with an **exactly-known** KV working set. Runnable: prints the shape. |
| `inference/client.py` | Open-loop async load generator: Poisson arrivals, streaming completions, per-request TTFT / ITL / e2e. |
| `inference/server.py` | One vLLM server per sweep point — launch, wait for `/health`, parse real KV sizing from the startup log, scrape `/metrics`, tear down. Runnable standalone with `--hold`. |
| `inference/run_sweep.py` | Inference driver: config matrix → server lifecycle → warmup → measured load → JSON + CSV. Includes the stall watchdog. |
| `inference/compute_walls.py` | The four wall thresholds, computed from measured constants rather than asserted. |
| `finetune/finetune_sweep.py` | LoRA finetuning driver: batch-size probe, then the offload sweep. |
| `finetune/layer_offload.py` | Streams a layer's frozen weights from pinned DRAM **correctly through backward**, with optional depth-*k* prefetch and tail/interleaved placement. |
| `inference/probe_offload_stall.py` | Diagnostic: measures offload store cost sequentially and under a concurrent burst. |
| `inference/rebuild_summary.py` | Regenerates a summary CSV from `output/raw/*.json` through the current derivation. |
| `common/status.py` | Live, read-only view of any running sweep. |
| `plots/plot_*.py` | Figures: access patterns, inference sweep, walls, finetuning strategies. |
| `decode/disagg_launch.sh` | Brings up prefill (GPU0) + decode (GPU1) + router, with pinned-pool caps and a preflight RAM guard. |
| `decode/disagg_p2p_proxy.py` | Router for `P2pNcclConnector`: mints request ids carrying both peer addresses, sends the prefill leg with `max_tokens=1`, then the decode leg. |
| `common/mem_guard.sh` | Kills the instances if available RAM collapses — pinned memory cannot be reclaimed, so overcommit freezes the box rather than triggering the OOM killer. |

---

## How the experiments are set up

### Inference

**Workload:** 32 sessions × 6144-token unique prefixes = **exactly 24.0 GiB of KV** (Llama-3-8B is 0.125 MiB/token). Each request is one session's prefix plus a 128-token unique suffix, generating 128 tokens. Prompts are sent as **raw token IDs** so prefix lengths are exact and block-aligned. Sessions drawn Zipf-1.1 or uniform.

**Knob:** `--gpu-memory-utilization = budget / 31.3536 GiB`. Weights and overhead never yield, so every GiB removed comes out of the KV cache. The harness parses the real KV size from each server's startup log rather than assuming it.

**Two arms:** `nooffload` (misses are recomputed) and `offload` (`--kv-offloading-size 24 --kv-offloading-backend native`). `--disable-hybrid-kv-cache-manager` is passed to **both** so they share one allocator.

> vLLM's native offloading is a **second-level prefix cache, not paging**. An actively decoding sequence still needs all its KV in VRAM. It rescues reuse *across* requests; it does not let a 40 GiB model run on a 32 GiB card.

### Finetuning

LoRA r=16 on `q,k,v,o` across all 32 layers — 13.6 M trainable params (0.17%), **208 MB** of adapter + gradient + optimizer state. That is what makes the experiment clean: full finetuning would need ~128 GB of optimizer state against 60 GiB of DRAM.

Batch **2 × 2048 = 4096 tokens/step** (found empirically; batch 4 OOMs on the fp32 logits tensor). Gradient checkpointing on — mandatory, not a tuning choice.

The knob is how many of the 32 layers keep their weights resident; the rest stream from pinned DRAM, crossing PCIe twice per step (forward, then the recomputed forward inside backward).

---

## Environment

Measured on this machine. The designs depend on these, so re-check if you move it:

| | |
|---|---|
| GPU | 2× RTX 5090, 32607 MiB each; **GPU 0 only**, GPU 1 left idle |
| **PCIe** | **Gen4 ×8** (`Host Max: 4`) → measured **14.47 GB/s** H2D, 14.33 GB/s D2H. The cards do Gen5 ×16; this host does not. |
| DRAM | 60 GiB. A 24 GiB pinned pool allocates fine despite `ulimit -l` = 7.5 GiB. |
| model | `NousResearch/Meta-Llama-3-8B-Instruct`, fp16, 8192-token context |

### Why two venvs

The base conda env cannot run vLLM at all — it has `transformers` 4.45 against vLLM 0.15's `>=4.56` floor — and its `torch` is a March-2025 nightly that `torch._dynamo` cannot compile vLLM 0.15 with. Rather than mutate an env other projects depend on:

- **`.venv`** (`--system-site-packages`) — newer transformers layered on top, plus `peft`/`accelerate`. Used for **finetuning and plotting**.
- **`.venv-matched`** (clean) — `vllm==0.15.1` + `torch 2.9.1+cu128`, a supported pair. Used for **inference**, with torch.compile and CUDA graphs enabled.

### Gotchas worth knowing

1. **Slow `pip install` on this box is the NGC extra-index.** `~/.pip/pip.conf` and `~/.config/pip/pip.conf` (both written by `nvidia-pyindex`) add `pypi.ngc.nvidia.com`, which does not resolve here — pip consults it for *every* package. `PIP_CONFIG_FILE=/dev/null` bypasses it; it took an install from 644 retries to zero.
2. **`GLIBCXX_3.4.32 not found`** — conda ships libstdc++ 6.0.29, `vllm._C` needs 3.4.32. `server.py` preloads the system libstdc++ automatically.
3. **`OffloadingConnector` requires `--disable-hybrid-kv-cache-manager`**, which the harness passes to both arms so the comparison stays fair.
4. **A stalled vLLM leaves its engine child alive holding VRAM.** `pkill -9 -f VLLM::EngineCore` is needed to actually free the card.
5. **`model.train()` is load-bearing for gradient checkpointing.** HF applies it only when `self.gradient_checkpointing and self.training`, and `from_pretrained` returns a model in eval mode — the flag reads `True` while checkpointing silently does nothing (14 GiB of activations instead of 1).
6. **`accelerate`'s `device_map` CPU offload cannot train.** It leaves parameters on the `meta` device and backward fails with *"expected device meta but got cuda:0"*. Hence `layer_offload.py`.
7. **`P2pNcclConnector` pins 32 GiB of host memory per instance by default** (`DEFAULT_MEM_POOL_SIZE_GB = 32`). Disaggregation means ≥2 instances, so the default asks for ≥64 GiB of **unswappable, unreclaimable** memory. On this 60 GiB box that is not an OOM kill but a **hard freeze and reboot** — the kernel cannot reclaim pinned pages, so it starves the display server and the OOM killer alike. Always set `mem_pool_size_gb` in `kv_connector_extra_config`; `disagg_launch.sh` sets 4 GiB and refuses to start if the pools would exceed a third of RAM.
