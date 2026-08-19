# VRAM Limit: how far can you shrink an LLM server's VRAM before DRAM offloading stops saving you?

An empirical study on **RTX 5090 (32 GB) + Llama-3-8B**, where the application's working set
(~39 GiB) does not fit in the GPU and the overflow lives in DRAM.

- **Case A (primary):** weights stay resident, the **KV cache** is oversubscribed. Shrink the
  VRAM budget from ~the whole card down to "model + 2 GiB" and measure what it costs.
- **Case B (later):** the **model** doesn't fit; sweep how much of it is streamed from DRAM.

Design, reasoning and expected results: [PLAN.md](PLAN.md).
Running narrative of what was done and what broke: [PROGRESS.md](PROGRESS.md).
Findings with figures: [RESULTS.md](RESULTS.md).

---

## Quick start

```bash
# 1. one-time environment setup (see "Environment" below for why this is needed)
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --index-url https://pypi.org/simple "transformers>=4.56,<5"

# 2. Phase 0 -- measure the machine's physical constants
.venv/bin/python scripts/calibrate_pcie.py

# 3. validate the whole pipeline in ~3 minutes (2 configs, tiny working set)
cd scripts && ../.venv/bin/python run_sweep.py --smoke

# 4. the full Case A sweep (~2-3 h; run it detached, it outlives a terminal)
cd scripts && nohup ../.venv/bin/python run_sweep.py > ../output/logs/sweep.out 2>&1 &

# 5. figures + results
.venv/bin/python scripts/plot_case_a.py
```

Progress is visible while a sweep runs: `output/summary.csv` is rewritten after **every**
config, and each run's server log is `output/logs/<run-name>.log`.

---

## Layout

| path | what it is |
|---|---|
| `scripts/` | everything executable (harness + plotting) |
| `output/` | results: `summary*.csv`, `calibration_pcie.json`, `raw/<run>.json`, `logs/` |
| `figures/` | generated PNGs |
| `PLAN.md` | experiment design and the reasoning behind each choice |
| `PROGRESS.md` | append-only work log |
| `RESULTS.md` | findings, figures, interpretation |

`output/raw/` and `output/logs/` are gitignored (bulky, regenerable); the CSV summaries,
calibration JSON, and figures are committed.

## The scripts

| script | role |
|---|---|
| `calibrate_pcie.py` | **Phase 0.** Pinned/pageable H2D+D2H bandwidth across transfer sizes, PCIe link state under load, and a 24 GiB pinned-allocation test. Derives `t_load`, the PCIe cost of one 8k-token prefix — the constant every later number is read against. |
| `workload.py` | Builds the synthetic multi-session workload with an **exactly-known** KV working set. Importable and runnable (`python scripts/workload.py` prints the shape). |
| `client.py` | Open-loop async load generator. Poisson arrivals, streaming completions, per-request TTFT / ITL / e2e. |
| `server.py` | Starts one vLLM server per sweep point, waits for `/health`, parses actual KV sizing out of the startup log, scrapes `/metrics`, tears down. Runnable standalone: `python scripts/server.py --util 0.9 --hold`. |
| `run_sweep.py` | The driver. Config matrix → per-point server lifecycle → warmup → measured load → `output/raw/*.json` + `output/summary*.csv`. |
| `probe_offload_stall.py` | Diagnostic written to explain an anomaly (see PROGRESS). Measures offload store cost sequentially and under a concurrent burst. |
| `plot_case_a.py` | Figures from `output/summary*.csv` into `figures/`. |

## How the experiment is set up

**Workload.** 32 sessions × 6144-token unique prefixes = 196,608 tokens = **exactly 24.0 GiB
of KV** (Llama-3-8B is 0.125 MiB/token). Each request is one session's prefix plus a 128-token
unique suffix, generating 128 tokens. Sessions are drawn Zipf-1.1 (90% of requests land on 17
sessions = 12.75 GiB) or uniform. Prompts are sent as **raw token IDs** so prefix lengths are
exact and block-aligned — text would be re-tokenized and drift.

**The knob.** `--gpu-memory-utilization = budget / 31.3536 GiB` (vLLM sizes against torch's
total, not nvidia-smi's). GPU KV ends up ≈ `budget − 14.96 (weights) − ~1.3 (overhead)`. The
harness never assumes this: it parses the real number out of each server's startup log.

**Two arms at every budget**, which is what makes the plot mean anything:
- `nooffload` — misses are **recomputed** (and scarcity drives V1 recompute-preemption)
- `offload` — `--kv-offloading-size 24 --kv-offloading-backend native`, misses come over PCIe

The CPU pool is **fixed at 24 GiB** while the GPU tier shrinks, so only the hot-tier size
varies. `--disable-hybrid-kv-cache-manager` is passed to **both** arms (the offload connector
requires it) so they share one allocator.

**What "offloading" means here.** vLLM's native offloading is a *second-level prefix cache*,
not paging: an actively decoding sequence still needs its blocks in VRAM. It rescues **reuse
across requests**; it does not let one request's KV exceed the GPU.

## Environment

Measured on this machine — the design depends on these, so re-check them if you move it:

| | |
|---|---|
| GPU | 2× RTX 5090, 32607 MiB each; **GPU 0 only**, GPU 1 left idle |
| **PCIe** | **Gen4 ×8** (`Host Max: 4`) → measured **14.47 GB/s** H2D, 14.33 GB/s D2H. The cards do Gen5 ×16; this host does not. |
| DRAM | 60 GiB (≈55 free). A 24 GiB pinned pool allocates fine despite `ulimit -l` = 7.5 GiB. |
| vLLM | `0.15.0rc2.dev23+g5d3d6e44e` (source build, cu128) in the base conda env |
| model | `NousResearch/Meta-Llama-3-8B-Instruct`, fp16, 8192-token context, already in the HF cache |

Three environment fixes the harness applies or needs, all discovered the hard way:

1. **`transformers` 4.45 in the base conda env** is below vLLM 0.15's `>=4.56` floor — vLLM
   is unusable there. Hence the `--system-site-packages` venv with transformers 4.57 layered
   on top; the base env other projects use is left alone.
2. **`GLIBCXX_3.4.32 not found`** — conda ships libstdc++ 6.0.29, `vllm._C` needs 3.4.32.
   `server.py` preloads the system libstdc++ (3.4.33) into the server's environment.
3. **`--enforce-eager` is currently required.** The local vLLM build (Feb 2026) sits on a
   March-2025 torch nightly and `torch._dynamo` cannot compile vLLM's parameter
   `__torch_function__`. Eager mode inflates absolute latencies but applies equally to both
   arms. Drop `--extra --enforce-eager` once a matched torch/vLLM pair is installed.

## Reproducing a single point

```bash
cd scripts && ../.venv/bin/python run_sweep.py --budgets 24 --skews zipf --arms offload --requests 100 --tag oneoff
```

Or just bring a server up by hand and poke it:

```bash
.venv/bin/python scripts/server.py --util 0.7655 --kv-offload-gib 24 --extra="--enforce-eager" --hold
```

Useful `run_sweep.py` flags: `--budgets`, `--skews`, `--arms`, `--requests`, `--qps`,
`--repeats`, `--cpu-pool-gib`, `--tag`, `--dry-run`.
