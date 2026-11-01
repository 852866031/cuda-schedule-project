# Decode-Only Serving under a Shrinking VRAM Budget — Test Plan

> **Design written before running — kept intact as the pre-registration record. The study
> is complete**: results in [reports/report_split_inference.md](reports/report_split_inference.md).
> How reality diverged from this plan, in brief: the P2pNcclConnector transport was made
> to work (four patches; the request-id handshake below §4 was the connector being broken
> in vLLM 0.15.1) but the architecture was superseded by a shared LMCache (report §1
> explains why). The closed-loop N=32 design of §5 was replaced by the same open-loop
> 2 QPS workload as the other studies, for comparability. Of §9's predictions: the
> decode node's *throughput* does fall smoothly with budget (1), preemptions are ~zero
> but for an admission-arithmetic pocket (3), and the "fixed ~64 ms transfer" (5) was
> falsified — the KV path costs far more, and where it lands (TTFT vs the token-1→2 gap)
> turned out to be router policy, the study's central finding.

**Question:** in a real disaggregated prefill/decode deployment, how does **decode**
performance degrade as the decode node's VRAM is reduced?

Third study on the same machine, after [PLAN.md](PLAN.md) (inference) and
[PLAN_FINETUNE.md](PLAN_FINETUNE.md) (LoRA finetuning). Same hardware, same model, same
measured constants, so all three are comparable.

**GPU0 runs prefill only. GPU1 runs decode only, and its VRAM is the swept variable. Only GPU1
is measured.**

---

## 1. Why decode-only is a better-posed question

The inference study had a structural weakness: at a fixed 2 QPS open loop, **throughput was
pinned at the arrival rate** (254.7 tok/s in every healthy config) and only moved once the
server failed. Throughput was never a dependent variable.

Splitting the phases fixes that:

1. **Prefill leaves the node.** GPU1's capacity becomes entirely about how many sequences it can
   keep decoding at once — no prefill work competing for the same engine steps.
2. **VRAM caps that number directly.** KV for running sequences is the only elastic consumer, so
   shrinking VRAM shrinks the batch, and batch size *is* decode throughput.

Driving GPU1 to **saturation** (closed-loop load, §5) then makes throughput measure the server
rather than the client.

## 2. What actually limits decode

Every decode step reads **all model weights** and **all KV of every running sequence** from
VRAM. It is memory-bandwidth-bound, not compute-bound:

| term | size | cost per step |
|---|---|---|
| weights, re-read every step | 14.96 GiB | **11.2 ms** — a fixed floor at *any* batch size |
| KV per sequence (ISL 6144 + OSL 512) | 0.812 GiB | 0.61 ms each |

(Against ~1.79 TB/s theoretical VRAM bandwidth, taken at 80% achievable.)

**The weight read is a fixed per-step cost amortised over the batch.** At batch 1 you pay
11.2 ms to emit one token; at batch 16 you pay 21 ms to emit sixteen. Decode throughput is
therefore driven by batch size, and VRAM sets the batch.

Same amortisation argument as the finetuning study's weight streaming — but here the weights are
already resident and the re-read is against ~1.4 TB/s instead of 14.5 GB/s.

## 3. Predicted curve

`batch = (budget − 14.96 − 1.3) / 0.812`, `step = 11.2 ms + batch × 0.61 ms`:

| decode budget | GPU KV | max batch | step | **predicted tok/s** |
|---|---|---|---|---|
| 30 GiB | 13.74 | 16 | 21.0 ms | **763** |
| 28 | 11.74 | 14 | 19.8 ms | 709 |
| 26 | 9.74 | 11 | 17.9 ms | 614 |
| 24 | 7.74 | 9 | 16.7 ms | 539 |
| 22 | 5.74 | 7 | 15.5 ms | 452 |
| 20 | 3.74 | 4 | 13.7 ms | 293 |
| 19 | 2.74 | 3 | 13.0 ms | 230 |
| 18 | 1.74 | 2 | 12.4 ms | **161** |

**Predicted shape: smooth, monotonic, diminishing returns.** 18 → 22 GiB (4 GiB) buys
291 tok/s; 26 → 30 GiB (also 4 GiB) buys only 149. Early GiB amortise the fixed weight read
across more sequences; later ones only add KV bandwidth.

**No cliff expected.** With a closed loop there is no arrival process, so none of the queueing
feedback or preemption spiral that ended the inference study.

## 4. Testbed

### Topology

```
        client (closed loop, N in flight)
                     │
              proxy  │  scripts/disagg_proxy.py
             ┌───────┴───────┐
   max_tokens=1              │
             ▼               ▼
      GPU0: prefill  ──KV──▶ GPU1: decode
      kv_producer            kv_consumer
      fixed 28 GiB           ** swept 30 → 18 GiB **
```

### Transport: `P2pNcclConnector`, in-tree, no extra dependencies

```bash
--kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer",
  "kv_connector_extra_config":{"send_type":"PUT_ASYNC","http_port":"8100",
                               "proxy_ip":"","proxy_port":""}}'
```

`proxy_ip`/`proxy_port` may be empty — they only drive service discovery pings. The KV path
itself is ZMQ for handshake plus NCCL for the transfer. `nixl` and `mooncake` are **not
installed** and are not needed.

### The topology objection, measured and dismissed

P2P is disabled between these consumer cards (`CNS`, PHB topology), so KV stages through host
DRAM. That turns out to cost almost nothing:

| path | measured |
|---|---|
| raw GPU0→GPU1 staged copy (1 GiB) | **13.19 GB/s** |
| **NCCL send/recv (1 GiB)** | **12.62 GB/s** |
| single-hop H2D reference | 14.47 GB/s |

NCCL falls back to SHM transport and pipelines the two hops, so missing P2P costs ~13%, not 2×.
**One 6144-token prefix crosses in ~64 ms against ~530 ms to prefill it** — 12% of the work it
saves.

There is **no way to get true peer DMA on these cards**: `nvidia-smi topo -p2p r` reports `CNS`
(Chipset Not Supported) and `can_device_access_peer` is False. NVIDIA disables it on GeForce.
The staged path is the only option, and it is fast enough that this does not matter.

**Verified in phase 0:** both instances came up and completed the NCCL handshake —
`🤝ncclCommInitRank Success, 192.168.68.91:21001👉192.168.68.91:22001` on the producer and
`22001👈21001` on the consumer. The transport layer is not in question.

### Router

`scripts/disagg_proxy.py`, vendored from vLLM v0.15.1's
`examples/online_serving/disaggregated_serving/disagg_proxy_demo.py` (pip does not ship
`examples/`). It sends each request to a prefill instance with `max_tokens=1`, then to a decode
instance. Upstream marks it as a demo slated for replacement by PDController (PR #15343), so it
is a fixture of this experiment, not a supported interface.

```bash
python scripts/disagg_proxy.py --model $MODEL --prefill localhost:8100 \
    --decode localhost:8200 --port 8000
```

### The prefill node must not be the bottleneck

GPU0 gets a **fixed 28 GiB** budget and is never swept. Its job is to keep GPU1 fed. If prefill
saturates first, the decode curve measures the wrong thing — so §7 records prefill-side
utilisation as a **validity check**, not as a result.

## 4b. The DRAM transfer reservation

`P2pNcclConnector` keeps arrived-but-not-yet-decoded KV in GPU memory up to `kv_buffer_size`
(default 1 GB), then **spills to a pinned host pool** sized by `mem_pool_size_gb`. That spill
path lives in `listen_for_requests` under `cmd == "PUT"` — the **receiving** side — so with
`send_type=PUT_ASYNC` only the decode instance uses it. The producer's pool is dead allocation.

**Reserved for this study: 24 GiB on the decode node**, 1 GiB on prefill.

| | pinned pool | why |
|---|---|---|
| prefill (producer) | 1 GiB | never spills under PUT_ASYNC |
| **decode (consumer)** | **24 GiB** | holds queued KV: 24 / 0.81 ≈ **29 requests** in flight ahead of decode |
| total | 25 GiB of 60 | leaves ~28 GiB after both engines' ~8 GiB of other host memory |

24 GiB pinned is known-good on this box — the Phase 0 PCIe calibration allocated exactly that
in 2.95 s and sustained full bandwidth with it held.

**This reservation is what makes a deep queue safe**, and it removes the constraint that would
otherwise have forced client concurrency down: at N=32 with a decode batch of 2, up to 30
requests' KV (24 GiB) can sit queued. That is precisely the reservation.

> **The default is a trap.** `DEFAULT_MEM_POOL_SIZE_GB = 32`, *per instance*. Two instances ask
> for 64 GiB of unswappable memory, which on a 60 GiB box is not an OOM kill but a hard freeze
> and reboot — it happened once here. `scripts/inference/split_simple/disagg_launch.sh` now sets both pools explicitly, prints
> the budget, and refuses to start if the total exceeds a safe share of RAM. `scripts/common/mem_guard.sh`
> kills the instances if available RAM collapses anyway.

## 5. Load model — closed loop, deliberately

The inference study used **open-loop** Poisson arrivals, right for latency work and wrong here:
it caps throughput at the offered rate. This study uses a **closed loop with fixed concurrency
N**: N requests in flight at all times, a new one issued as each completes.

Set **N = 32**, comfortably above the largest batch any budget can hold (16), so the decode node
runs at its ceiling at every point and measured throughput *is* capacity.

Reusing `client.py` needs one change: replace Poisson scheduling with a semaphore of N.

## 6. Workload

- **ISL 6144, OSL 512** → 6656 tokens, inside Llama-3's 8192 window. Long OSL is deliberate:
  this is a decode study, so most of each request's life should be decode.
- **Unique prompts, prefix caching disabled.** The opposite of the inference study — a decode
  node has no reuse to exploit, and a cache hit would confound the batch-size measurement.
- **Equal-length sequences**, so batch composition is uniform and `batch × 0.812 GiB` is exact.

**Second dimension (phase 2): ISL** ∈ {2048, 4096, 6144} at a fixed mid budget. KV per sequence
scales with ISL, so batch — and therefore throughput — should fall roughly as 1/ISL. That tests
the model, not just the curve.

## 7. Metrics

From **GPU1 only** unless noted:

- **output tok/s** (primary — finally a real dependent variable)
- **achieved batch size** from `vllm:num_requests_running`. The causal chain is
  VRAM → batch → throughput, so the middle term must be measured, not inferred.
- ITL p50/p95; TTFT (now includes the KV transfer, so it prices the interconnect)
- `vllm:num_preemptions` — expected **zero** throughout; if not, the model is wrong
- GPU KV size parsed from the startup log, as in the other studies
- **validity check:** GPU0 utilisation and queue depth, to confirm prefill never bottlenecks

## 8. Walls

| wall | threshold | expected? |
|---|---|---|
| **Batch = 1** | KV < 0.812 GiB → decode budget < 17.1 GiB | hard floor; below it one sequence will not fit |
| **VRAM bandwidth** | weights + KV reads saturate ~1.4 TB/s | **binding throughout** — this is what the curve measures |
| **PCIe / NCCL** | 12.6 GB/s GPU→GPU | ~64 ms per request against a ~2.6 s decode phase — should not bind |
| **Prefill capacity** | GPU0 at 28 GiB | must not bind; monitored as a validity check |
| **Concurrency/queueing** | — | **not applicable**: closed loop, no arrival process |

The three studies will then have hit three different limiting resources on one machine:
**concurrency** (inference), **PCIe bandwidth** (finetuning), **VRAM bandwidth** (decode).

## 9. Predictions, recorded before running

1. Throughput falls **smoothly and monotonically**, no cliff, matching §3 to within ~25% (the
   80%-of-peak bandwidth assumption is the weakest input).
2. **Diminishing returns:** GiB added near the floor are worth ~2× GiB added near the top.
3. **Zero preemptions** at every budget.
4. ITL rises modestly (12.4 → 21.0 ms predicted) while throughput falls 4.7× — ITL is per-token,
   throughput is per-batch. If ITL rises faster than predicted, something other than batch size
   is degrading.
5. TTFT carries a roughly **fixed ~64 ms** of transfer, independent of the decode budget.

## 10. Optional arm: DRAM offloading on the decode node

The original framing included offloading on GPU1. It is kept as an optional arm because **its
role on a decode node is genuinely unclear**: every running sequence needs its KV resident to
take a step, and prefix reuse is gone. The plausible job is holding KV for *paused* sequences
instead of preempting and recomputing them — but with a closed loop and no preemptions expected,
there may be nothing for it to do.

Combining it with `P2pNcclConnector` needs `MultiConnector`, which is untested here. Attempt
only after phases 1–2 land, and only if preemptions actually appear.

## 11. Risks

| risk | mitigation |
|---|---|
| **`P2pNcclConnector` pins 32 GiB of host memory per instance by default** | **Already bit us: two instances asked for 64 GiB of unswappable memory on a 60 GiB box and froze the machine hard enough to force a reboot.** `scripts/inference/split_simple/disagg_launch.sh` now sets `mem_pool_size_gb: 4` and refuses to launch if the pools exceed a third of RAM. Phase 0 must also confirm 4 GiB is enough at the target concurrency and that exhaustion degrades gracefully rather than hanging. |
| `P2pNcclConnector` config is otherwise fiddly and lightly documented | phase 0 brings up both instances and transfers a single request before anything is swept |
| The vendored proxy is a demo, slated for upstream removal | pinned to v0.15.1 and vendored into the repo, so it cannot drift |
| NCCL between two instances may pick a bad transport | measured: 12.62 GB/s via SHM; assert this at startup |
| Prefill node becomes the bottleneck | fixed generous budget, plus a validity check on GPU0 |
| Three processes to orchestrate instead of one | the harness's server lifecycle generalises to a list of servers; the stall watchdog carries over |
| Both instances share one host bridge | transfers are ~64 ms against ~2.6 s of decode; if contention appears it will show as TTFT variance |

## 12. Phases

| phase | work | time |
|---|---|---|
| **0** | done — instances, proxy, NCCL and the request-level handshake all work after four connector patches (`scripts/inference/split_simple/p2p_patch/`) | done |
| **1** | decode budget sweep 30 → 18 GiB, closed loop N=32, ISL 6144 / OSL 512 | ~1.5 h |
| **2** | ISL sweep {2048, 4096, 6144} at a fixed decode budget | ~40 min |
| **3** | figures + `RESULTS_DECODE.md` | ~30 min |
| **4** | optional offloading arm, only if phase 1 shows preemptions | ~1 h |

Reused: `server.py` (lifecycle, log parsing, metrics scraping, stall watchdog), `run_sweep.py`'s
driver pattern, `status.py`, `rebuild_summary.py`, and the Phase 0 PCIe calibration.
