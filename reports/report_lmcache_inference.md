# The same VRAM sweep with LMCache: same cliff, kinder collapse

Third pass over the Case A question — *how far can a serving GPU's VRAM budget shrink before
DRAM offloading stops saving you?* — with one variable changed: the DRAM tier is
**LMCache** (`LMCacheConnectorV1`) instead of vLLM's native `OffloadingConnector`. Same
machine, same model, same workload generator, same seed, same 2 QPS open loop as
[report_simple_inference.md](report_simple_inference.md), so every row here compares
directly with a row there.

**Headline: the wall does not move, but the failure mode changes sides.** LMCache pays a
higher per-retrieval cost above the wall (visible in uniform medians) and buys tighter tails;
below the wall it never preempts and never stalls — it fails like a queue, where the native
backend failed like a crash.

---

## 1. What changed, and only what changed

| | native arm (original) | lmcache arm (this report) |
|---|---|---|
| offload connector | `OffloadingConnector` (in-tree) | `LMCacheConnectorV1` + lmcache 0.4.4 (pip) |
| DRAM tier | 24 GiB pinned pool | 24 GiB pinned local-CPU cache (`max_local_cpu_size`) |
| transfer granularity | 16-token blocks | 256-token chunks |
| cache keys | block hashes (vLLM) | token-chunk hashes (`PYTHONHASHSEED=0` pinned) |
| disk | none | none — `local_disk: null`, everything in DRAM |
| invocation | `--kv-offloading-backend native` | `--kv-offloading-backend lmcache` |

Single GPU, single process, no server — the cache is in-process, exactly as the native pool
was. One caveat to keep in mind throughout: lmcache 0.4.4's compiled CUDA ops do not load
against torch 2.9.1 (`undefined symbol`), so it runs on its Python/torch fallback path.
Its per-retrieval overhead below is therefore an upper bound.

Workload, unchanged: 32 sessions × 6144-token block-aligned prefixes (24 GiB KV working
set), request = prefix + 128 unique tokens, 128 out, Poisson 2 QPS, 300 measured requests
per config, zipf-1.1 and uniform session draws.

## 2. Results

![native vs lmcache across the VRAM sweep](../figures/lmcache_offload.png)

*Columns: access pattern. Top: TTFT p50 with the band out to p95, log scale. Bottom:
output throughput against the 256 tok/s offered load. Grey = native backend (original
study), blue = LMCache. Red rings mark configs where the engine stalled mid-run — all of
them are native's; at the 18–19 GiB floor the blue line keeps flowing at 143–172 tok/s
where the grey one wedges below 45.*

### zipf-1.1 (hot set: 17 of 32 sessions)

| budget | GPU KV | native p50/p95 (ms) | lmcache p50/p95 (ms) | lmcache GPU/DRAM hits | tok/s |
|---|---|---|---|---|---|
| 30 | 13.77 | 54 / 305 | **56 / 217** | 75% / 23% | 254.8 |
| 28 | 11.77 | 55 / 302 | **57 / 204** | 73% / 25% | 254.7 |
| 26 | 9.77 | 57 / 303 | **58 / 232** | 69% / 29% | 254.8 |
| 24 | 7.77 | 59 / 315 † | 64 / 337 | 63% / 35% | 254.8 |
| 22 | 5.77 | 73 / 332 | 108 / 361 | 49% / 43% | 254.5 |
| 20 | 3.77 | 135 / 1528 | 250 / 1715 | 11% / 53% | 254.2 |
| 19 | 2.77 | 2078 / 5451 | 4804 / 11792 | 4% / 61% | 236.7 |
| 18 | 1.77 | 4856 / 10666 **stalled** | 31186 / 69639 *flowing* | 3% / 71% | **172.4** |

### uniform (no hot set — the adversarial case)

| budget | native p50/p95 (ms) | lmcache p50/p95 (ms) | tok/s |
|---|---|---|---|
| 30 | 73 / 594 | 126 / **321** | 254.5 |
| 28 | 124 / 600 | 184 / **342** | 254.5 |
| 26 | 126 / 598 | 187 / **348** | 254.5 |
| 24 | 131 / 606 | 193 / **455** | 254.5 |
| 22 | 135 / 892 | 216 / 1067 | 254.4 |
| 20 | 1205 / 4720 | 4612 / 12805 | 234.3 |
| 19 | 4952 / 10923 **stalled** (43.7 tok/s) | 18340 / 43030 *flowing* | **195.8** |
| 18 | 9675 / 23528 **stalled** (29.5 tok/s) | 52984 / 111835 *flowing* | **142.9** |

† the native zipf/b24 raw files were accidentally overwritten during this study's smoke test
and re-measured the same day; the printed numbers are the original report's, which the
re-measurement reproduced within noise.

## 3. Discussion

Both arms share everything except the miss path: same vLLM, same scheduler, same GPU
prefix cache, same 24 GiB of pinned DRAM. The backend is only consulted when a request's
prefix is *not* resident on the GPU. So every difference in the tables has to trace back
to one of two things: **what a miss costs** (the per-retrieval path), or **what a miss does
to the rest of the system** (eviction, admission, preemption). That lens explains each
observed difference.

### Where they are identical, and why

On zipf at healthy budgets the medians are indistinguishable (56 vs 54 ms at 30 GiB)
because the median request is a GPU-cache hit — the hot 17 sessions fit in the GPU tier,
and on a hit the backend's code never runs. Any DRAM backend would produce this row. The
comparison only becomes informative where misses are common: uniform access, tight
budgets, and the tails.

### The median toll: lmcache's retrieval is ~2.3× slower per miss

Measured directly from the engine logs: lmcache retrieves a full 6144-token prefix
(0.75 GB) in **~151 ms**, a consistent ~5 GB/s. The native connector's loads, instrumented
in the companion split-system work, move the same KV at ~11 GB/s (~65 ms); the machine's
pinned-DRAM ceiling is 14.5 GB/s. Three ingredients, in decreasing confidence:

1. **Compiled kernel vs Python fallback.** The native path copies blocks with a dedicated
   CUDA kernel (`ops.swap_blocks`) on its own stream. lmcache 0.4.4's compiled ops do not
   load against torch 2.9.1, so every copy runs through its pure-torch fallback —
   gather/scatter through intermediate ops rather than one kernel.
2. **Bookkeeping per retrieval.** lmcache hashes token chunks at lookup time, resolves
   them through its memory-object layer, and reassembles 24 chunks per prefix; the native
   spec reuses the block hashes vLLM's prefix cache already computed.
3. **Granularity.** 256-token chunks vs 16-token blocks changes batching of the copies
   (fewer, larger — which should *help*; the fact that it loses anyway points at 1 and 2
   as the dominant costs).

Under zipf this toll is invisible (misses are rare). Under uniform, where 42–92% of
requests take the DRAM path, it surfaces as the ~1.5× median gap (126 vs 73 ms at
30 GiB). The honest caveat cuts lmcache's way: with its compiled ops working, ingredient
1 disappears, and the toll should shrink toward parity.

### The tighter tails: lmcache's tier covers more of the working set

At 30 GiB, add up each arm's two hit rates. Native: 95.2% (zipf), 93.4% (uniform) — so
5–7% of requests miss *both* tiers and pay the ~530 ms full recompute. lmcache: **98.0%
and 97.9%** — the recompute fraction is a third of native's. p95 sits exactly where the
recompute fraction puts it: native's 594–606 ms uniform p95 is the recompute cost plus
queueing; lmcache's 321–455 ms is a slow retrieval instead. **The p95 difference is not a
faster tail path — it is fewer catastrophic misses.**

Why does the same 24 GiB cover more? The likely mechanism (hypothesis, consistent with
the hit-rate data but not separately instrumented): eviction granularity. The native pool
evicts 16-token blocks LRU, so a warm prefix can lose interior blocks piecemeal — and a
prefix match stops at the first hole, so one evicted block converts the rest of that
prefix into recompute. lmcache evicts whole 256-token chunks against a chunk-level LRU,
which keeps prefixes contiguous: a session is either resident or gone, rarely
Swiss-cheesed.

### The cliff does not move, because no cache can shrink resident KV

At 19 GiB the GPU tier holds 2.77 GiB ≈ 3.5 requests' full KV, while 2 req/s arriving
with multi-second service times need 5+ sequences *resident and growing* to keep up.
Decoding reads its KV from VRAM every step; no backend changes that. Both arms therefore
cross from "keeping up" to "over capacity" at the same budget — and their GPU-tier hit
rates track point-for-point down the sweep (75/75% at 30 GiB, 55/49% at 22, 9/11% at 20)
because the GPU tier's capacity, not the backend behind it, sets them. This is the
original report's concurrency-wall claim, now confirmed by swapping the entire offload
implementation and watching nothing move.

### Below the wall: the difference is what a miss does, not what it costs

Overloaded, the native arm enters the preemption spiral the original report documented:
evict a running request, recompute its 6144-token prefix from scratch, fall further
behind, evict again. Every preemption converts ~530 ms of *already-paid* work into new
work, so the spiral is self-feeding — throughput collapses to 28–44 tok/s and the engine
eventually stops making progress (the red rings). With lmcache the same eviction costs a
~151 ms retrieval instead of a recompute: falling behind no longer compounds, and the
sweep recorded **zero preemptions in 16 of 16 configs**. A second, softer mechanism
likely helps (hypothesis): the serialized per-request retrieval acts as admission
pacing — requests trickle into the running batch behind their loads rather than being
admitted in bursts that later collide over KV growth.

One comparison in the tables should not be read at face value: at the floor, native shows
*lower* latency numbers than lmcache (4.9 s vs 31 s p50 at zipf/18). Native's floor runs
were **aborted by the stall watchdog** — their latencies are censored snapshots of a
system that had stopped serving, while lmcache's are complete measurements of a system
that served every request. The honest floor comparison is throughput and completion:
143–172 tok/s with 300/300 requests finished, against 28–44 tok/s with a wedged engine.

## 4. Operational notes, learned the hard way

- **Version pairing is strict.** lmcache 0.5.4 fails to import against vLLM 0.15.1's
  vendored adapter (`CudaIPCWrapper` missing); 0.4.4 pairs cleanly. The pip resolver also
  silently upgraded `transformers` to 5.x during install, which vLLM 0.15.1 forbids —
  re-pin after installing.
- **`PYTHONHASHSEED=0` is mandatory**, not advisory: chunk keys are Python builtin hashes.
  Unset, keys differ per process and across restarts; in a single-process run this silently
  empties the cache on every restart, and in the split deployment (companion work) it made
  the decode node recompute everything while *looking* correct.
- The compiled CUDA ops don't load against torch 2.9.1; everything ran on the Python
  fallback. The uniform-skew median toll should be re-measured if a matching wheel appears.
- Environment drift vs the original run: transformers 4.56.x → 4.57.6, lmcache and its
  dependency tree newly installed. The zipf/b24 re-measurement doubling as a drift control
  reproduced the original numbers.

## 5. Where this fits

This is the colocated arm of a larger comparison. The companion split-system work
(prefill on GPU0, decode on GPU1) has both a bespoke P2P-transfer arm and an
LMCache-server arm standing; the decode-node VRAM sweep over those is the next experiment.
Between them, the three studies now hold the same workload against: a single GPU with a
native DRAM tier, a single GPU with LMCache (this report), and a disaggregated pair — with
the failure mode at the VRAM floor as the sharpest differentiator so far.

## Reproducing

```bash
cd scripts/inference/simple
../../../.venv-matched/bin/python run_sweep.py --backend lmcache --arms offload --tag lmcache
../../../.venv/bin/python ../../plots/plot_lmcache.py
```

Data: `output/summary_lmcache.csv`, raw per-request records in `output/raw/*_lmcache.json`,
per-config engine logs in `output/logs/*_lmcache.log`. 59.3 min wall-clock for the 16
configs on this machine.
