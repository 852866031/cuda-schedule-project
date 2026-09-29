# A second tenant in the margins: colocating a 0.5B model with the 8B decode GPU

Sixth study. The [split-inference study](report_split_inference.md) left GPU1 with
≥1.9 GiB free beside the 8B decode at the 26 GiB budget (`split_lmcache_zipf_b26_fwd_ng`:
TTFT p50 92.67 ms, 254.41 tok/s). The [inf+ft study](report_colocation_ft.md) filled
that margin with a fine-tune neighbor. This study fills it with **a second inference
model** — a Qwen2.5‑0.5B tenant — and asks what the incumbent pays and what the tenant
gets, across three ways of placing the tenant's prefill.

## 1. The goal

The 8B split stack is the fixed incumbent; every scenario keeps its configuration
bit‑identical (same budget, router, workload, seeds), and the driver asserts the
engine‑reported 8B KV grants match the baseline (9.77 GiB decode, 6.66 GiB prefill) on
every launch — so any movement in the 8B's numbers is the tenant's doing, not drift.

Two questions:
1. **Incumbent cost** — how far does the 8B's TTFT/throughput move when the tenant
   shares GPU1?
2. **Tenant service** — what does the 0.5B get, versus running alone (solo) and across
   prefill placements?

## 2. The view

The successor to the split study's Act II layout, now with a second tenant. The 8B split
is unchanged in every scenario — prefill on GPU0 writes each session's KV to its own
LMCache DRAM store (:8300); decode on GPU1 retrieves it and runs. The Qwen tenant is
dropped into the **margin GPU1 leaves free** (≥1.9 GiB beside the 27 GiB decode), with
its **own** store (:8301) so the two models share no cache state. The single most
important object in each panel is the callout in the middle: the two stores are separate
*processes*, but they drive the **same host copy engine** (CPU memcpy + host DRAM
bandwidth), and that shared path — not PCIe, not GPU compute — is where the incumbent and
the tenant actually collide (§7). What changes between scenarios is only *where the
tenant's prefill runs and how its KV reaches GPU1*:

**A — whole Qwen on GPU1.** One Qwen engine does prefill *and* decode on GPU1; on
oversubscription it spills to / reloads from its own store. GPU0 carries only the 8B.

![scenario A view](../figures/inf_coloc_view_A.png)

**B — Qwen decode-only ("prefill from an invisible GPU").** The dashed GPU0 box is a
*temporary* Qwen prefill that populates the store once and is then killed; during
measurement the GPU1 engine only retrieves KV from DRAM and decodes — it never prefills.

![scenario B view](../figures/inf_coloc_view_B.png)

**C — split Qwen (memory-deferred).** A second full split beside the 8B's: Qwen prefill
on GPU0, decode on GPU1, its own forwarding proxy (:8001). Two complete stacks — the
configuration that trips the pinned-memory guard (§7).

![scenario C view](../figures/inf_coloc_view_C.png)

## 3. The four placements

![tenant placements](../figures/inf_coloc_layout.png)

| | 8B stack | Qwen prefill | Qwen decode | Qwen KV source | client → |
|---|---|---|---|---|---|
| **solo** | absent | temp GPU0 (populate, killed) | GPU1 | DRAM store (:8301) | :8201 |
| **A** | present | **GPU1 (live)** | GPU1 | own VRAM prefix cache + DRAM spill | :8201 |
| **B** | present | temp GPU0 (populate, killed) | GPU1 | DRAM store (:8301) | :8201 |
| **C** | present | GPU0 (live) | GPU1 | forwarded + DRAM store | :8001 (proxy) |

- **solo** — tenant alone on GPU1, decode‑only. The baseline for B.
- **A — whole Qwen on GPU1.** One engine does prefill *and* decode on GPU1, with a DRAM
  offload tier so oversubscription spills to DRAM instead of recomputing. The tenant's
  prefill compute lands on the shared GPU.
- **B — Qwen decode‑only, "simulated prefill."** The tenant's prefixes are computed once
  by a *temporary* GPU0 engine that writes their KV to the DRAM store and is then killed
  ("prefill from an invisible GPU"). During measurement the GPU1 engine only *retrieves*
  KV from DRAM and decodes — it never prefills. This is the disaggregated‑decode case.
- **C — split Qwen.** A second full split (prefill GPU0 + decode GPU1 + own proxy) beside
  the 8B's. Two complete stacks. **Deferred on this box** — it doubles the host‑DRAM
  infrastructure and trips the pinned‑memory guard (§7).

**Measured: solo + A + B** (both workloads). C is defined here for completeness but is
memory‑infeasible on this box (§7) — it needs a second full split beside the 8B's.

### Why B's TTFT is reported as N/A

In B (and solo) there is **no prefill compute in the measured window** — the prefix KV
already exists in DRAM, put there by the invisible‑GPU populate step. A request sends a
whole preloaded prefix (no unique suffix), which is a 100% prefix‑cache hit, so the
engine skips prefill entirely and only decodes. There is therefore no
prompt→first‑token transition to time; a "TTFT" would be nothing but queue + KV copy.
The tenant's metric in B is **decode throughput / TPOT**, and its TTFT column is N/A.

## 4. The workloads

Both models share the reference session template; only counts and (for the decode‑only
tenant) the request shape differ.

| | 8B (incumbent) | Qwen W‑fits | Qwen W‑offload |
|---|---|---|---|
| distinct prefixes / sessions | 32 | 8 | 48 |
| prefix (fixed) | 6,144 tok | 6,144 tok | 6,144 tok |
| unique suffix | 128 | **0 (decode‑only)** | **0 (decode‑only)** |
| output (forced) | 128 | 128 | 128 |
| requests | 300 | 300 | 300 |
| arrival | 2 QPS open‑loop, Poisson | 2 QPS | 2 QPS |
| KV / token | 128 KiB | 12 KiB | 12 KiB |
| working set | 24.0 GiB | 0.56 GiB | 3.38 GiB |
| vs ~1.04 GiB VRAM KV grant | — | resident | 3.2× over → DRAM spill |

For the decode‑only tenant (solo, B) each request reuses one of the N distinct prefixes
verbatim (uniform pick, no suffix), so after a one‑time warm the prefix is resident and
every request is a pure decode. **W‑fits** (8 prefixes, 0.56 GiB) fits the VRAM grant
entirely → zero DRAM traffic during measurement. **W‑offload** (48 prefixes, 3.38 GiB)
overflows the grant → the engine keeps evicting and reloading prefixes from the DRAM
store, which is the transfer load C's/B's mechanism is meant to expose. The 8B uses seed
0 (the reference stream); Qwen uses seed 1000 so their arrivals are uncorrelated. Both
clients run concurrently over aligned windows.

## 5. Run structure

Every run: one GPU monitor (DCGM, both GPUs) → launch → assert 8B KV grants →
[decode‑only: populate + kill temp prefill] → 8B warmup (0.5 QPS) → Qwen warmup (resident
the prefixes) → scrape before‑counters (per‑engine /metrics, log markers, /proc/vmstat
swap) → **measure both clients concurrently** (8B in‑process on the exact baseline path;
Qwen as a separate process so it cannot contend for the GIL) under a combined stall
watchdog → after‑scrapes → per‑GPU telemetry over the overlap window → raw JSON + summary
row → teardown.

**Populate (B, solo):** POST each distinct prefix once to the temporary GPU0 engine
(6,144 tok = 24 full 256‑tok chunks, `save_unfull_chunk:false`), verify via its
`prompt_tokens_total` and a retrieval probe against the decode engine (must return fast),
then a PGID‑targeted kill of only that engine (heartbeat is pidfile‑gated so it does not
false‑trip). A failed verify aborts the run rather than measuring a lie.

**Isolation guarantees:** the two models get **separate LMCache servers** (8B :8300,
Qwen :8301 — content‑hash keys would otherwise collide) and separate L1 pools;
`PYTHONHASHSEED=0` everywhere; a single GPU monitor (never two DCGM openers); the 8B's
launch flags are copied byte‑for‑byte from the split study.

## 6. Results

Baseline (8B alone, b26): TTFT p50 **92 ms**, TPOT p50 **28 ms**, **254 tok/s** —
reproduced on the current box by a fresh 8B‑alone control (92.4 ms / 254.4 tok/s), so
nothing below is box drift. Qwen solo (decode‑only, alone on GPU1): **274 tok/s** at both
W‑fits and W‑offload — the tenant's own throughput is unharmed by 3.2× oversubscription
because its DRAM reloads overlap with decode.

![what the incumbent pays and what the tenant keeps](../figures/inf_coloc_result.png)

*Left: the incumbent pays on both axes — TTFT p50 (orange, the GPU0 prefill leg) and TPOT
p50 (blue, the GPU1 decode leg) as ratios to the 8B‑alone baseline (log), with absolute
8B throughput falling on the right axis. Right: the 0.5B tenant keeps 84–99% of its solo
decode throughput.* The measured points, per cell (8B TTFT / 8B TPOT / 8B tok/s ·
Qwen tok/s · Qwen KV streamed from DRAM):

- **A/fits** (whole Qwen) — TTFT **119 ms** (n=4: 95–141) / TPOT 97 ms / 236 · 271 (99%) · 0.00 GiB
- **B/fits** (decode‑only) — TTFT **157 ms** (n=4: 142–175) / TPOT 93 ms / 237 · 271 (99%) · 0.05 GiB

The fits TTFTs were each measured **4×** (the figure shows their min–max range): A/fits
mean 119 ms, B/fits mean 157 ms, with **non‑overlapping ranges** (A max 141 < B min 142).
So B/fits is a **real ~40 ms above A/fits**, not run‑to‑run noise — the tenant's
decode‑only engine queries the LMCache store on every request (its retrieval path is
always live), while A's whole‑Qwen engine mostly hits its own VRAM prefix cache and
touches the store less, so B loads the shared host path a little more. Both, though, sit
far below the offload cells: with the tenant's KV resident, the 8B is within ~30–70 ms of
its 92 ms baseline.
- **A/offload** — 312 ms / 199 ms / 212 · 261 (95%) · 8.26 GiB
- **B/offload** — 7.7 s / 370 ms / 175 · 230 (84%) · 14.6 GiB

TPOT (time‑per‑output‑token) is each request's mean inter‑token gap *including* the waits
when it is time‑shared out — the honest decode latency. The raw single‑gap ITL median
barely moves (~17 ms) because most gaps stay short; it is the per‑request average that
triples (fits) to ×13 (offload) as the incumbent's decode waits behind the tenant's.

### The real price: capacity, not the 2‑QPS latency

At the reference 2 QPS the open loop pins throughput at the offered rate while both models
still have headroom, so the incumbent looks only ~7% down — that single point *hides* the
cost. Sweeping the 8B's QPS (tenant fixed at 2 QPS) exposes it as a **capacity ceiling**:

![QPS sweep — the tenant lowers the 8B's capacity ceiling](../figures/inf_coloc_qps.png)

| 8B QPS (offered tok/s) | 8B alone | 8B + tenant |
|---|---|---|
| 2 (256) | 254 tok/s | 237 |
| 3 (384) | 317 | 245 |
| 4 (512) | **327** (saturated) | **239** (saturated) |

The 8B alone saturates near **~327 tok/s**; with the decode‑only tenant on GPU1 (no MPS)
it saturates near **~245** — the tenant consumes **~25% of the 8B's serving capacity**, and
its TTFT climbs faster with load (right panel). This is the honest way to price a colocated
tenant: not the flattering 7% at a single sub‑capacity QPS, but the QPS headroom it costs.
(This is the no‑MPS ceiling; MPS, which recovers the per‑request TPOT in §7, would be the
lever to test for recovering the *ceiling* too.)

## 7. Where the bottleneck is

A request has **two legs** — the prefill leg on GPU0 (whose finish is the TTFT, thanks to
`--forward-first-token`) and the decode leg on GPU1 (whose per‑token pace is the TPOT).
The clean, resident‑KV (*fits*) case isolates them: there the tenant's only real cost is
on the decode leg, and it is **GPU1 context serialization** — proven and undone by MPS
(§ below). The *offload* case is messier: the tenant's DRAM streaming and the box's memory
pressure pull the **prefill leg** down too, so the same GPU1 serialization ends up
back‑pressuring TTFT as well. So the two legs are the structure; a single cause — sharing
GPU1 without spatial partitioning — dominates both, with offload adding a host‑streaming
residual on top.

![the two legs of a request](../figures/inf_coloc_mechanism.png)

*One 8B request has two legs. Token #1 (TTFT) is emitted by the prefill leg on GPU0 the
instant its forward pass finishes and is forwarded by the proxy — so TTFT is a GPU0
quantity that the tenant reaches only through the shared host store path. Tokens 2–128
come from the decode leg on GPU1, shared with the Qwen decode. In the resident‑KV (fits)
case this cleanly splits — the tenant's cost is on the decode leg (TPOT). Under offload it
does not: the loaded decode back‑pressures the prefill pipeline, so the prefill leg's TTFT
collapses too (see § offload below).*

**Axis 1 — the decode leg (GPU1): TPOT + throughput.** With `--forward-first-token`,
token #1 comes from the prefill GPU; GPU1 only produces token #2 onward. So GPU1 sharing
shows up in the 8B's **TPOT and throughput, not its TTFT**. Wherever the Qwen decode is
active (A/fits, B/fits, A/offload) the 8B TPOT rises from **28 ms to ~93–97 ms** (3.3×)
and throughput drops ~6–16%; in B/offload TPOT reaches **370 ms** (13×). The tell: GPU1's
*mean* utilization *falls* when the tenant is added (sm 0.52→0.48, dram 0.46→0.39) — the
8B decode is **waiting** for GPU time, not the GPU saturating (if it were HBM-bandwidth
saturation, DRAM-active would *rise*, not fall).

**An MPS control settles the mechanism: it is context serialization.** Without MPS two
processes' CUDA contexts cannot run kernels concurrently on GPU1 — they time-slice. Rerun
under MPS (both engines become clients, verified live via the MPS server), so their
kernels co-reside on the SMs, and the decode cost nearly vanishes:

![MPS collapses the decode-side cost](../figures/inf_coloc_mps.png)

| fits cell | TPOT no-MPS (n=4) | TPOT +MPS | 8B tput no-MPS → +MPS |
|---|---|---|---|
| A/fits | 95 ms | **33 ms** | 236 → **254** |
| B/fits | 98 ms | **34 ms** | 237 → **254** |

MPS drops TPOT from ~3.4× the 28 ms baseline back to ~1.2× and recovers throughput to the
254 tok/s baseline, for both scenarios — and *without* hurting TTFT (still ~120–130 ms).
Since the decode runs at only ~5% SM occupancy, there is ample room for two decodes to
co-reside; the large no-MPS penalty despite that spare capacity is exactly the signature
of coarse whole-context serialization, and MPS removing it is the proof. So the axis-1
cost is **GPU1 context serialization, and MPS is its mitigation** — the same lever the
[fine-tune study](report_colocation_ft.md) used. TPOT (which counts the waits) is what
moves; the raw single-gap ITL median does not.

**Axis 2 — the prefill leg (GPU0) via the host store path: TTFT.** The 8B TTFT is the
prefill leg alone (verified: the proxy sends token #1 the instant GPU0 finishes its
forward pass). It moves with the tenant's **DRAM streaming volume** (0→0.05→8.3→14.6 GiB
gives 92→175→312→7700 ms), because the prefill leg's host‑side store step (retrieve on a
prefix‑cache miss, store the computed KV back) contends on the shared host copy machinery.
Two clarifications settle the *fits* case (A +27 ms, B +65 ms over the 92 ms baseline,
means of n=4, with ~zero streaming):
- **Not GPU0 compute.** The 8B‑alone control reproduces 92 ms with the identical model, so
  the prefill compute is untouched — the extra time is entirely the host‑side store step.
- **Not host‑DRAM bandwidth.** The 8B decode already streams ~225 GB from the store in the
  baseline at 92 ms; W‑fits adds ~0 host traffic (Qwen KV is VRAM‑resident). No new
  bandwidth wall is introduced. By elimination the residual is a **host‑side effect** on
  the GIL‑bound Python‑fallback LMCache path (server + proxy) — the box bursts to 99% user
  CPU during the run — a *measurement‑stack artifact* rather than a fundamental GPU or
  memory limit. That B (+65) exceeds A (+27) fits this: B's decode‑only engine hits the
  store on every request while A mostly hits VRAM, so B adds more host‑path load.
- **What the taskset test showed (and didn't).** Pinning the 8B store server + proxy onto
  dedicated cores 0–5 with the Qwen processes excluded (kept on 6–13) — affinity verified
  live — moved a control B/fits of 182 ms to 152 ms, but 152 ms sits **inside** B/fits's
  normal 142–175 range, so the effect is not distinguishable from run‑to‑run noise at n=1
  each. So CPU isolation *from the tenant* did **not** clearly help — consistent with the
  contention coming from total host‑CPU load (the 8B's own engine threads still span those
  cores), not from the tenant's processes specifically. The host‑side sub‑mechanism is
  therefore established only as *host‑side, non‑bandwidth*; the finer attribution is left
  open rather than overclaimed.

**The offload cells: a prefill‑leg collapse, not a clean second axis (correcting an
earlier draft).** In offload the 8B TTFT collapses to *seconds* — on a clean box B/offload
measured TTFT p50 **11.1 s**, and it is genuinely the prefill leg (proxy `leg1` == TTFT ==
11.07 s), whose effective throughput has fallen to ~8000 tok/s, below the ~12.5k tok/s the
2 QPS offered load needs, so requests queue without bound. An earlier draft called this a
pure host‑copy‑bandwidth saturation (a "second axis" MPS could not touch). **The MPS
experiment refutes that:** rerun under MPS, B/offload's TTFT dropped to **348 ms** and its
prefill‑leg throughput recovered to ~10.8k tok/s — a GPU1 co‑residency fix removing most
of a GPU0 prefill‑leg queue. So the collapse is *not* mainly a fixed host‑bandwidth wall;
much of it rides on the same GPU1 serialization the fits cells showed, back‑pressuring the
prefill pipeline. **But the offload comparison is confounded:** at ~57 GiB it sits at the
box's RAM edge, and the no‑MPS control paged in 27× more (36k vs 1.3k swap‑in pages) than
the MPS run, because run‑ordering changes the swap state. So on this box the offload
collapse entangles GPU‑serialization back‑pressure, host‑copy streaming (14.6 GiB), and
swap pressure, and they cannot be cleanly separated — the honest statement is that offload
is far worse than fits and MPS helps it a lot, with a residual cost (offload+MPS still
348 ms TTFT / 169 ms TPOT / 13 s p95) that the streaming and memory pressure impose. The
fits result is the clean, unconfounded one.

**Earlier faulty B (for the record).** A first version ran the decode engine with prefix
caching *off*, re‑fetching the full 72 MiB prefix every request even in fits (21 GiB of
needless copies) and collapsing the 8B. The fix — prefix caching on + whole‑prefix reuse —
keeps the fits set VRAM‑resident (axis‑2 traffic ~0) and lets only the offload overflow
stream. Nothing is ever recomputed (counters: 300 of 1.84 M Qwen prompt tokens computed;
the rest cached), and it is not a PCIe limit (independent x8 links; ~5 GB/s copies under a
~16 GB/s link).

**Telemetry attribution caveat.** DCGM is per‑GPU, not per‑process: on GPU1 the SM‑active,
occupancy, and DRAM‑active counters blend the 8B decode and the Qwen tenant. Every
per‑model claim rests on client‑side latencies and per‑engine /metrics deltas; the
per‑GPU numbers bound the *combined* load. Swap counters (pswpin/pswpout) are recorded
per run to rule host‑DRAM thrash in or out before attributing an 8B slowdown to GPU
contention.

**Why C is deferred.** C stands up a *second* full split — another prefill engine, another
store, more pinned L1s — pushing the host into the kernel's reclaim zone; in the first
pass its offload cell drove available RAM below the 4 GiB mem‑guard floor and the guard
killed the engines (as designed — pinned overcommit has frozen this box before). The
pinned total that governs the freeze (8B 2×6 + Qwen 2×1 = 13 GiB) is safe; it is the
swappable store growth that is not, so C is left for hardware with more host RAM.

## 8. Colocating inference vs colocating fine-tuning

This is the sixth study; the [fifth](report_colocation_ft.md) put a **fine-tune** neighbor
in the same GPU1 margin, at the same b26 budget, against the same reference workload. Side
by side, the two studies answer one question: *what kind of neighbor can the decode GPU
actually afford?*

![inference vs fine-tuning neighbor](../figures/inf_coloc_vs_ft.png)

*Left: decode TPOT × the decode-alone baseline (log). The fine-tune arms ran under MPS, so
the fair comparison is MPS‑vs‑MPS: a resident (fits) second‑model tenant with MPS costs
**1.2×**, right alongside the fine‑tune arms' 1.2–1.7× (bracket). Without MPS the tenant
serializes (3.5×); MPS removes that. Offload keeps a residual (6× even with MPS). Right:
why — decode leaves ~95% of its SMs idle, so MPS lets any resident neighbor use them; the
inference tenant reaches the binding HBM bandwidth only when it streams KV (offload).*

The first cut of this section drew an unfair contrast — a *no‑MPS* inference tenant
(3.3–13×) against the *with‑MPS* fine‑tune arms (1.2–1.7×) — and concluded a second decode
was near‑worst‑case. The MPS control corrects it. Three points:

- **The decode GPU's idle resource is compute; MPS is what unlocks it.** Decode runs at
  **SM occupancy under 5%** while **DRAM‑active sits at 45–62%** — memory‑bound, SMs nearly
  empty. But two processes cannot use those idle SMs concurrently *without MPS* — their
  contexts time‑slice. This bites **both** neighbors: the fine‑tune study's un‑MPS'd
  time‑slice arm collapsed decode (TPOT 1193 ms), and the inference tenant's no‑MPS arms
  serialize to 3.5× (fits) / 13× (offload). MPS is the shared enabler, not a fine‑tune
  detail.
- **With MPS, a resident right‑sized tenant is cheap either way.** A fine‑tune neighbor
  costs **+23% TPOT** at a 10% SM cap; the resident second‑model tenant costs **+20%**
  (34 vs 28 ms) and recovers full throughput. When the neighbor's working state fits in the
  margin, the ~95%‑idle SMs absorb its kernels and the incumbent barely notices.
- **The inference tenant's distinctive cost is KV streaming, not "being inference."** Its
  one hazard the fine‑tune neighbor lacks is that its KV can oversubscribe the VRAM margin
  and spill to DRAM (offload); that streaming loads the host path and leaves a residual MPS
  cannot remove (offload+MPS still ~6× TPOT). A fine‑tune adapter's state stays resident, so
  it never triggers this.

So the corrected lesson refines the fifth study's §3.2 thesis. It is **not** "compute‑dense
good, a decode tenant bad." It is: **MPS is mandatory** for either neighbor to share the
decode GPU without serializing; given MPS, a **resident, right‑sized** tenant — fine‑tune or
a second model — costs ~20% and is very affordable; the real thing to avoid is letting the
tenant's KV **oversubscribe the VRAM margin and stream**, which is a capacity‑planning
problem (size the tenant to fit, per §3), not an inherent property of colocating inference.

## Reproducing

```bash
# this study: solo baseline + A (whole Qwen) + B (decode-only from DRAM), both workloads
.venv/bin/python scripts/inf_inf_coloc/coloc2_sweep.py --scenarios solo A B --wls fits offload
# 8B-alone control (current-box baseline, via the split harness)
.venv/bin/python scripts/inference/split_simple/disagg_sweep.py --stack lmcache \
    --skew zipf --budgets 26 --warmup-qps 0.5 --forward-first-token --max-inflight 999 \
    --name-suffix _ctrl --tag split_ctrl

# figures (run from the repo root; view them before believing them)
.venv/bin/python scripts/plots/plot_inf_coloc_view.py         # §2 system view
.venv/bin/python scripts/plots/plot_inf_coloc_layout.py       # §3 memory placement
.venv/bin/python scripts/plots/plot_inf_coloc_mechanism.py    # §7 two-axis mechanism
.venv/bin/python scripts/plots/plot_inf_inf_coloc.py          # §6 results
.venv/bin/python scripts/plots/plot_inf_vs_ft.py              # §8 inf-vs-ft comparison
```

Data: `output/summary_inf_coloc*.csv` (main solo+B, plus `_A`, and the `_iso`/`_cpu`/`_legs`
diagnostic re-runs), raw per-request records in `output/raw/infc_*.json`, per-second GPU
telemetry in `output/gpumon/`, engine logs in `output/logs/`. The 8B-alone control is
`output/raw/split_lmcache_zipf_b26_ctrl.json`. `scripts/inf_inf_coloc/` is self-contained:
`workload.py`/`client.py`/`lmc_proxy.py`/`lmc_server_main.py` are copies, and `coloc2_*.sh`
launch/stop/heartbeat drive both stacks with a single teardown.
