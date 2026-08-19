# Progress Log

Append-only. Newest entries at the bottom. Each entry: what step, what happened, what's next.

---

## 2026-08-18 17:10 — Phase 0 start: branch setup + plan

- Cloned `852866031/cuda-schedule-project` into this dir, branched `vram-limit` off `colocation` (`0baec85`).
- Cleared the branch: removed `colocator/`, `docs/`, and the Orion-era markdown/html at root. All of it
  is still on `colocation` — nothing lost.
- Probed the machine before designing anything. Two findings that drove the plan:
  - **PCIe is Gen4 x8** (`Host Max: 4`, width `8x`), not Gen5 x16. ~16 GB/s theoretical.
    The cards can do Gen5 x16; this host cannot. Offload cost is ~4x what a spec-sheet guess would give.
  - vLLM `0.15.0rc2.dev` already ships `--kv-offloading-size/-backend` and `--cpu-offload-gb`,
    so no external install is needed for the primary sweep.
- Wrote `PLAN.md` (committed `4d220b1`).
- Decision (user): **vLLM native offloading** for the sweep; LMCache deferred. Native is a pure
  DRAM<->VRAM path with no extra tiers to confound the measurement, and needs no install against
  this nightly-torch env.

**Next:** Phase 0 calibration — PCIe pinned-transfer bandwidth, and verify a 24 GiB pinned host
allocation is actually possible (`ulimit -l` is only 7.5 GiB, so this needs proving, not assuming).

## 2026-08-18 17:15 — Phase 0 done: PCIe calibrated

Wrote and ran `scripts/calibrate_pcie.py` -> `output/calibration_pcie.json`.

| measurement | result |
|---|---|
| pinned H2D | **14.47 GB/s** (91% of Gen4 x8 theoretical) |
| pinned D2H | **14.33 GB/s** |
| pageable H2D | 14.28 GB/s at 1 GiB, but only 8.99 GB/s at 1 MiB |
| PCIe link under load | ramps gen 1 -> **gen 4 x8**, confirming the earlier reading |
| **24 GiB pinned host alloc** | **succeeds** in 2.95 s despite `ulimit -l` = 7.5 GiB, and H2D holds at full speed with the pool held |

**t_load for an 8192-token prefix (1 GiB of KV) = 74 ms.** That is the constant the whole
Case A curve gets read against.

The pinned-alloc result matters: it was the one risk that could have forced the CPU pool down
to 20 GiB and shrunk the whole experiment. It's clear.

## 2026-08-18 17:30 — Harness built; three environment blockers hit and fixed

Wrote `scripts/workload.py`, `scripts/client.py`, `scripts/server.py`, `scripts/run_sweep.py`.

Workload shape settled: **32 sessions x 6144 tokens = exactly 24.0 GiB of KV.** Had to move
off the plan's 24x8192 because Llama-3 has an 8192-token context window and prefix + suffix +
output must fit inside it. 6144 + 128 + 128 = 6400. Zipf-1.1 puts 90% of requests on 17
sessions (12.75 GiB), so the GPU tier goes from holding ~88% of requests' prefixes at the top
of the sweep to ~42% at the bottom -- a real dynamic range.

Three blockers, all environmental, none in the experiment design:

1. **`GLIBCXX_3.4.32 not found`** — the conda env ships libstdc++ 6.0.29, `vllm._C` needs
   3.4.32. Fixed by preloading the system libstdc++ (3.4.33) in the server's env rather than
   mutating the conda install.
2. **`cannot import name 'Gemma3Config'`** — base conda env has transformers 4.45, vLLM 0.15
   requires >=4.56. vLLM was simply unusable in that env. Fixed with a `.venv` created with
   `--system-site-packages` and transformers 4.57.6 installed into it, so the base env other
   projects depend on is untouched.
3. **`Connector OffloadingConnector does not support HMA`** — needs
   `--disable-hybrid-kv-cache-manager`. Now passed on **every** arm, offload or not, so both
   arms share one allocator and stay comparable.

Also open, and the reason the sweep currently runs eager: **`torch._dynamo.exc.Unsupported`**
compiling vLLM's parameter `__torch_function__`. The local vLLM build is from Feb 2026 but
sits on a **March 2025 torch nightly** — ~11 months apart. `--enforce-eager` gets past it. A
matched `vllm==0.15.1` stack is downloading into a separate `.venv-matched` in the background;
if it works, the sweep reruns with torch.compile + CUDA graphs for realistic absolute numbers.

**Smoke test passed** (`run_sweep.py --smoke`, 2 configs, 6 GiB working set):
- server up in 20 s, GPU KV **7.77 GiB at a 24 GiB budget** — matches the plan's predicted 7.7
- offload arm wrote **6.98 GB GPU->CPU in 0.488 s = 14.3 GB/s**, i.e. exactly the D2H number
  from Phase 0. The connector achieves full PCIe bandwidth.
- both arms identical on TTFT (52 ms) and throughput (207 tok/s) — correct, since a 6 GiB
  working set fits in 7.77 GiB of GPU KV and nothing needs to move.

**Next:** two things found in the smoke logs need explaining before the real sweep —
a 40 s zero-throughput stall in the offload arm's warmup, and a metric label case mismatch
(`GPU_to_CPU`, not `gpu_to_cpu`) that made the CSV's transfer columns read 0.

## 2026-08-18 17:45 — Investigated the "stall"; it isn't one

Wrote `scripts/probe_offload_stall.py` to test two hypotheses for the 77 s warmup: a one-time
cost (lazy pinned-pool allocation) vs a per-store cost (stores blocking the engine loop).

First run of the probe was wrong and I fixed it: I sampled cold requests from a uniform draw,
which repeats sessions, so 5 of 8 "cold" requests were actually cache hits. `warmup_requests()`
gives exactly one request per distinct session; the probe now uses that.

Corrected result — **offloading costs nothing on the store path**:

| | offload | no-offload |
|---|---|---|
| sequential cold prefill (6272 tok), median | 0.61 s | 0.60 s |
| 8-request concurrent burst, wall | 4.46 s | 4.45 s |
| burst TTFT p50 | 1.278 s | 1.496 s |
| stored per request | 0.77 GiB in **0.057 s (13.5 GB/s)** | — |

Stores fully overlap with compute: 0.057 s of PCIe per request adds 0.01 s of wall time.
The 13.5 GB/s matches the Phase 0 D2H ceiling of 14.33 GB/s.

So the 77 s warmup did not reproduce. Most likely explanation: in the smoke run the offload
arm was the first server started after the 15 GB model was read from disk, and allocating the
pinned pool forced the kernel to reclaim page cache. Consistent with the engine sitting at
zero throughput with only 39% KV used and 0.39 s of actual PCIe time in the window.

Not a threat to the measurements, because **the pool is allocated during warmup, and warmup is
excluded from every reported number**. Worth watching at the real 24 GiB pool size, which is
3x larger. Also fixed a real bug this uncovered: vLLM labels the transfer metrics `GPU_to_CPU`,
not `gpu_to_cpu`, so the CSV's transfer columns were silently reading 0. Now matched
case-insensitively.

**Next:** pilot at the two extremes (30 GiB and 18 GiB budgets) with the full 24 GiB working
set, to validate the 24 GiB pool at scale and preview the curve before committing to the
full matrix.

## 2026-08-18 17:55 — Pilot at budget 30 GiB: the effect is real and large

Ran the two extremes (30 GiB and 18 GiB budgets) with the full 24 GiB working set, 100
requests, Zipf-1.1. First point in:

| budget 30 GiB (GPU KV **13.77 GiB**, as predicted 13.7) | offload | no-offload |
|---|---|---|
| TTFT p50 | **59 ms** | 183 ms |
| TTFT p95 | **470 ms** | 1103 ms |
| GPU prefix hit rate | 68.6% | 68.6% |
| DRAM (external) hit rate | **25.3%** | n/a |
| KV pulled over PCIe | 19.4 GiB | 0 |
| output tok/s | 234 | 234 |

**3.1x better TTFT p50, 2.3x better p95** — at the *most generous* budget in the sweep, where
offloading should matter least. The GPU hit rate is identical across arms, which is the
control working exactly as intended: same GPU tier, same hits; the only difference is what
happens on the 31% that miss. Offloading converts a quarter of all queries from a ~1 s
recompute into a ~74 ms PCIe read.

Throughput is identical (234 tok/s) in both arms, and that is expected rather than a null
result: at a fixed 2 QPS open loop the system is far from saturation, so throughput is set by
the arrival rate, not by capacity. **TTFT is the signal in this regime.** A saturation sweep
(rising QPS at a fixed budget) would be the right way to make throughput the dependent
variable — noting it as a follow-up.

Also wrote `scripts/plot_case_a.py` and `README.md` (layout, environment, how to reproduce
any single point, and the three environment fixes needed to run vLLM on this box).

**Next:** wait for the 18 GiB point, then launch the full 8-budget x 2-arm x 2-skew matrix.

## 2026-08-18 17:50 — The offload path deadlocks at the tightest budget

The 18 GiB point (GPU KV **1.77 GiB**, offload arm) **wedged the engine**. Evidence:

- last engine log line: `Running: 0 reqs, Waiting: 14 reqs, GPU KV cache usage: 85.0%`,
  `Avg prompt throughput: 0.0 tokens/s`
- the engine then stopped logging **entirely** — vLLM prints stats every ~10 s while requests
  are in flight, so this is the engine loop blocked, not an idle engine
- GPU utilisation 0%, held for the ~7 minutes before I killed it
- external (DRAM) hit rate had climbed to 49% just before the freeze, so the offload path was
  active and doing real work right up to the moment it stopped

Couldn't get a stack: `kernel.yama.ptrace_scope=1` blocks py-spy against a non-descendant, and
the process needed killing to free the GPU. The mechanism is most likely a circular wait for
blocks — at a 1.77 GiB tier each 6272-token request needs 0.77 GiB, so ~2.3 requests fill the
whole tier, and with loads in flight holding block references there is nothing left to evict.

Note the shape of it: **the tighter the VRAM, the more offloading you need, and the more likely
this becomes.** It bites exactly where the feature is supposed to earn its keep.

Handled it as a result rather than an error. Added a **stall watchdog** to `run_sweep.py`:
if the server log goes untouched for `--stall-timeout` (default 180 s) while the client still
has requests outstanding, the server is killed, in-flight requests fail immediately, and the
config is recorded with `hung=true` in the CSV. Also cut the per-request client timeout from
900 s to 300 s. Without this, one wedged config costs 15+ minutes of wall clock; with it,
3 minutes and a labelled data point. Hung rows stay in the CSV as data and are excluded from
the plots.

Cleanup note: a wedged run leaves the engine-core child alive holding 19 GiB of VRAM after the
parent dies — `pkill -9 -f VLLM::EngineCore` is needed to actually free the card.

**Next:** re-running 18 GiB with the watchdog, both arms, to confirm (a) the watchdog fires
cleanly and (b) whether the no-offload arm survives where the offload arm doesn't. Then bisect
for the budget where offloading stops being viable.

## 2026-08-18 17:52 — The 18 GiB hang is deterministic, and the toolchain fight is settled

Re-ran 18 GiB with the watchdog. The offload arm retraced the *exact same* trajectory as the
first attempt — same engine throughput figures (8152.6 tok/s), same queue depths (Running 3 /
Waiting 16), same external hit rate (48.6%) at the same point in the run. Same seed, same
path, same wedge. That rules out a race or a transient: it is a **deterministic deadlock** at
this tier size, which makes it reproducible and therefore reportable.

Warmup at 18 GiB did complete this time (18.6 s for 32 sessions), so the wedge is specific to
the measured phase, where 100 requests arrive at 2 QPS against a 1.77 GiB tier.

Separately, the matched-stack install that had been crawling for an hour: the cause was this
box's pip config, which sets `extra-index-url = pypi.ngc.nvidia.com` — a host with no working
DNS here. pip consulted it for **every** package and burned ~30-60 s of retries each time
(644 retry lines to show for ~13 MB downloaded). `--extra-index-url` on the command line does
not override it; `PIP_CONFIG_FILE=/dev/null` does. Verified with a one-package test: 644
retries -> 0. Reinstalling `vllm==0.15.1` cleanly now.

This is worth recording for anyone reusing this machine: **any slow pip install here is
probably the NGC extra-index, not the network.**

**Next:** watchdog outcome for the 18 GiB offload arm, then the no-offload arm at the same
budget (the interesting question: does the control survive where offloading doesn't?).

## 2026-08-18 17:57 — 18 GiB refined: not a clean deadlock, a collapse *then* a wedge

The watchdog fired exactly as designed (190 s of no engine progress -> server killed), and the
partial data it saved changes the story in a useful way. The offload arm at 18 GiB did **not**
wedge from the start:

| budget 18 GiB, offload arm | |
|---|---|
| requests completed before the wedge | **85 / 100** |
| TTFT p50 | **6033 ms** (vs 59 ms at 30 GiB — **102x worse**) |
| TTFT p95 | 12907 ms |
| output throughput | 33 tok/s (vs 234) |

So the sequence is: performance collapses by two orders of magnitude, *then* the engine wedges.
The deadlock is the endpoint of the degradation, not a separate cliff. Per the user's call,
18 GiB stays in as the sweep's lower bound and gets reported with its partial results — it
still produces data, and the collapse is the point.

Fixed a bug this exposed: when the watchdog killed the server, the final `/metrics` scrape had
nothing to talk to, so every counter delta came out negative (`preemptions = -7`). Metrics are
now scraped **before** the kill. Also fixed the SSE parser in `client.py`: it treated a chunk
whose detokenized text was `""` as "no token", which silently dropped 2 of 100 requests in
*both* arms (deterministic, indices 30 and 89). Random-token prompts make the model emit runs
of incomplete-UTF-8 byte fragments that render empty but are real tokens; vLLM sends one chunk
per token regardless, so chunk presence is now the signal, not text truthiness.

Also traced the slow-pip mystery to its source for the record: `~/.pip/pip.conf` and
`~/.config/pip/pip.conf`, both stamped "autogenerated by NVIDIA PyIndex", add
`pypi.ngc.nvidia.com` as an **extra** index — consulted for every package, and unresolvable
from this box.

**Next:** launch the full matrix — 8 budgets (30 down to 18) x 2 arms x 2 skews, 300 requests
each, zipf first so partial results are useful early. ~2-2.5 h.

## 2026-08-18 18:05 — 18 GiB control arm, and the full sweep is running on a matched stack

**Both arms at the 18 GiB floor** (GPU KV 1.77 GiB, 13.6x oversubscribed):

| | TTFT p50 | TTFT p95 | preemptions | outcome |
|---|---|---|---|---|
| offload | **6.0 s** | 12.9 s | — | **wedged** at 85/100 |
| no-offload | **20.1 s** | 45.7 s | **28** | survived, 98/100 |

The control is 3.3x *worse* on latency than offloading, and its 28 preemptions name the
mechanism: with a 1.77 GiB tier the scheduler keeps evicting running sequences and recomputing
them from scratch. So at the floor the choice is **fast but fragile** (offload: 6 s, then a
wedge) versus **stable but unusable** (no-offload: 20 s TTFT, 45 s at p95). Neither is a
serving configuration anyone would ship; that is the real answer to "how far can you shrink
it".

**Toolchain upgraded.** `vllm==0.15.1` + `torch 2.9.1+cu128` installed cleanly in
`.venv-matched` once `PIP_CONFIG_FILE=/dev/null` took the NGC index out of the path. Verified:
`vllm._C` imports, sm_120 supported, same `--kv-offloading-*` flags, and a server comes up
**with torch.compile and CUDA graphs enabled** (startup 90 s vs 24 s eager). It reports GPU KV
7.771 GiB at budget 24 against eager's 7.770 — the same memory split, so the two stacks are
directly comparable and the eager caveat is now gone from the headline numbers.

**Full sweep launched** on the matched stack: 8 budgets (30 -> 18) x 2 arms x 2 skews = 32
configs, 300 requests each, zipf first. Expect ~2.5-3 h.

Housekeeping: I had accumulated nine redundant polling loops watching for the same events.
Killed them. Lesson for this session's workflow — one waiter per event, not one per check.

## 2026-08-18 18:15 — Why throughput is flat, and a metrics gap on the release build

User asked why throughput barely moves between arms. Worked it out with numbers rather than
assertion, and the answer is that **throughput here measures the load generator, not the
server**:

- offered: 2.0 req/s x 128 tokens = 256.0 tok/s
- measured: 254.73 tok/s = **99.5% of offered**

With `ignore_eos=True` every request emits exactly 128 tokens, so below saturation tokens-out
is forced to equal requests-in x 128. It only becomes informative when the server *cannot*
keep up, which is precisely what the 18 GiB floor showed (123 and 33 tok/s against 256
offered).

The follow-on question — shouldn't all that data movement make compute less efficient? — has
the opposite sign, and the run data shows it. For the 377,824 tokens served from DRAM in
`zipf_b30_offload`:

| | |
|---|---|
| GPU prefill avoided | **31.7 s** |
| PCIe time paid instead | **3.4 s** |
| PCIe utilisation | 46.1 GiB / 151 s = 0.33 GB/s = **2.3% of the link** |

Offloading **substitutes** a cheap transfer for expensive compute; the GPU does *less* work,
not more. The 9.3x ratio here matches the 9.5x recompute-vs-fetch ratio from calibration,
which is a nice independent confirmation at whole-run scale. The cost lands on TTFT, where the
transfer is on the critical path — not on throughput.

To make throughput a genuine dependent variable would need a **rising-QPS sweep at a fixed
budget** (PCIe would need ~43x this traffic to saturate, roughly 26 req/s of pure misses).
Logged as the natural second experiment.

**Metrics gap found:** vLLM **0.15.1 release does not implement KVConnectorStats for
OffloadingConnector** — `vllm:kv_offload_total_bytes/_time` are simply absent, though the
Feb-2026 dev build had them. So the sweep's `kv_load_gib` column reads 0 on the matched stack.
Recoverable: the `external_prefix_cache_*` counters are in **tokens** (queries came to exactly
300 x 6272), so KV volume is derivable analytically. What is genuinely lost is transfer
*time*, i.e. achieved bandwidth — already measured on the dev build (13.5-14.3 GB/s, matching
calibration), so that evidence is banked rather than missing.

Added `scripts/rebuild_summary.py`, which regenerates the summary CSV from `output/raw/*.json`
through the current `derive_row()`. A derivation fix now costs a rebuild, not a re-run — which
matters because the sweep in flight is using the code as it was at launch.

## 2026-08-18 18:45 — The knee is found, and the scope narrowed to offload-only

**User narrowed the scope:** the experiment is about how *offloading* performance changes with
VRAM size, so the no-offload control is only needed once, as a full-VRAM reference — not at
every budget. Switched the remaining matrix to offload-only plus one uniform reference. The
5 zipf controls already collected (b30/28/26/24 + b18 from the pilot) are kept: they were
already paid for and they let the zipf half answer both questions.

**The zipf offload curve, complete down to the knee:**

| budget | GPU KV | TTFT p50 | TTFT p95 | DRAM hit | preempt |
|---|---|---|---|---|---|
| 30 | 13.77 | 54 | 305 | 20% | 0 |
| 28 | 11.77 | 55 | 302 | 22% | 0 |
| 26 | 9.77 | 57 | 303 | 27% | 0 |
| 24 | 7.77 | 59 | 315 | 32% | 0 |
| 22 | 5.77 | 73 | 332 | 40% | 0 |
| **20** | **3.77** | **135** | **1528** | 51% | **4** |

**Headline:** p95 is flat at ~305-332 ms while GPU KV falls from 13.77 to 5.77 GiB — a **58%
cut in KV for a 9% p95 cost** — then jumps **4.6x in a single step**. The DRAM tier absorbs the
slack smoothly (20% -> 40% hit rate) right up to the cliff.

**I made a prediction before these points ran, and it was wrong by one step.** I estimated from
Little's law that 3.7 requests in flight x 0.77 GiB = 2.8 GiB minimum, so b20 (3.77 GiB) would
be tight-but-fine and b19 (2.77 GiB) would break. The onset is at b20. The mistake: I used
*average* concurrency, but arrivals are Poisson, so bursts of 5-6 in flight are routine and
need ~4.6 GiB. The wall is hit when *peak* concurrency exceeds capacity, not the mean.
Mechanism was right -- preemptions appear exactly at the knee, which is the concurrency
signature, not a cache-miss signature. Threshold needed a burst factor.

**Practical answer taking shape:** with this workload you can cut VRAM from 30 to ~22 GiB
(weights + ~7 GiB of KV) at almost no cost, and below that it degrades fast.

**Next:** b19 and b18 offload finish the zipf curve past the knee, then the uniform curve
(8 budgets) and its full-VRAM reference. Then figures + RESULTS.md.

## 2026-08-18 19:25 — Finetuning experiment designed and harness written

User asked for a LoRA finetuning version of the same question: throughput vs available VRAM.
Wrote `PLAN_FINETUNE.md` and `scripts/finetune_sweep.py` while the inference sweep finishes.

**Why training flips the verdict on weight offloading.** The inference study showed weight
streaming is hopeless for serving (297 ms/token at 4 GiB offloaded). Training amortizes the
same transfer over `B x S` tokens instead of `B`:

- streaming all 14.96 GiB costs **2.22 s/step** (fetched for forward, again for backward)
- compute at 8192 tokens/step is **2.63 s**
- **break-even ~6,900 tokens/step** -- above it, streaming can hide behind compute

**Memory accounting for LoRA r=16 on attention (q,k,v,o), all 32 layers:**

| | |
|---|---|
| trainable | 13.6 M params (0.17% of model) |
| adapters + grads + Adam fp32 | **208 MB** |
| frozen base weights | 14.96 GiB |

That 208 MB is why LoRA is the right vehicle: a full finetune needs ~128 GB of optimizer state
against 60 GiB of DRAM and simply does not fit. LoRA removes the optimizer from the experiment,
leaving a clean split between frozen weights and activations, and frozen weights are read-only
so streaming needs no write-back.

Gradient checkpointing is **mandatory**, not a tuning choice: without it, activations are
4.0 MB per token (the three 14336-wide MLP tensors dominate), so 8192 tokens = 32 GiB on its
own. With it, 4.43 GiB.

**Batch size:** my arithmetic says **4** (25.5 GiB) with 8 OOMing at 33.9 GiB — but that hangs
entirely on HF's `logits = logits.float()`, which makes the logits tensor exist in bf16 *and*
fp32 against a 128,256 vocab. At B=8 that is 11.7 GiB, larger than all 32 layers of
checkpointed activations combined. Too implementation-dependent to trust, so **phase 0
bisects it empirically** over descending powers of 2.

**Tooling — good news, all installed** (`peft 0.15.0`, `accelerate 1.5.2`, `deepspeed 0.16.4`).
The out-of-the-box knob is an **explicit per-layer `device_map`**: put the last N of 32 layers
on `"cpu"` and accelerate's hooks stream them in per forward and per backward. Exactly the
"offload N layers" knob requested, zero custom code. Its limitation is that it is naive
fetch-on-demand with no prefetch overlap, so it measures the pessimistic bound -- DeepSpeed
ZeRO-3 is the prefetching comparison, and the gap between them is precisely the value of
overlap.

**Predictions recorded before running:** with prefetch, offloading all 32 layers should cost
<20% throughput at 8192 tok/step; naive should be roughly linear, ~1.8x slower at full offload.
If the naive path degrades linearly and DeepSpeed does not, that is a finding about tooling
rather than about offloading.

**Next:** inference sweep is on its last configs (`uniform_b18_offload` wedging as zipf did,
watchdog pending). Then figures + RESULTS.md, then run the finetuning sweep.

## 2026-08-18 19:35 — Inference sweep complete; figures and RESULTS.md written

All 22 configs done. Three figures in `figures/`, report in `RESULTS.md`.

**The headline:** VRAM can be cut **30 -> 22 GiB (58% less KV) for a 9% p95 cost**, then it
collapses two orders of magnitude within two steps.

**The last config delivered the study's sharpest contrast.** At full VRAM, offloading's value
depends entirely on access skew:

| at 30 GiB | no offload | offload | gain |
|---|---|---|---|
| zipf-1.1 | 61 ms | 54 ms | **1.1x** |
| uniform | 559 ms | 73 ms | **7.7x** |

Under zipf the GPU cache already holds the hot set, so offloading is nearly redundant. Under
uniform there is no hot set, the cache thrashes, and offloading is worth 7.7x. Both skews
collapse at the *same* budget though, because the concurrency wall is set by request size and
arrival rate, not by locality.

**The wall analysis held up.** Computed threshold 5.36 GiB (Poisson p95 of 7 in flight x
0.766 GiB); observed last-clean 5.77 GiB, first-broken 3.77 GiB. Both skews collapse across
that line. The clinching diagnostic: **ITL improves** (13.3 -> 11.8 ms) as everything else
collapses, because fewer sequences run concurrently -- proving the failure is in *admission*,
not in token generation. A cache or bandwidth wall would not look like that.

Report structure follows what the user asked for: setup, what affects TTFT vs throughput, the
four walls with computed thresholds and why two bind and two do not, results, limitations,
follow-ups. Wrote the limitations honestly -- including that p50 is a weak signal by design
(75% of tokens hit free), that the 128-token suffix makes relative gains an upper bound, that
throughput was never a dependent variable, and that every conclusion is a statement about
PCIe Gen4 x8 rather than about offloading as an idea.

**Next:** finetuning phase 0 -- find the largest power-of-2 batch that fits with no offloading.

## 2026-08-18 19:50 — Report revisions, and a real bug in the finetuning harness

**Report changes per user:** replaced "wedged" with "stalled" throughout `RESULTS.md` and the
figures; removed the no-offload curve from the headline figure (the question is how offloading
degrades with VRAM, and the control is reported in the results table instead); added a
**throughput row** to the main figure; reversed the walls-figure x-axis so generous VRAM is on
the left in every plot. Added two new sections: what zipf-1.1 and uniform access actually mean,
and what prefix caching is and affects -- including a decomposition of where the TTFT rise
comes from when most prefill is skipped.

That decomposition is worth recording. Three sources, and which dominates changes across the
sweep:
- **hits migrating GPU -> DRAM**: a DRAM hit is not free, it crosses PCIe first. Predicted
  transfer per request rises 11 -> 23 ms from budget 30 -> 22, and measured p50 rises 54 -> 73.
- **blocks missing both tiers**: ~530 ms each, drives p95 while leaving p50 alone.
- **queueing below the concurrency wall**: at 19 GiB the DRAM hit rate is *higher* (59%) than
  at 30 GiB, so the cache is working *better*, yet TTFT is 2078 ms. That time is neither fetch
  nor recompute -- it is waiting for blocks. The metric stops measuring the cache and starts
  measuring the queue.

**Finetuning harness bug, found and fixed.** The batch probe reported OOM at *every* batch size
including 1. Root cause: `from_pretrained` returns a model in **eval mode**, and HF applies
gradient checkpointing only when `self.gradient_checkpointing and self.training`. The flag read
`True` while the condition was `False`, so checkpointing silently did nothing -- 14.05 GiB of
activations at 2048 tokens instead of ~1 GiB. Adding `model.train()` fixed it. Worth
remembering: the flag being set is not evidence that checkpointing is active.

**Batch size settled at 2 (4096 tokens/step)**, peak 23.04 GiB. B=4 OOMs even in a fresh
process -- activations are ~8 GiB at 4096 tokens, roughly 3.6x my estimate, because the logits
tensor (4096 x 128256 in both bf16 and fp32, plus backward buffers) dominates everything else.

**This changes what the finetuning sweep will show, and the prediction should be corrected
before the data arrives:** 4096 tokens/step is *below* the ~6,900-token break-even, so compute
(1.32 s) is less than the streaming cost (2.22 s). Even with perfect prefetch the transfer
cannot fully hide. The flat region I predicted will not appear. Note also that gradient
accumulation cannot rescue this -- weights stream once per **micro-batch**, so amortisation is
set by the micro-batch, not the effective batch.

**Next:** the 9-point layer sweep at B=2, then RESULTS_FINETUNE.md.

## 2026-08-18 20:30 — Prefetch arm complete; RESULTS_FINETUNE.md written

Re-ran with prefetch over the user's requested 0-16 range. The comparison is stark:

| layers | GiB freed | on-demand | **prefetch** | on-demand cost | **prefetch cost** |
|---|---|---|---|---|---|
| 4 | 1.63 | 2965 tok/s | **3554** | 17% | **0.7%** |
| 8 | 3.25 | 2523 | **3523** | 30% | **1.6%** |
| 12 | 4.88 | 2197 | **3299** | 39% | **7.8%** |
| 16 | 6.50 | 1947 | **3033** | 46% | **15.3%** |

**Freeing 3.25 GiB costs 1.6% with overlap and 30% without** -- a 19x difference in price for
an identical amount of memory saved. Had I written the report an hour earlier it would have
said "offloading costs 30%", which would have been true of my implementation and wrong about
offloading.

Prefetch approaches but does not reach the theoretical `max(compute, transfer)` bound: +0.7% at
4 layers, +1.7% at 8, but +18% at 16. That pattern is diagnostic of **depth-1 lookahead** --
each layer's copy has to fit inside *one* layer's compute (~36 ms), not the whole step, so once
per-layer transfer (~60 ms at 16 layers) exceeds per-layer compute the pipeline stalls no matter
how much total headroom exists. Deeper prefetch is the obvious fix and is logged as a follow-up.

Wrote `RESULTS_FINETUNE.md` and updated `figures/finetune_offload.png` to carry both arms.

**The comparison between the two studies is the most interesting output of the day:**

| | inference (KV offload) | finetuning (weight offload) |
|---|---|---|
| binding wall | concurrency | PCIe bandwidth |
| shape | flat, then cliff | smooth, monotonic |
| free region | 58% of KV for 9% p95 | 25% of VRAM for 1.6% |
| failure mode | engine stalls | none, just slower |
| fix that helps | nothing -- capacity is capacity | overlap, worth 1.6x |

Same machine, same offloading idea, opposite limiting resource -- because serving reads KV *per
reuse* while training reads weights *per step*. And the failure modes differ in kind, not just
degree: inference ends in a cliff, training degrades gracefully.

**Next:** the disaggregated prefill/decode plan (GPU0 prefill, GPU1 decode with offloading,
resize GPU1's VRAM).

## 2026-08-18 22:05 — Five offload strategies compared; two predictions falsified

Ran the full set the user asked for: no prefetch, prefetch depth 1 and 2, and interleaved
placement at both depths, 0-16 layers.

| layers | no prefetch | prefetch d1 | prefetch d2 | interleaved d1 | interleaved d2 |
|---|---|---|---|---|---|
| 4 | 2965 | **3554** | 3551 | 3471 | 3460 |
| 8 | 2523 | 3523 | **3529** | 3434 | 3423 |
| 12 | 2197 | 3299 | 3379 | 3376 | **3378** |
| 16 | 1947 | 3033 | 3032 | **3127** | 3127 |

**Prefetching is the whole story.** No-prefetch loses 46% at 16 layers; every prefetching
variant loses 12-15%. The spread among prefetching strategies is 3% against a 34-point gap to
none at all.

**Two things I predicted turned out wrong, both now corrected in the report:**

1. **Depth 2 does nothing** (and costs 0.41 GiB every point). I had reasoned that depth-1 gives
   each copy one layer's compute window (~35.8 ms) for a ~60.3 ms transfer, so more lookahead
   should help. But the constraint is a **rate mismatch, not a scheduling one**: with all
   offloaded layers at the tail, that region demands 482 ms of transfer against 190 ms of
   compute. Queuing copies earlier on an already-saturated stream changes nothing.
2. **Interleaving is not a clean win** -- it *crosses over*. Below 12 layers tail placement is
   better (3554 vs 3471 at 4), because interleaving puts an offloaded layer at index 0 with no
   preceding compute to hide behind, so every forward starts with a cold synchronous stall. At
   16 layers interleaving wins (+3.1%) because spreading beats the tail region's rate deficit.

Practical rule: offload the tail when offloading a little, interleave when offloading a lot --
and note that plain depth-1 tail prefetch is within 3% of the best at every point, costs no
extra VRAM, and is the simplest to build. The elaborations buy little because the system is
bandwidth-bound rather than scheduling-bound.

Also: my `until ! pgrep -f "..."` waiters self-matched on their own command line for the
**third** time today, stalling one for 1h23m and silently preventing the interleave test from
starting. Switched to polling a completion marker in a log file, which cannot self-match.
