# Related work: limiting or scheduling memory bandwidth inside a GPU

Survey note, Nov 9 2026. Written for the colocation prototype (`REPORT.md` §2.6 and the co-run
measurements that follow it). It is a working note, not a finished related-work section.

**Terminology.** Papers call the same thing "DRAM", "HBM", "VRAM" or "global memory" bandwidth. Here
it is always **GPU memory bandwidth**: the path between the SMs and the GPU's own memory. The
PCIe path to host RAM is a different resource (section 2.8). Names in `code font` are the papers'
or tools' own identifiers.

**How far each source was checked.**
**[T]** read in the paper's text. **[P]** partly read; the rest came from an automatic summary of
the unread part. **[A]** abstract, README or search result only. **[M]** from memory, not checked.
"Not found" below means not found in what was read, about 25 sources in one afternoon.

## 1. The question and the short answer

Our setting: an RTX 5090 (170 SMs, nominal 1.78 TB/s), a decode step that reads 15.8 GB per step and
tops out at 1.34–1.40 TB/s, and a LoRA tenant that moves about 14 MB of GPU memory traffic per
token. The tenant slows decode even on disjoint SMs. The question: who has limited or scheduled GPU
memory bandwidth between co-running jobs, with which knob, and does an SM limit do it?

| Approach | Knob | Examples | Usable on our GPU | Controls bandwidth? |
|---|---|---|---|---|
| Hardware partition | separate memory paths per instance | MIG | no (GeForce is not supported) | yes, static share |
| Memory-path colouring | which channels / banks / L2 sets a job's pages use | Fractional GPUs, SGDRC | not as is (driver changes, per-model reverse engineering) | yes, static share of channels |
| Parallelism throttling | active warps per application | HPCA'18 effective bandwidth | simulator only | indirectly |
| SM partitioning | SM count or SM set | MPS percentage, green contexts, LithOS | yes | **no** (every paper that tested it) |
| Kernel-level gating | when a best-effort kernel may launch | Orion, Mosaic, TGS, Tally, Hummingbird | yes, with call interception | indirectly, by time or predicted slowdown |
| Byte budget | bytes per period | MemGuard, BWLOCK++ (CPU), Ripple (unified-memory SoC) | yes, cooperative | yes, a rate |
| LLM co-serving | tokens or tasklets per iteration | LLMStation, FlexLLM, ConServe, Nexus | yes | indirectly, by latency budget |

Three points stand out.

1. **SM partitioning does not isolate bandwidth; the literature agrees with our measurement.**
   Elvinger et al., Mosaic, Hummingbird and Ripple each state it, with numbers (section 2.4).
2. **Mechanisms that do give a bandwidth share are static partitions of the memory path** (MIG,
   colouring). None is available on a consumer Blackwell card without months of work.
3. **A metered byte rate for a best-effort GPU tenant exists only for CPU victims.** Ripple budgets
   the GPU's bytes to protect CPU tasks on a unified-memory device. For a decode step as the victim
   on a discrete GPU, the systems found gate kernels or tokens by latency; none meters bytes.

## 2. By mechanism

### 2.1 Hardware partitioning: MIG [T]

The MIG User Guide says each instance has "separate and isolated paths through the entire memory
system": crossbar ports, L2 cache banks, memory controllers and address busses are assigned to one
instance. An instance keeps its L2 allocation and bandwidth even when other tenants saturate their
own memory interface. This is the only vendor mechanism that isolates bandwidth.

Limits for us: MIG starts with Ampere data-centre GPUs; the supported list (A100, A30, H100/H200,
B200, RTX PRO Blackwell) has no GeForce card. Instances can be created and destroyed once the GPU
is in MIG mode, but shares are fixed slices, so a decode step that needs about 77 % of the device's
bandwidth cannot lend the rest for a few seconds. MPS runs on top of a MIG instance.

### 2.2 Memory-path colouring: Fractional GPUs and SGDRC

**Fractional GPUs** (Jain, Baek, Wang, Rajkumar; RTAS 2019) [T for abstract, sections III–IV and
the Fig. 8 discussion; evaluation not read]. Reverse-engineers the hash functions from physical
address to L2 cache set and to GPU memory bank, then uses page colouring so two partitions never
share cache sets or banks. Compute is partitioned by pinning persistent thread blocks to SM sets,
because MPS only divides compute. Needs a modified NVIDIA kernel driver and changes to application
source. Tested on GTX 1070/1080 and Tesla V100. Results: run-time variation under multi-tenancy
135 % with MPS, 9 % with FGPU; a co-runner can slow memory transactions more than 10×. Even with
different sets and banks, two partitions in the same "memory module" still interfere through the
L2's miss-status registers (they estimate 16 per module).

**SGDRC** (Zhang et al.; PPoPP 2025; arXiv 2407.13996) [T]. Same idea for newer GPUs. A "VRAM
channel" is the set of L2 cache lines a 1 KiB block of physical GPU memory maps to. The mapping is
"not purely XOR-based", so they train a small network per GPU model on 15 k samples ("we spend one
month collecting") and derive a lookup table. Colouring is done with a shadow page table: about 2 k
lines in the `nvidia-uvm` kernel module plus a kernel transformer on TVM-generated code.
- *Bandwidth share*: best-effort gets a fraction ChBE of the channels (1/3 in the evaluation),
  latency-sensitive gets the rest.
- *Bimodal tensors*: each memory-bound best-effort tensor exists twice, once on all channels and
  once on the ChBE channels; the best-effort job uses the first copy when the latency-sensitive
  queue is empty and the second during colocation.
- *Tidal SM masking*: SM masks set through `libsmctrl`; latency-sensitive kernels evict
  best-effort kernels from their SMs.
- Channel isolation alone cuts p99 latency of the latency-sensitive job by 28.7 % (Tesla P40) and
  47.5 % (RTX A2000) with SMs split evenly in both cases.
- Limits stated: only P40 and A2000 evaluated; no generative LLMs; A100/H100 "need adaptation";
  closed-source kernels (cuBLAS, cuDNN) are not supported; memory footprint nearly doubles without
  tensor reuse.

Elvinger et al. call this family "code-intrusive". Our reading: a channel share also caps the
victim. A decode step that alone uses 75–79 % of the device's bandwidth would lose throughput with
2/3 of the channels, so colouring suits small latency-sensitive models, not a bandwidth-bound LLM.

### 2.3 Parallelism throttling in simulation: effective bandwidth (HPCA 2018) [T for sections I–III]

Wang, Luo, Ibrahim, Kayiran, Jog. GPGPU-Sim, two applications on exclusive sets of cores. The knob
is each application's thread-level parallelism: the number of warps allowed to run. Metric:
effective bandwidth = attained bandwidth ÷ combined L1×L2 miss rate. Their pattern-based search over
TLP pairs improves weighted speed-up by 20 % and fairness 2× over each application's own best TLP.
It builds on single-application throttling (DynCTA, CCWS). Two things matter for us: the idea that
a job's memory traffic is steered by its active warps, not by its core count; and that no real GPU
exposes this knob.

### 2.4 SM partitioning: what the papers say about bandwidth

- **Elvinger, Strati, Enright Jerger, Klimovic, "Understanding GPU Resource Interference One Level
  Deeper"** (SoCC 2025; arXiv 2501.16909) [T]. H100. Llama-3.1-8B decode (batch 8) on 64 SMs next to
  a copy kernel on the other 68 SMs, each in its own green context: P90 time between tokens rises
  from 16.9 ms to 22 ms as the copy kernel's bandwidth use rises to 81 %, "up to 1.3× slowdown".
  "Even when workloads run on disjoint SMs, using strict SM isolation mechanisms such as green
  contexts, they can still contend for L2 capacity and L2 and global memory bandwidth." Also:
  the MPS limit "does not enforce mutual exclusion of SMs between MPS clients"; decode on fewer
  than half the SMs costs only 1.19×; Nsight Compute cannot collect its metrics under colocation
  because it replays kernels. They propose no mechanism; they ask for a kernel-level interference
  estimator and for "programmer-friendly ways to partition SMs and DRAM channels".
- **Mosaic** (Strati et al.; arXiv 2610.07504) [T to §4.4, rest P]. "SM partitioning reduces
  interference but limits the throughput attainable by the admitted best-effort kernels." With
  green contexts the high-priority job still slows 1.1–7× [P].
- **Hummingbird** (Hu et al.; arXiv 2601.04071) [T to §4.2, rest P]. LithOS and green contexts
  give "TPC-level compute control, but the other resources, e.g., HBM bandwidth and L2 cache
  interference, still remain uncontrolled."
- **Nexus** (Shi, Cai, Du, Jia; arXiv 2507.06608) [T to §4.1, rest P]. Decode "is memory-bound and
  fails to utilize added SMs effectively": 30 % → 40 % of SMs gives 10 %, beyond 50 % under 3 %
  per step. A longer prefill KV (2 k → 10 k tokens) slows the same decode batch by 36 %, which they
  attribute to bandwidth contention.
- **Ripple** (Zhang, Ma; arXiv 2611.01067) [T; appendix numbers P]. DGX Spark, 48 SMs, memory
  shared with the CPU. Cutting the LLM from 48 to 8 SMs "has very limited effect on mitigating CPU
  slowdown". Appendix: decode on 16 of 48 SMs lowers its byte rate by only 8 % [P]. This is the
  one published SM-count-to-byte-rate figure found, and it is for a bandwidth-bound job.
- **LithOS** (Coppock et al.; SOSP 2025; arXiv 2504.15465) [P]. TPC-level scheduling, kernel
  atomisation, right-sizing. No treatment of bandwidth or L2 contention found.
- **LLMStation** [T]. The MPS percentage is "fixed at framework startup"; with an unrestricted
  fine-tuning share, decode misses its target. On an RTX 3090, fine-tuning next to serving raises
  p99 time-to-first-token up to 13.2× and p99 time per output token up to 3.6×.

Our own data fit: slowdown per 1 k tenant tokens/s is the same on shared and on disjoint SMs.
Not found anywhere: the curve "a compute-heavy tenant's bandwidth use versus its SM count".

### 2.5 Kernel-level gating: when a best-effort kernel may launch

- **Orion** (Strati, Ma, Klimovic; EuroSys 2024) [T]. Offline profile per kernel from Nsight
  Compute: compute throughput, memory bandwidth utilisation, duration, SMs needed. A kernel is
  memory-bound if its bandwidth utilisation is over 60 % (or by the roofline analysis). Policy
  (Listing 1): launch a best-effort kernel if no high-priority kernel is running, or if it needs
  fewer SMs than `SM_THRESHOLD` **and** its profile differs from the running high-priority
  kernel's. A second check stops launches when outstanding best-effort kernels exceed
  `DUR_THRESHOLD` (2.5 % of the high-priority request latency). Motivating example on a V100: two
  batch-norm kernels, each using 40 % of the SMs and 80 % of the bandwidth, interfere; a
  convolution (89 % compute, 20 % bandwidth) next to a batch-norm does not. LLMs are "not our
  target workloads"; PCIe scheduling is left as future work.
- **Mosaic** [T/P]. Predicts each kernel pair's slowdown from isolated Nsight Compute profiles
  (`DRAM_read_bw`, `DRAM_write_bw`, L2 read/write bandwidth, SMs, stall ratios) with XGBoost, and
  admits a best-effort kernel only if the predicted end-to-end latency stays within the target.
  Two findings to remember: "aggregate bandwidth alone does not capture the interference
  behavior" (reads hurt their victim more than writes), and two victims with similar bandwidth and
  L2 profiles slow 1.3× and 2× under the same aggressor.
- **Hummingbird** [T/P]. Criticises Orion's 60 % average threshold: under Orion, Llama-8B's
  per-token time varies 4.2× more than under REEF. Its own answer is preemption: split
  best-effort kernels into pieces of at most 400 µs [P], launch the next piece by a "kernel tick",
  run best-effort work only in detected gaps. It does not cap bandwidth.
- **TGS** (Wu et al.; NSDI 2023) [A]. Adaptive rate control: watches the production job's
  performance and adjusts the opportunistic job's share. A feedback rate limiter, like our
  TPOT-feedback cap, with no bandwidth model.
- **Tally** (Zhao, Jayarajan, Pekhimenko; ASPLOS 2025; arXiv 2410.07381) [A]. Thread-block-level
  slicing and preemption of best-effort kernels; 7.2 % p99 overhead for the high-priority job
  against 188.9 % for TGS.
- **REEF** (OSDI 2022) [M]: reset-based preemption, padding of real-time kernels with
  best-effort ones.

All of these decide *when* a best-effort kernel runs: yes or no per kernel, by time or by a
predicted slowdown. None counts the bytes the best-effort job moves.

### 2.6 Byte budgets: CPU systems and one unified-memory device

- **MemGuard** (Yun et al.; RTAS 2013) [A]. Each core gets a budget of memory transactions per
  regulation period; a performance counter raises an interrupt when the budget is used and the
  core is stalled until the next period. Guaranteed plus best-effort parts, with reclaim of unused
  budget. Short periods and small budgets cost overhead.
- **BWLOCK++** (Ali, Yun; ECRTS 2018; arXiv 1712.08738) [A]. On integrated CPU–GPU boards: while
  a GPU kernel runs, best-effort CPU cores are throttled.
- **Intel RDT memory bandwidth allocation** [M]: hardware per-core throttling on server CPUs.
- **Ripple** [T]. The nearest work to a byte budget for GPU traffic. The device "provides no
  direct bandwidth allocation mechanism" and its bandwidth counter "returns no reading", so the
  scheduler is counter-free: the forward pass is cut into layer units; the scheduler admits
  contiguous "grants" of units; a CPU job's allowance is A = min(max(0, D − ā)/κ, ρD), with D the
  deadline, κ the CPU delay per GPU byte and ρ the peak GPU byte rate. Engines are paused between
  grants through SGLang hooks. It also runs decode on 8 or 16 SMs while CPU jobs overlap, because
  a narrow decode keeps most of its throughput and costs less CPU delay per GB.
  Differences from our case: the victim is a CPU task, the device has 273 GB/s shared memory, and
  the LLM is the party being throttled.

### 2.7 LLM co-serving systems

- **LLMStation** (He, Yang, Lu, Klimovic, Alonso; USENIX ATC 2025) [T]. The closest system to our
  setting: LoRA-style fine-tuning next to decode, sharing the base model. An iteration-level
  scheduler picks how many fine-tuning *tasklets* to run with each decode iteration, from profiled
  or predicted latencies, so the iteration stays inside the decoding target; fine-tuning is
  suspended during prefill; a suspendable autograd engine cuts the backward pass at operator
  boundaries. The stated reason it works is "co-execution of memory-bound and compute-bound
  operations". The budget is latency, not bytes. RTX 3090 and H100.
- **Nexus** [T/P]. Prefill and decode in one vLLM engine on green-context partitions that are
  resized at run time. Its cost model takes max(compute time, memory time) per operator and gives
  decode a share of bandwidth B in proportion to its bytes against prefill's. Bandwidth enters the
  prediction only; nothing throttles it.
- **FlexLLM** (arXiv 2402.18789) [A]: token-level co-serving of inference and fine-tuning.
  **ConServe** (arXiv 2410.01228) [A]: offline requests next to online ones, token-level batch
  sizing plus preemption points between layers.

### 2.8 The host link: GUST (CIDR 2026) [T]

Jiang, Nicholson, Sanca, Ailamaki, "Data Movement-Aware GPU Sharing for Data-Intensive Systems".
About PCIe, not GPU memory bandwidth: analytics queries saturate the link, inference stresses the
device, and the scheduler interleaves transfer-intensive and device-intensive kernels. Slowdown
against dedicated GPUs drops from 3.9–7× to 2.8× for four mixed workloads (V100S, A100). MIG "provides no
PCIe isolation". Relevant to our host↔GPU section (`REPORT.md` §3), not to the decode slowdown.

### 2.9 Placement inside the chip [T/P]

"Dissecting How Die Scaling Breaks GPU Fine-grained Scheduling" (Fan et al.; arXiv 2609.24270).
On H200/B200 the L2 is split in two partitions; a remote access raises memory latency by up to
67 %. Existing systems "all control how many SMs are used, but none considers which SMs are
assigned". Their fix is page and SM placement, not rate control. Worth one check on the 5090:
whether which SMs a green context gets changes the bandwidth it can reach.

## 3. What this means for the prototype

1. **Our SM result is expected, and the citations are ready**: Elvinger et al. for decode on
   disjoint SMs, Nexus for decode's flat SM curve, Ripple for SM count against byte rate.
2. **What we can add in measurement.** Nobody reports how a tenant's bandwidth use changes with
   its SM count, or checks byte-for-byte accounting under colocation. Elvinger et al. note that
   Nsight Compute cannot measure during colocation. The device-wide GPM counter can: we calibrated
   it against the decode step's known read rate (1 % ≈ 17.8 GB/s at every point from 16 to 170
   SMs). It gives the sum of all jobs, so each job's part has to come from its own byte count.
3. **Mechanisms open to us on this GPU are cooperative, in the tenant's own loop.**
   - *Byte budget* (MemGuard and Ripple style): the VRAM scheduler hands the tenant GB/s; the
     tenant converts it with its bytes per token (≈ 14 MB) and pauses between micro-batches.
   - *Iteration-level tasklets* (LLMStation style): the same idea counted in tasklets and latency.
   - *Phase-aware placement*: tenant work in prefill and idle windows (LLMStation suspends
     fine-tuning during prefill for the opposite reason, compute).
   - *Kernel-level admission* (Orion, Mosaic): needs call interception and offline profiles; a
     larger build.
   Colouring and MIG are out for this card.
4. **Open point for a contribution.** A byte-metered budget for a best-effort tenant, with a
   bandwidth-bound decode step as the victim, on a discrete GPU without MIG: not found. Two
   warnings from Mosaic apply to any byte model: reads and writes do not cost the victim the same,
   and victims with the same profile slow down differently.

## 4. Sources

- MIG User Guide [T]: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/
- Fractional GPUs, RTAS 2019 [T, partial]: https://github.com/sakjain92/Fractional-GPUs
- SGDRC, PPoPP 2025 [T]: https://arxiv.org/abs/2407.13996
- Effective bandwidth, HPCA 2018 [T, partial]: https://par.nsf.gov/servlets/purl/10082111
- Elvinger et al., SoCC 2025 [T]: https://arxiv.org/abs/2501.16909
- Orion, EuroSys 2024 [T]: https://anakli.inf.ethz.ch/papers/orion_eurosys24.pdf
- Mosaic [T/P]: https://arxiv.org/abs/2610.07504
- Hummingbird [T/P]: https://arxiv.org/abs/2601.04071
- LithOS, SOSP 2025 [P]: https://arxiv.org/abs/2504.15465
- Nexus [T/P]: https://arxiv.org/abs/2507.06608
- LLMStation, USENIX ATC 2025 [T]: https://www.usenix.org/system/files/atc25-he-yongjun.pdf
- Ripple [T, appendix P]: https://arxiv.org/abs/2611.01067
- Die scaling [T/P]: https://arxiv.org/abs/2609.24270
- GUST, CIDR 2026 [T]: https://www.vldb.org/cidrdb/papers/2026/p13-jiang.pdf
- TGS, NSDI 2023 [A]: https://www.usenix.org/conference/nsdi23/presentation/wu
- Tally, ASPLOS 2025 [A]: https://arxiv.org/abs/2410.07381
- ConServe [A]: https://arxiv.org/abs/2410.01228
- FlexLLM [A]: https://arxiv.org/abs/2402.18789
- MemGuard reproduction report (describes the original mechanism) [A]: https://arxiv.org/abs/2609.04547
- BWLOCK++ [A]: https://arxiv.org/abs/1712.08738
- Not checked [M]: REEF (OSDI 2022), Intel RDT memory bandwidth allocation, iGniter, gpulet.
