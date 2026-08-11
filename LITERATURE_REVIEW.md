# Literature Review: Fine-Grained GPU Sharing and Co-Location Scheduling

## Scope and Reading Guide

This review covers the seven PDF files stored directly in the main `orion-citation` directory. The collection spans several layers of the GPU-sharing stack. Bless, Hummingbird, Tally, MMK, and LithOS directly manipulate GPU execution at kernel, thread-block, stream, context, or SM/TPC granularity. SMore performs cluster-level admission and placement for serverless inference functions. The file named `Harli SLO-Aware Co-location of LLM Inference and PEFT Finetuning.pdf` contains the OSDI 2024 paper *Usher: Holistic Interference Avoidance for Resource Optimized ML Inference*, not Harli; this review follows the paper content and records the filename mismatch.

The review first presents direct kernel schedulers from focused launch-time mechanisms to broader hierarchical and GPU-OS designs, then covers complementary workload-level systems. The papers are reviewed with an emphasis on five questions:

1. What resource-underutilization or interference problem does the paper target?
2. At what layer and granularity does it make scheduling decisions?
3. How does it observe, intercept, transform, or control GPU work?
4. What guarantees and performance improvements does it provide?
5. How closely does it match the Orion-style research direction of transparent kernel interception and scheduling?

## 1. Bless: Improving GPU Sharing Performance through Adaptive Bubbleless Spatial-Temporal Sharing

> **In brief:** Bless intercepts CUDA kernel launches and schedules short cross-application kernel squads across preconfigured MPS contexts, reclaiming transient SM bubbles while preserving each tenant's GPU quota. Its key contribution is adaptive spatial-temporal sharing that improves request latency without sacrificing quota-based fairness.

### Problem and motivation

Bless addresses a limitation shared by conventional temporal and spatial GPU multiplexing. Temporal sharing controls how frequently each application's kernels may be launched, but non-preemptive and heterogeneous kernels make time-based quotas imprecise. Spatial sharing, such as NVIDIA MPS, allocates a fraction of the GPU's SMs to each application, but statically reserved SMs become bubbles whenever an application temporarily lacks enough parallel work. MIG provides stronger isolation, but at coarse, inflexible partition sizes. In all three cases, nominal allocation is not equivalent to useful execution: a tenant can own a quota while leaving part of it idle, yet other tenants cannot safely and fairly reclaim the unused capacity.

Bless defines a bubble as GPU capacity that is temporarily unused under an application's assigned quota. Its objective is to squeeze these bubbles without violating any application's promised quota. This is more subtle than maximizing aggregate throughput. The system must improve latency fairly across tenants, avoid making a quota-compliant tenant slower than isolated execution at that quota, and react within a request rather than waiting for request boundaries.

![Bless motivation: temporal and static spatial sharing both leave bubbles that ideal sharing could reclaim.](assets/paper_figures/bless_motivation.png)

*Motivation (Figure 1): temporal sharing creates gaps between non-preemptive kernels, while static spatial sharing leaves reserved SMs idle. Bless targets the bubble-free schedule in the bottom timeline while retaining quota guarantees.*

### System design

Bless combines temporal and spatial sharing at the granularity of a **kernel squad**. A kernel squad is a short group of kernels drawn from different applications and scheduled together. Because a squad lasts much less time than an end-to-end inference request, Bless can reconfigure GPU allocation multiple times during a request and reclaim transient bubbles.

![Bless mechanism: runtime components for forming kernel squads and selecting execution configurations.](assets/paper_figures/bless_mechanism.png)

*Main mechanism (Figure 5): wrapped application requests enter the multi-task scheduler; the execution-configuration determiner selects a profiled allocation, and the concurrent-kernel manager dispatches the resulting kernel squad to resource-configured GPU contexts.*

The system contains four main components:

- **Offline profiler.** Bless profiles each long-running application under multiple SM configurations using MPS. It records kernel sequences, execution progress, and latency/resource behavior under different allocations. The resulting profiles let the runtime estimate request progress and the performance of possible concurrent kernel configurations without exhaustively measuring every online combination.
- **Multi-task scheduler.** The scheduler tracks the progress of every active request and decides which kernels should form the next squad. It attempts to preserve quota-derived progress while giving temporarily unused capacity to applications that can use it. This makes the policy work-conserving while maintaining fairness.
- **Execution-configuration determiner.** For each candidate squad, Bless estimates how kernels will perform under alternative concurrent configurations. It chooses a configuration that reduces bubbles and latency without violating quota guarantees.
- **Concurrent kernel manager.** Bless enforces the chosen configuration using multiple pre-established GPU contexts with different resource allocations. Applications submit through wrapped CUDA runtime APIs, including `cudaLaunchKernel`, and Bless redirects kernel execution to an appropriate context. Contexts encode heterogeneous SM configurations, allowing the runtime to change effective allocation without paying the full cost of creating or reconfiguring a context for every squad.

The important distinction is that Bless does not merely set a static MPS percentage. It performs host-side scheduling of kernel squads and selects among resource-configured contexts during execution. This allows it to approximate a continuously adjustable allocation using a finite set of contexts.

### Scheduling policy and guarantees

Bless uses request progress as a fairness signal. A tenant with quota \(q\) should make at least as much progress as it would make in isolation with that quota. When one application falls behind its expected progress, the scheduler prioritizes its kernels. When an application has no useful work or cannot consume its full allocation, another application can borrow the bubble.

The performance estimator predicts a squad's duration under possible concurrent kernel allocations. Bless thereby avoids relying on the hardware scheduler to colocate arbitrary kernels blindly; uncontrolled concurrency could create cache, bandwidth, or compute contention that increases latency even when idle SM capacity exists.

### Implementation and evaluation

Bless is implemented in more than 5,000 lines of C++ and supports applications built with TVM and PyTorch. It is evaluated primarily on NVIDIA A100 GPUs using synthetic and real-world query-arrival traces, multiple applications, quota ratios, and model combinations.

The main reported result is a **21.1%-37.3% average latency reduction** over state-of-the-art sharing approaches while preserving promised quotas. In homogeneous cases, colocating two BERT inference instances reduces average latency by 39.1%; with four BERT instances, the reduction reaches 41.2%. The paper also evaluates beyond pairwise sharing, quota combinations, kernel-squad granularity, prediction accuracy, and scheduling overhead. Kernel squads in the evaluation generally last approximately 0.7-10 ms, enabling allocation changes substantially faster than request-level schedulers.

### Comparison with Orion

Bless and Orion share the same basic control opportunity: both transparently observe CUDA launches, profile kernel behavior, and decide which kernels from independent applications should execute concurrently. Orion is primarily **interference-aware**: it classifies kernels by resource behavior and attempts to overlap complementary kernels, especially compute-intensive and memory-intensive work, while respecting application priorities. Bless is primarily **quota- and bubble-aware**: it asks whether each application is making the progress promised by its SM quota and lets another application reclaim capacity that the quota owner cannot currently use.

Their scheduling units and enforcement mechanisms also differ. Orion schedules individual, non-preemptive kernels through intercepted queues and relies on the GPU's concurrent-kernel execution; once admitted, a kernel normally runs to completion. Bless groups several kernels into short **kernel squads** and dispatches them through pre-created MPS contexts representing different SM allocations. This gives Bless more explicit spatial control and a clearer quota guarantee, but requires offline profiles and a discrete context/configuration space. Orion is conceptually simpler and more directly targets beneficial kernel pairing; Bless is better suited when proportional allocation and fair bubble reclamation are first-class requirements.

### Assessment

Bless belongs strongly in the kernel-scheduling category. It wraps CUDA launch APIs, observes individual kernels, creates cross-application kernel squads, and enforces spatial configurations through multiple contexts. Relative to Orion, Bless focuses more directly on accurate quota enforcement and bubble reclamation. Its key limitation is dependence on offline profiling and a discrete set of pre-created contexts. Its ability to generalize to dynamic-shape workloads, unseen kernels, rapidly changing LLM batches, and GPUs with different resource-partitioning behavior depends on the quality and coverage of those profiles.

## 2. Hummingbird and Tally: Transparent Fine-Grained Temporal Sharing

> **In brief:** Hummingbird and Tally implement the same high-level policy: turn best-effort kernels into bounded execution units, run those units only while the high-priority workload is inactive, and stop admitting new best-effort work as soon as high-priority work returns. Hummingbird specializes this policy around explicit bubble detection and split-kernel launch control, whereas Tally contributes a more general per-kernel choice between slicing and persistent-worker preemption.

### Common motivation and system abstraction

Both systems address the utilization-isolation conflict created by bursty GPU workloads. Reserving a GPU for a latency-critical application provides strong isolation but wastes the intervals in which that application has no GPU work. Conventional temporal sharing can fill those intervals with a best-effort job, but CUDA offers no public mechanism that can cheaply terminate an arbitrary running kernel. If a long best-effort kernel has already been admitted when a high-priority kernel arrives, the high-priority kernel may wait for the best-effort kernel's remaining execution time. Stream priority does not solve this problem because it influences pending work rather than evicting work that is already resident.

Hummingbird and Tally reach the same central conclusion: the application-visible kernel is too coarse a scheduling unit, but its thread blocks provide natural, much shorter boundaries. Both transparently intercept CUDA calls from unmodified applications, transform best-effort kernels when necessary, and expose execution units whose residual running time can be bounded. Their common control loop is:

1. Observe high- and low-priority CUDA work below the framework.
2. Convert a best-effort kernel into smaller schedulable units.
3. Admit those units only while the high-priority workload is inactive.
4. When high-priority work arrives, stop the admission of new units and wait only for the current unit to drain.
5. Resume the remaining best-effort work after the next high-priority idle interval begins.

Thus, both systems implement **cooperative software preemption** rather than immediate hardware preemption. Neither can interrupt an instruction, warp, or arbitrary point inside a running thread block. The isolation bound is determined by the remaining duration of the currently admitted best-effort unit plus runtime and launch overhead:

\[
T_{\mathrm{HP\ wait}} \lesssim T_{\mathrm{residual\ BE\ unit}} + T_{\mathrm{runtime}}.
\]

The following figure uses this shared vocabulary and then separates the mechanisms that realize the bounded best-effort unit.

![Unified architecture and execution model for Hummingbird and Tally. Both intercept unmodified applications, convert best-effort kernels into bounded execution units, harvest high-priority idle intervals, and drain before high-priority execution resumes. Hummingbird controls a sequence of split-kernel launches; Tally chooses slicing or persistent-worker preemption.](assets/paper_figures/hummingbird_tally_unified.svg)

*Unified high-level view. Blue units are best-effort work admitted into gaps between red high-priority intervals. Hummingbird bounds interference by keeping at most one split-kernel in the device queue; Tally either does the same with slices or lets persistent workers stop between logical thread blocks.*

### Hummingbird: bubble-aware split-kernel scheduling

Hummingbird is an application-transparent runtime that intercepts low-level CUDA Driver APIs. Its distinctive contribution is not a different overall sharing objective, but a combination of **explicit bubble discovery**, **PTX-based kernel splitting**, and **low-overhead launch pacing** designed to exploit even microsecond- and millisecond-scale gaps in modern inference and distributed workloads.

For a best-effort kernel, Hummingbird first profiles how execution time changes with grid size. It chooses an efficient split size by considering GPU parallelism and memory-bandwidth saturation: making the sub-grid smaller initially reduces its execution time, but further reduction eventually stops helping and only sacrifices utilization. The PTX transformer then injects offset parameters into uses of `blockIdx`, allowing the original logical grid to be expressed as a sequence of smaller sub-grid launches without changing the application's view of block indices. The remaining work is represented by the not-yet-launched split-kernels.

Hummingbird divides high-priority idle intervals into two operational categories. **Small bubbles**, typically caused by CPU-GPU synchronization or inter-GPU communication, are identified from repeatable host-side CUDA and NCCL API patterns. For example, iteration boundaries may expose a `cudaMemcpyAsync`/`cudaStreamSynchronize` pattern, while pipeline-parallel gaps may be associated with NCCL send and receive calls. Lightweight CUDA events connect these host-side hints to actual GPU progress. **Large bubbles**, caused by effects such as request fluctuation or network delay, are detected when the high-priority queue remains empty beyond a workload-specific threshold; a request-interval predictor can further estimate how much best-effort work will fit.

During a detected bubble, Hummingbird issues best-effort split-kernels through a **kernel-tick scheduler**. It limits the GPU device queue to at most one such split-kernel, so an arriving high-priority kernel is delayed by no more than the current split rather than an arbitrarily deep queue. Instead of synchronizing after every split, the runtime uses the profiled split duration as a tick and launches the next split as the previous one approaches completion. On high-priority arrival, the scheduler changes its control flag and the asynchronous launch thread stops issuing further splits. Importantly, this flag controls the **host-side launch loop**; Hummingbird does not convert the kernel into Tally's persistent-worker loop and does not make the already-running split exit early.

Very small splits improve responsiveness but impose launch and synchronization costs and may underfill the GPU. Hummingbird therefore consolidates adjacent splits, potentially restoring the original grid size, when it detects or predicts a sufficiently large bubble. This makes its granularity explicitly phase-dependent: conservative units protect short, structured bubbles, while larger units recover best-effort throughput in long idle periods.

Hummingbird also treats memory capacity as part of colocation. Its extended memory manager prioritizes local HBM for the high-priority task and can offload best-effort pages to NVLink-connected remote HBM or, as a fallback, host DRAM. This capability is orthogonal to the common compute-yield loop, but it matters when two models cannot otherwise coexist in local GPU memory.

The evaluation spans inference and training, including LLM and distributed configurations. The paper reports **9.7x** higher high-priority SLO attainment than representative spatial-sharing approaches and **3.5x** higher attainment than temporal-sharing approaches, with less than **1%** loss relative to exclusive high-priority execution. It also reports up to **2.4x** higher best-effort throughput than REEF. These results should be understood as evidence for the complete Hummingbird design - splitting, bubble hints, kernel ticks, consolidation, and memory management - rather than for a new hardware preemption primitive.

### Tally: profile-guided slicing or persistent-worker preemption

Tally begins from the same temporal-sharing loop but focuses more heavily on making the yield mechanism general across kernels. It inserts a client/server CUDA virtualization layer beneath unmodified applications. Client-side interception forwards CUDA operations to a centralized server that owns the GPU context, maintains application-visible ordering, profiles kernels, and makes priority-aware dispatch decisions. Best-effort kernels execute only while the high-priority application is inactive; high-priority work is dispatched immediately when it appears.

Tally exposes two alternative ways to create bounded best-effort units. The first, **kernel slicing**, is conceptually the same grid decomposition used by Hummingbird: Tally launches a sequence of small sub-grids and rewrites block indices with an offset. The end of each slice is a yield point. Finer slices reduce high-priority waiting time but increase the number of launches.

The second, **kernel preemption**, converts a kernel into a persistent-worker form. Tally launches a fixed set of physical worker blocks once. Each worker repeatedly obtains a logical block ID from a global counter, executes the original block body for that ID, and then checks a preemption flag before claiming more work. When a high-priority kernel arrives, the scheduler sets the flag. Workers finish their current logical blocks, stop claiming new IDs, and collectively exit; after the high-priority interval, Tally resumes from the counter that records the unfinished logical grid. This mechanism avoids issuing a separate CUDA launch for every small slice, but adds counter operations, flag checks, control flow, and synchronization inside the transformed kernel.

Persistent transformation requires more than wrapping the original block body in a loop. Original kernels may contain divergent returns and block-wide barriers. If some threads return because of preemption while others reach an original barrier, the block can deadlock. Tally's unified-synchronization transformation rewrites return and synchronization paths so that threads agree on safe block-level exit. This semantics-preserving transformation is one of Tally's most important technical contributions.

Neither primitive dominates for every kernel. Slicing is lighter-weight but pays repeated launch overhead; persistent preemption uses a single launch but may substantially perturb kernels that are sensitive to additional synchronization and control flow. Tally therefore profiles each best-effort kernel under candidate methods and parameters. It chooses the slicing factor or persistent-worker configuration that maximizes best-effort performance while keeping turnaround below its configured bound. Unlike Hummingbird, Tally does not depend on recognizing semantic LLM, synchronization, or NCCL bubble types: an empty/inactive high-priority side is the generic opportunity signal. Its adaptivity is primarily **per kernel and per preemption primitive**, rather than per predicted bubble length.

Across high-priority inference and best-effort training combinations, Tally reports an average high-priority P99 latency overhead of **7.2%**, substantially below time slicing, MPS, MPS priority, and TGS in its experiments. It retains more than **80%** of TGS's aggregate throughput. The throughput gap under heavy high-priority load reflects a deliberate design choice: Tally avoids priority-class co-execution and sacrifices some best-effort throughput for robust tail-latency isolation.

### Direct comparison between the two systems

At the policy level, Hummingbird and Tally should be placed in the same family rather than described as fundamentally different systems. Both harvest high-priority gaps with best-effort work, both rely on PTX transformation and sub-kernel/block boundaries, and both react to high-priority arrival by preventing additional best-effort work from starting. Their differences lie in how they define an opportunity and how they realize the yield point:

| Dimension | Hummingbird | Tally |
|---|---|---|
| Application contract | Unmodified applications; low-level CUDA interception | Unmodified applications; client/server CUDA virtualization |
| Opportunity to run BE work | Explicitly detects small CUDA/NCCL bubbles and large idle intervals | Runs BE work whenever the high-priority application is inactive |
| Execution primitive | Sequence of PTX-rewritten split-kernel launches | Per-kernel choice of slicing or persistent-worker preemption |
| Response to HP arrival | Host launch thread stops issuing splits; current split drains | Stop issuing slices, or set a device-visible flag so workers drain at logical-block boundaries |
| Granularity adaptation | Split size is profiled; adjacent splits are consolidated for predicted/detected large bubbles | Method and parameters are profiled per kernel to meet a turnaround bound |
| Principal specialization | Bubble harvesting for inference/distributed workloads, plus memory-capacity management | General, semantics-safe block-level yield across heterogeneous DL kernels |
| Main cost | Split launches, tick synchronization/pacing, workload-specific hints, PTX availability | Virtualization, profiling, transformation, counters/checks, and synchronization rewriting |

The distinction is therefore narrower than “bubble scheduling versus preemption.” Both systems schedule bubbles and both provide software preemption. More precisely, **Hummingbird realizes preemption by controlling the sequence of ordinary split-kernel launches**, while **Tally additionally supports an intra-kernel persistent-worker protocol**. Hummingbird invests more in identifying and sizing the high-priority gap; Tally invests more in choosing and safely implementing the best yield mechanism for each kernel.

### Comparison with Orion

All three systems create a transparent control point below unmodified ML frameworks, but Orion uses that control point differently. Orion primarily performs **interference-aware spatial co-execution**: it schedules whole kernels from different workloads onto CUDA streams and overlaps resource-compatible kernels when doing so is expected to improve utilization without excessive interference. Hummingbird and Tally primarily perform **fine-grained temporal borrowing**: best-effort work occupies an interval in which high-priority work is absent and is withdrawn before the next high-priority phase.

This difference produces distinct isolation properties. Once Orion has admitted a best-effort kernel, it normally runs to completion and may execute concurrently with a high-priority kernel. Priority streams can order pending launches but cannot evict the resident best-effort blocks. Orion's high-priority delay and interference therefore depend on the remaining duration and resource behavior of previously admitted whole kernels. Hummingbird and Tally pay transformation and scheduling overhead to bound that residual work at a split or logical-block boundary, and they normally avoid cross-priority co-execution. They consequently provide a clearer tail-latency isolation story, especially when best-effort kernels are long or memory-bandwidth interference is difficult to predict.

Orion retains two potential advantages. First, spatial overlap can use compute and memory resources simultaneously even while the high-priority kernel is active; strict temporal sharing leaves complementary capacity unused during that interval. Second, Orion can schedule an opaque library kernel as a whole even when transformable PTX is unavailable. Hummingbird's and Tally's finer control depends on successful transformation or must fall back to a coarser path. The systems therefore occupy different points in the same design space: Orion favors low-overhead throughput through compatible overlap, whereas Hummingbird and Tally spend additional machinery to obtain bounded withdrawal and stronger performance isolation.

### Summary and research implications

Hummingbird and Tally demonstrate a shared recipe for transparent fine-grained GPU sharing: intercept CUDA, replace a monolithic best-effort launch with resumable bounded units, admit those units during high-priority inactivity, and make the remaining unit small enough that high-priority work starts quickly. Hummingbird's main extension is workload-structure-aware bubble detection, launch pacing, dynamic consolidation, and memory offloading. Tally's main extension is a profile-guided choice between conventional slicing and a semantics-safe persistent-worker mechanism.

For a kernel-scheduling research agenda, the papers expose a useful composition opportunity. Hummingbird's estimate of **when a bubble begins and how long it may last** could drive Tally's choice of **how the current kernel should yield**. Orion contributes a complementary third option: when temporal withdrawal would waste substantial resources, an interference model could decide whether controlled spatial overlap is safe. A unified scheduler could therefore choose among full overlap, split-kernel temporal borrowing, and persistent-worker execution according to the current phase, predicted gap length, kernel transformability, and SLO budget.

## 3. MMK: A Hybrid Scheduling Framework for Fine-Grained GPU Sharing for Deep Learning Applications

> **In brief:** MMK composes MIG, MPS, and intercepted kernel scheduling into a three-level hierarchy: MIG supplies coarse isolation, MPS controls intra-partition SM shares, and the kernel scheduler handles short-term contention. The framework shows that hardware partitioning and fine-grained software scheduling are complementary rather than competing approaches.

### Problem and motivation

MMK starts from the observation that no single NVIDIA sharing mechanism simultaneously provides strong isolation, fine-grained flexibility, and high utilization. MIG provides hardware-enforced partitions of compute, memory bandwidth, cache, and memory capacity, but supports only a small set of static configurations and is expensive to reconfigure. MPS permits concurrent execution and configurable SM percentages, but offers weaker isolation and leaves important resources shared. Kernel-level scheduling can respond at very fine granularity, but adds runtime overhead and must manage interference explicitly.

MMK therefore proposes a hybrid hierarchy that combines **MIG**, **MPS**, and **kernel scheduling**. Rather than treating them as competing alternatives, it assigns each mechanism a role at a different layer.

![MMK motivation: SM utilization and execution time vary substantially across MIG and MPS allocations.](assets/paper_figures/mmk_motivation.png)

*Motivation (Figures 3-4): workload mixtures leave different amounts of SM capacity unused, and sensitivity to MPS limits changes across MIG partitions. A fixed choice of either mechanism cannot consistently provide both isolation and efficiency.*

### Three-level scheduling architecture

At the outer level, MMK partitions a GPU with MIG. Workloads that strongly interfere or require hard memory/fault isolation can be separated into different MIG instances. This limits cross-group contention and creates coarse resource envelopes.

Within each MIG instance, MMK uses MPS to control the share of SM resources available to colocated processes. MPS provides a more flexible spatial division than MIG and allows concurrent kernels from multiple clients.

At the finest level, MMK intercepts kernel launches and schedules kernels according to priority and interference characteristics. This corrects problems that static MIG/MPS allocations cannot handle, such as short-term phase changes, long blocking kernels, and latency-critical work arriving behind best-effort execution.

![MMK mechanism: hierarchical predictor, MIG partition scheduler, and per-partition kernel scheduler.](assets/paper_figures/mmk_mechanism.png)

*Main mechanism (Figure 7): offline job- and kernel-level profiles train a performance predictor. At runtime, the hybrid partition scheduler assigns jobs and MPS quotas across MIG instances, then a kernel scheduler controls launches inside each partition.*

The central systems idea is hierarchical control:

1. use MIG to isolate workloads whose interference is difficult to control;
2. use MPS to create a configurable compute partition inside an instance;
3. use kernel scheduling to exploit transient idle resources and protect high-priority execution.

### Profiling and scheduling

MMK profiles workloads to characterize kernel duration, resource demand, and pairwise interference. The scheduler uses these profiles to decide which workloads should share a MIG instance, what MPS allocation each receives, and how kernels should be ordered inside the shared execution domain.

The runtime distinguishes latency-sensitive and throughput-oriented work. It prioritizes latency-sensitive kernels while allowing best-effort kernels to fill unused capacity. Hybrid placement reduces the search space and the amount of interference the kernel scheduler must handle. Compared with a global kernel scheduler, the MIG layer prevents the most damaging pairs from sharing low-level resources; compared with pure MIG, the inner layers recover capacity that would otherwise remain stranded.

MMK must coordinate changes across very different time scales. MIG configuration is coarse and slow, so it is suitable for stable workload grouping. MPS allocation changes are finer but still not intended for every kernel. Kernel scheduling handles short-term variation. This separation of time scales is one of the paper's most important design choices.

### Evaluation and contributions

The evaluation compares the hybrid design against standalone MIG, MPS, and existing sharing/scheduling mechanisms using diverse DL training and inference workloads. It examines throughput, latency/SLO behavior, utilization, and interference under different workload mixes. The results show that the hybrid design provides better aggregate efficiency than relying on any one mechanism while retaining stronger isolation than unconstrained MPS or stream concurrency.

The broader contribution is not a new hardware primitive but an orchestration framework for composing existing and software-defined controls. MMK demonstrates that coarse isolation and fine scheduling are complementary: isolation reduces the scheduler's burden, and kernel scheduling recovers utilization lost by isolation.

### Comparison with Orion

Orion uses a single fine-grained software layer: it intercepts launches, predicts kernel resource behavior, and schedules compatible kernels across application queues. MMK places a similar kernel-level control plane at the bottom of a **three-level hierarchy**. MIG first separates workloads requiring stronger isolation, MPS assigns SM shares within each partition, and the kernel scheduler handles transient contention inside those envelopes. MMK thus restricts the interference domain before applying Orion-like launch scheduling.

The hierarchy addresses a limitation that Orion cannot fully solve in software: two kernels may use different execution units yet still interfere through L2 cache, HBM bandwidth, or faults. MIG can isolate several of those resources, making performance more predictable. The tradeoff is substantially greater operational complexity and less agility: MIG layouts are coarse and slow to change, MPS quotas add another control loop, and profiling must inform decisions at all three levels. Orion is simpler and can react directly at every launch, while MMK is preferable when strong isolation and predictable service quality justify a more static outer partitioning layer.

### Assessment

MMK belongs to the target category because kernel interception and scheduling are part of its essential mechanism. However, its novelty is broader than the interception layer: it is a policy for deciding when to use MIG, MPS, or software scheduling. For related-work positioning, MMK is best described as a **hybrid hierarchical GPU-sharing framework**, whereas Orion and Tally focus more directly on fine-grained runtime execution control. A limitation is the operational complexity of coordinating MIG configuration, MPS processes, profiling, and kernel scheduling. The design also inherits MIG's generation-specific constraints and MPS's incomplete isolation of shared caches, memory bandwidth, and interconnect resources.

## 4. LithOS: An Operating System for Efficient Machine Learning on GPUs

> **In brief:** LithOS is a GPU operating-system layer that schedules individual TPCs and atomized kernel fragments rather than whole kernels or processes. By integrating TPC stealing, hardware right-sizing, and power management, it provides work-conserving isolation and substantially improves colocated ML latency and throughput.

### Problem and motivation

LithOS argues that GPU resource management needs an operating-system-like layer rather than isolated mechanisms for priority, sharing, or power control. Existing software typically schedules whole kernels, streams, processes, or inference requests. These units are too coarse: a kernel can occupy the GPU long after a latency-critical request arrives, static SM partitioning strands capacity, and allocating a fixed hardware width ignores the fact that different kernels saturate at different numbers of SMs/TPCs.

LithOS treats the GPU's Texture Processing Clusters (TPCs) as schedulable compute units. Its goal is to provide transparent spatial scheduling, work conservation, fine-grained preemption-like behavior, performance isolation, hardware right-sizing, and power management within one runtime.

![LithOS motivation: MPS concurrency still produces head-of-line blocking and idle GPU capacity.](assets/paper_figures/lithos_motivation.png)

*Motivation (Figure 3): even when MPS admits two workloads concurrently, long kernels and fixed resource use can delay later requests and leave capacity unused. LithOS seeks finer control than stream- or process-level multiplexing.*

### Architecture

LithOS introduces four mechanisms:

![LithOS mechanism: a GPU OS layer exposing TPC scheduling, kernel atomization, right-sizing, and power management.](assets/paper_figures/lithos_mechanism.png)

*Main mechanism (Figure 7): unmodified frameworks send work through LibLithOS to a unified device driver. The runtime jointly controls TPC placement, atomized kernel execution, hardware width, and power instead of treating them as separate policies.*

- **TPC scheduler.** Each workload receives a logical allocation of TPCs. The scheduler maps ready work onto physical TPCs and can steal idle TPCs from a workload that cannot currently use its allocation. This separates reservation from instantaneous use: quotas provide isolation, while stealing makes the system work-conserving.
- **Kernel atomizer.** A kernel is transformed into smaller atoms, each containing a subset of the original thread blocks. Atoms are scheduled independently, so LithOS can change a workload's physical width between atoms instead of waiting for a full kernel to finish. This reduces head-of-line blocking and provides a software analogue of fine-grained preemption.
- **Hardware right-sizing.** More TPCs do not always reduce kernel latency proportionally. LithOS predicts each kernel's scaling curve and allocates the smallest width that stays within an acceptable performance loss, leaving the remaining TPCs for other workloads.
- **Transparent power management.** The runtime chooses frequency/power settings using the characteristics of in-flight work. It can reduce frequency for kernels that are insensitive to compute frequency while maintaining latency or throughput targets.

Applications interact with a userspace layer that submits work to LithOS queues. The GPU-side execution engine decouples logical kernel work from physical thread-block execution. Kernel atomization makes the original grid schedulable in pieces, and the TPC scheduler dynamically maps those pieces to resources.

### Scheduling behavior

LithOS combines reservations, priorities, and TPC stealing. Latency-sensitive work receives enough dedicated capacity to meet its target. Best-effort work consumes otherwise idle TPCs but yields as atom boundaries permit. Right-sizing prevents a single workload from receiving resources beyond its saturation point. Because decisions occur below the model/request layer, LithOS can support inference-inference and inference-training colocation without requiring each ML framework to implement a custom scheduler.

Its performance models operate online and are kernel-dependent. The system learns how execution time scales with TPC count and uses this information for subsequent atoms. This is more adaptive than a fixed per-process MPS percentage, although early invocations and shape changes can cause prediction errors.

### Implementation and evaluation

LithOS is implemented in Rust and evaluated against NVIDIA mechanisms and research systems including MPS, MIG, time slicing, REEF, TGS, priority-based execution, and Orion. Workloads include Llama 3, GPT-J, BERT, RetinaNet, YOLO, MobileNet, DLRM, and training/inference combinations.

For inference stacking, LithOS reports up to **13x lower tail latency than MPS**. Relative to the best-performing prior system, it reduces tail latency by about **4x** while improving aggregate goodput by approximately **1.3x**. For inference-training stacking, it reduces tail latency by **4.7x** relative to MPS; against the strongest prior baseline, it reduces tail latency by about **1.18x** while improving aggregate throughput by roughly **1.35x**.

Right-sizing saves approximately one quarter of GPU capacity on average for less than 4% performance loss. The transparent power-management mechanism reduces total GPU energy by approximately one quarter at around 7% performance cost. The evaluation also studies atom size, prediction errors, kernel-dependent scaling, scheduler overhead, and the contribution of TPC stealing.

### Comparison with Orion

Orion is principally a host-side kernel scheduler: it intercepts launches, selects whole kernels from application queues, and relies on the existing GPU scheduler for block placement and concurrent execution. LithOS moves the control boundary deeper by presenting an operating-system abstraction over GPU **TPCs**. Its atomizer breaks kernels into smaller pieces, and its TPC scheduler explicitly controls their physical width, reservations, and borrowing. LithOS can therefore reclaim idle spatial capacity or reduce head-of-line blocking even within a kernel, whereas Orion must wait for an admitted kernel to complete and cannot directly assign its blocks to a chosen TPC subset.

LithOS also covers a broader resource-management scope. Kernel-dependent right-sizing avoids allocating TPCs beyond a kernel's saturation point, and power management incorporates energy into the scheduling policy; neither is a central Orion mechanism. In exchange, LithOS requires a more invasive and architecture-specific GPU-OS substrate, kernel atomization support, and online scaling models. Orion is easier to deploy as an interception-based scheduler and is useful when whole-kernel pairing is sufficient. LithOS is the more complex design, but it offers finer spatial control and a path toward a unified GPU resource manager rather than a dedicated colocation scheduler.

### Assessment

LithOS is highly relevant to kernel scheduling, but architecturally more ambitious than an interposition-only scheduler. It presents a unified GPU OS abstraction and uses transparent kernel atomization to enable sub-kernel scheduling. Its strength is the integration of isolation, work conservation, capacity right-sizing, and energy management. Its risks are implementation complexity, dependence on GPU-specific low-level mechanisms, and the cost of maintaining compatibility with proprietary drivers and rapidly changing architectures. It is a particularly useful comparison point for any new work claiming that a CUDA interception layer should evolve into a general resource-management substrate.

## 5. SMore: Enhancing GPU Utilization in Deep Learning Clusters by Serverless-Based Co-Location Scheduling

> **In brief:** SMore colocates short serverless inference functions with long-running training jobs using learned interference predictions, deadline-aware admission and placement, and proactive model warming. It improves cluster utilization at the workload level, but does not intercept or schedule individual CUDA kernels.

### Problem and motivation

SMore operates at a higher layer than the kernel schedulers in this collection. It observes that long-running DL training jobs often leave GPU capacity unused because of communication, synchronization, input-pipeline stalls, or inherently low SM demand. It proposes filling this capacity with short-lived serverless inference functions.

The paper divides workloads into two classes:

- **Serverful workloads:** long-running training jobs that form the stable background allocation.
- **Serverless workloads:** short inference functions with deadlines and bursty arrivals.

Naive colocation can slow both classes, cause serverless deadlines to be missed, and introduce large cold-start overheads when models must be loaded into GPU memory. SMore addresses all three issues with degradation prediction, admission/placement scheduling, and proactive model warming.

![SMore motivation: colocation degradation depends strongly on the training-inference workload pair.](assets/paper_figures/smore_motivation.png)

*Motivation (Figure 2): the heat maps show highly non-uniform slowdowns across training/inference pairs, including extreme outliers. Utilization alone is therefore insufficient for safe serverless admission and placement.*

### Co-location degradation predictor

SMore uses a two-stage predictor. A pairwise model estimates the degradation when one training workload and one inference function are colocated. Its 24-dimensional input concatenates twelve features from each model, including SM utilization, memory utilization, FLOPs, memory footprint, model depth, and operator composition. Random Forest performs best among the tested regressors.

![SMore mechanism: offline profiling and online degradation-aware serverless scheduling.](assets/paper_figures/smore_mechanism.png)

*Main mechanism (Figure 4): offline solo/co-location profiles train pairwise and multi-way degradation models. Online, the gateway, scheduler, and monitor use those predictions plus live cluster state to admit, place, and continually update serverless functions.*

For one training job colocated with multiple functions, SMore avoids profiling the combinatorial space. Its multi-way predictor represents total degradation as an online-updated weighted sum of pairwise degradation estimates. This is inexpensive and data-efficient, but assumes that higher-order cache, bandwidth, and saturation effects can be approximated by additive terms.

The predictor is initially trained offline and updated with observed online outcomes. With 20% of 1,024 collected samples, the Random Forest predictor reaches an RMSLE of approximately 0.3; the reported full comparison gives RMSLE 0.305 and MAE 0.473.

### Degradation-aware scheduling

For each function type, SMore computes a priority proportional to expected utilization gain divided by expected degradation. The scheduler then performs admission control and placement. A request is admitted only when GPU memory is sufficient, the function can meet its latency SLO, and predicted degradation of the serverful workload remains within the configured bound, set to approximately 10% in the paper.

At low load, the scheduler searches broadly for the GPU with minimum predicted interference. At high load, it samples a bounded number of GPUs and chooses the first feasible placement. This hybrid policy keeps decision latency low when request rates are high. The scheduling overhead is about 0.01 ms for eight GPUs and remains below 1 ms in simulations with 1,024 GPUs; under high load at that scale, the reported latency is approximately 0.1 ms per request.

### Cold-start management

The LS-LSTM prewarmer predicts whether and how many requests for a function will arrive in the next interval using both long- and short-term patterns. It preloads models before predicted arrivals and offloads models when continued idleness is expected. In one evaluated trace, it reduces cold-start rate by 15% at the cost of 10% more idle resource time. In a burstier trace, it keeps the cold-start rate nearly unchanged while reducing wasted loaded-model time from 75% to 32%.

### Evaluation

The prototype is implemented in more than 3,000 lines of Python. The primary testbed uses RTX 3090 GPUs; an additional experiment combines SMore with MISO-managed MIG instances on an A100 40 GB GPU. Workloads cover CV, NLP, recommendation, and multi-task models, while arrival patterns are derived from scaled Azure Functions traces.

Across multi-GPU configurations, average GPU utilization improves by **3%-34%**. Gains are smaller when the training workload already has high utilization. Compared with Random, EDF-util, and a modified ElasticFlow baseline, SMore achieves a favorable balance among deadline-satisfaction ratio, utilization improvement, serverful degradation, and serverless degradation. In the MIG experiment, it admits approximately 30% additional serverless functions while maintaining degradation within the configured range.

### Assessment

SMore is not an Orion-style kernel scheduler. It schedules functions and chooses GPUs; the underlying GPU-sharing mechanism performs actual concurrent execution. It does not intercept CUDA launches, transform PTX, preempt kernels, or schedule thread blocks. The paper is relevant as an upper-level policy that could feed a kernel scheduler: SMore could decide which workloads should colocate, while Orion, Tally, Bless, or Hummingbird could enforce the resulting priorities inside each GPU. Its limitations include the additive multi-way interference model, mostly non-LLM workloads, reliance on scaled rather than native GPU-serverless production traces, and evaluation centered on RTX 3090 with only a limited A100/MIG extension.

## 6. Usher: Holistic Interference Avoidance for Resource-Optimized ML Inference

> **In brief:** Usher jointly chooses model batch sizes, replication, GPU types, and placements using kernel-based compute/memory estimation, then merges compatible operator graphs to reduce cache interference. It is an upper-layer multi-model inference optimizer rather than a runtime kernel scheduler.

> **Filename note:** the directory file is named `Harli SLO-Aware Co-location of LLM Inference and PEFT Finetuning.pdf`, but its contents are the OSDI 2024 Usher paper. It should not be cited as Harli.

### Problem and motivation

Usher targets multi-model inference serving. Increasing batch size raises compute utilization but also raises latency and memory demand; it cannot independently tune compute and memory utilization. Spatially colocating multiple models can fill both resources, but existing systems often optimize only compute allocation and suffer severe cache and bandwidth interference. The paper reports that, in representative GPUlet and AlpaServe configurations, most colocated models retain no more than 55% of their standalone goodput.

Usher treats GPU compute capacity and memory capacity as a two-dimensional packing problem. It aims either to maximize goodput in a fixed cluster or to minimize GPU cost in an elastic cluster while satisfying all latency SLOs.

![Usher motivation: existing model-serving systems leave computation or memory capacity underutilized.](assets/paper_figures/usher_motivation.png)

*Motivation (Figure 1): Shepherd, GPUlet, and AlpaServe often fail to fill computation and memory simultaneously. Usher treats both dimensions as first-class placement constraints rather than tuning only batch size or compute allocation.*

![Usher mechanism: resource estimation, interference-aware scheduling, and operator-graph merging.](assets/paper_figures/usher_mechanism.png)

*Main mechanism (Figure 9): the kernel-based estimator predicts each model's compute and memory requirements; the scheduler jointly chooses configuration and placement; the graph merger then mitigates cache interference among colocated models.*

### GK-Estimator

Usher avoids per-model profiling with a GPU-kernel-based estimator. It expands an ONNX operator graph into a kernel-level graph using a pre-profiled mapping from known operators to the GPU kernels they invoke. A time regressor predicts kernel duration, while a memory regressor predicts intermediate-data footprint from batch size, tensor dimensions, FLOPs, weights, and GPU type. Kernels with predicted start times within a small threshold are treated as potentially concurrent; Usher sums their resource demands and takes the maximum over concurrent sets.

The stacked regression design combines Lasso, kernel-ridge, gradient-boosting, and XGBoost models. The reported computation- and memory-requirement accuracy is **99.98%**, with an estimation time of 31.6 ms. The comparison profiling approach is exact but requires about 4.8 hours and $42.7 for the evaluated model/configuration space. Usher therefore moves profiling cost from each new model to a reusable per-operator/per-kernel model.

### Interference-aware scheduler

For each model, Usher jointly chooses batch size, replication degree, GPU type, and placement. It classifies models as compute-heavy or memory-heavy and groups complementary models so that total compute demand is close to total memory demand. Within each group it enumerates candidate batch-size and replica configurations and uses a multidimensional best-fit heuristic to pack replicas.

A notable result is that increasing replication degree can reduce total GPU count even when one replica could satisfy the SLO. Splitting the request stream across more replicas permits smaller batches and footprints; those replicas may then pack more efficiently with complementary models. This illustrates why workload division, batch sizing, and placement must be optimized jointly.

### Operator Graph Merger

Compute and memory packing alone does not eliminate GPU-cache interference. Usher observes weight overlap among models with related CNN or Transformer architectures. The Operator Graph Merger identifies structurally similar operator graphs and common weight submatrices, then creates a merged graph that executes operations sharing cached weight regions together. Weight elements are considered equivalent only within a very small tolerance, approximately \(10^{-7}\), which keeps average accuracy loss near **0.0003% per request batch**.

Graph merging takes approximately 2.3 s; grouping takes 0.1 s, scheduling about 0.51 s, and model loading about 0.63 s. Since the paper reschedules only when request rate changes substantially, typically at intervals of 45-300 s in its traces, these control-plane costs are amortized.

### Evaluation

Usher evaluates twenty models, including CNNs, GNMT, BERT, GPT-2, and Llama-2 13B, on homogeneous and heterogeneous AWS GPU clusters and in large-scale simulation. Baselines include Shepherd, GPUlet, and AlpaServe.

The paper reports up to **2.6x higher goodput** and **3.5x better cost efficiency**. In heterogeneous non-fixed clusters it achieves 2.8x-3.5x lower cost, 19%-24.1% higher compute utilization, and 25.2%-40.3% higher memory utilization. Under varied SLOs, fixed-cluster goodput is 2.2x-2.7x higher, while non-fixed deployments use 3.2x-4.3x fewer GPUs. Ablation shows that compute/memory classification, workload division, model ordering, and graph merging all contribute materially; removing graph merging alone causes a large goodput loss, with the complete system achieving 55.7% higher goodput than that variant.

### Assessment

Despite using kernel-level graphs for estimation, Usher is not a runtime kernel scheduler. It does not intercept and reorder application kernel launches, implement preemption, or schedule thread blocks. Its decisions concern model configuration, replication, and GPU placement on a tens-of-seconds time scale. It is also not specifically an LLM-serving system: Llama-2 and GPT-2 are evaluation workloads, but the design does not center on autoregressive decoding, KV-cache management, continuous batching, or prefill/decode separation. Usher is best classified as interference-aware multi-model inference placement and graph optimization. It could complement a kernel scheduler by selecting good model combinations before a lower-level runtime controls their execution.

## 7. Cross-Paper Comparison

| Paper | Primary scheduling object | Control layer | CUDA/kernel interception | Kernel transformation | Main objective | Fit to Orion-style kernel scheduling |
|---|---|---|---|---|---|---|
| Bless | Kernel squads and SM configurations | Host runtime + multiple GPU contexts | Yes, CUDA runtime wrapping | No general PTX rewrite; controls launches/configured contexts | Reclaim bubbles while enforcing quotas | Very high |
| Hummingbird | Split-kernel sub-grids | CUDA Driver interposition runtime | Yes | Yes, PTX splitting and offset injection | Bubble-aware microsecond preemption and SLO protection | Very high |
| Tally | Kernels and logical thread blocks | CUDA virtualization server | Yes | Yes, slicing and persistent preemption | Non-intrusive performance isolation | Very high |
| MMK | MIG groups, MPS shares, and kernels | Hierarchical cluster/GPU runtime | Yes at fine-grained layer | Limited/implementation-dependent | Combine isolation and utilization across three mechanisms | High |
| LithOS | Kernel atoms and TPC allocations | GPU OS/runtime layer | Transparent low-level submission control | Yes, kernel atomization | Isolation, work conservation, right-sizing, energy | Very high |
| SMore | Serverless function admission and GPU placement | Cluster/serverless scheduler | No | No | Harvest idle training capacity | Low; complementary upper layer |
| Usher | Models, batches, replicas, and GPU placement | Inference-serving control plane | No runtime launch scheduling | Operator-graph merging, not scheduling transformation | Multi-model goodput and cost efficiency | Low; complementary upper layer |

## 8. Synthesis and Research Opportunities

The five low-level systems reveal a common architecture: transparent interception creates a global observation and control point; profiling or prediction estimates kernel behavior; a policy selects priorities or resource shares; and a mechanism translates policy into enforceable execution units. Their key difference is the unit of control. Bless uses kernel squads and context configurations. Hummingbird schedules PTX-rewritten sub-grids, while Tally adds slicing and persistent-worker yield at logical-block boundaries. LithOS generalizes fine-grained execution into kernel atoms scheduled onto TPCs. MMK composes kernel scheduling with coarse MIG isolation and intermediate MPS partitioning.

Three unresolved problems recur across the papers.

First, **preemption granularity and overhead are inseparable**. Smaller slices or atoms reduce blocking but increase launch, synchronization, counter, and scheduling overhead. Current systems choose granularity through profiling or heuristics. A promising direction is an online controller that predicts the marginal SLO benefit and overhead of the next finer granularity under changing shapes and request mixes.

Second, **SM isolation does not isolate shared resources**. L2 cache, HBM bandwidth, power limits, copy engines, PCIe/NVLink, and collective communication remain sources of interference. Bless and Usher explicitly model some interference; LithOS right-sizes compute; MMK uses MIG where stronger isolation is needed. A unified scheduler should jointly reason about compute allocation and bandwidth/cache pressure rather than treating SM count as the complete resource state.

Third, **upper- and lower-layer schedulers are disconnected**. SMore and Usher make workload-level placement decisions using predicted degradation, while Tally, Hummingbird, Bless, and LithOS control actual kernel execution. A hierarchical system could use workload-level models to choose colocations and reserve SLO budgets, then use a kernel scheduler to enforce those budgets and return online interference measurements. This feedback loop could correct prediction errors and adapt to phase changes without exhaustive pairwise profiling.

For LLM systems specifically, the most important extension is phase awareness. Prefill, decode, attention, MoE routing, KV-cache movement, and collectives have different compute, bandwidth, and latency behavior. Existing generic kernel schedulers can intercept them, but do not automatically understand TTFT/TPOT semantics or distributed parallelism dependencies. Hummingbird begins to bridge this gap through API-pattern bubble detection across vLLM, SGLang, llama.cpp, DeepSpeed, and Megatron. A strong next step would combine phase-aware request scheduling with portable kernel/thread-block control and explicit modeling of HBM and interconnect contention.

## 9. Overall Classification

For a literature review centered on transparent CUDA interception and fine-grained GPU scheduling, the primary papers are **Tally, Hummingbird, Bless, LithOS, and MMK**. Tally and Hummingbird are the closest methodological matches because both intercept CUDA execution and transform kernels to create software-controlled preemption points. Bless is particularly relevant for quota-aware kernel-squad scheduling and bubble reclamation. LithOS offers the broadest OS-level abstraction, while MMK provides the clearest argument for combining coarse hardware isolation with fine software scheduling.

SMore and Usher should be treated as adjacent rather than central work. Their value lies in admission, placement, and interference prediction above the kernel layer. They provide useful mechanisms and objective functions for deciding *what should share a GPU*, whereas the primary kernel-scheduling papers decide *how that sharing should be executed safely and efficiently*.
