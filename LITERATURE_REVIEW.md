# Literature Review: Fine-Grained GPU Sharing and Co-Location Scheduling

## Scope and Reading Guide

This review covers the seven PDF files stored directly in the main `orion-citation` directory. The collection spans several layers of the GPU-sharing stack. Bless, Hummingbird, Tally, MMK, and LithOS directly manipulate GPU execution at kernel, thread-block, stream, context, or SM/TPC granularity. SMore performs cluster-level admission and placement for serverless inference functions. The file named `Harli SLO-Aware Co-location of LLM Inference and PEFT Finetuning.pdf` contains the OSDI 2024 paper *Usher: Holistic Interference Avoidance for Resource Optimized ML Inference*, not Harli; this review follows the paper content and records the filename mismatch.

The review begins with fine-grained temporal sharing for HP/BE workloads, extends the bubble-reclamation perspective to quota-aware spatial sharing, and then moves to TPC-level GPU-OS control. It next covers hierarchical MIG/MPS allocation and concludes with complementary workload-level systems. The papers are reviewed with an emphasis on five questions:

1. What resource-underutilization or interference problem does the paper target?
2. At what layer and granularity does it make scheduling decisions?
3. How does it observe, intercept, transform, or control GPU work?
4. What guarantees and performance improvements does it provide?
5. How closely does it match the Orion-style research direction of transparent kernel interception and scheduling?

## 1. Hummingbird and Tally: Transparent Fine-Grained Temporal Sharing

> **In brief:** Both systems turn best-effort (BE) kernels into **bounded execution units**, place them in high-priority (HP) **GPU bubbles**, and stop admitting BE work when HP work returns. **Hummingbird** emphasizes bubble detection and split-kernel launch control; **Tally** chooses per kernel between slicing and persistent-worker preemption.

### Common motivation and system abstraction

Both systems explicitly target the same two-workload setting on a shared GPU:

- A **high-priority (HP) workload**, typically online inference, whose latency or SLO must remain close to exclusive execution.
- A **best-effort (BE) workload**, typically offline inference or training, whose throughput should be maximized without violating the HP objective.

The HP workload is often bursty: GPU kernels are separated by CPU processing, synchronization, communication, or request-arrival gaps. Exclusive execution protects HP latency but leaves these intervals unused. The common goal of Hummingbird and Tally is therefore to **harvest HP-idle intervals with BE execution while bounding the delay imposed on the next HP kernel**.

At the highest level, both systems try to maximize useful BE work subject to a bound on HP waiting time:

$$
T_{\mathrm{HP\ wait}} \lesssim T_{\mathrm{residual\ BE\ unit}} + T_{\mathrm{runtime}}.
$$

Here, **residual BE unit time** is the remaining execution time of the BE work already admitted when an HP kernel arrives, while **runtime overhead** includes detection, scheduling, and launch costs. The central design problem is therefore to keep the admitted BE unit short enough for rapid HP response, but large enough to avoid excessive fragmentation overhead and preserve BE throughput.

Whole-kernel temporal sharing cannot provide a tight bound because a long BE kernel may already be resident when HP work arrives. CUDA stream priority only prioritizes pending kernels; it cannot evict resident blocks. Neither Hummingbird nor Tally introduces instruction- or warp-level hardware preemption. Instead, both implement **cooperative software preemption** by turning a monolithic BE kernel into smaller units that drain at safe boundaries.

<img src="assets/paper_figures/hummingbird_tally_shared_control_loop.svg" width="440" align="right" alt="Vertical shared control loop for Hummingbird and Tally, with explanations for intercept, divide, harvest, yield, and resume." />

Only after establishing the HP waiting-time objective do the lower-level mechanisms enter the picture. Both systems transparently control HP and BE applications through a five-stage loop: they observe CUDA work, create bounded BE units, place those units in HP-idle intervals, yield when HP work returns, and resume unfinished BE work later.

The two systems mainly differ below this shared abstraction. Hummingbird discovers and predicts HP bubbles, then controls a sequence of ordinary split-kernel launches. Tally treats HP inactivity as the opportunity signal and chooses per BE kernel between slicing and persistent-worker preemption.

Both systems perform important transformations at the **PTX (Parallel Thread Execution)** level. PTX is NVIDIA's virtual GPU instruction-set representation: CUDA, Triton, and other frontends can compile a kernel into PTX, which the CUDA driver later translates into architecture-specific machine instructions (**SASS**) for the target GPU. Rewriting PTX is lower-level and more framework-independent than modifying PyTorch operators or CUDA source, while still retaining concepts such as thread/block indices, branches, and barriers that these systems need to manipulate. However, the approach depends on PTX being available; opaque or precompiled library kernels may require a fallback mechanism.

The shared loop is **cooperative**, not immediate hardware preemption. When HP work arrives, the currently admitted BE unit must still drain. Kernel division matters because it bounds this residual work, while interception lets the runtime stop further BE admission and prioritize the HP launch.

<br clear="right" />

*Shared control loop. Both systems repeatedly transform available HP-idle time into BE progress while bounding how long BE work takes to drain.*

![Unified architecture and execution model for Hummingbird and Tally. Both intercept unmodified applications, convert best-effort kernels into bounded execution units, harvest high-priority idle intervals, and drain before high-priority execution resumes. Hummingbird controls a sequence of split-kernel launches; Tally chooses slicing or persistent-worker preemption.](assets/paper_figures/hummingbird_tally_unified.svg)

*Unified high-level view. Blue units are best-effort work admitted into gaps between red high-priority intervals. Hummingbird bounds interference by keeping at most one split-kernel in the device queue; Tally either does the same with slices or lets persistent workers stop between logical thread blocks.*

### Hummingbird: bubble-aware split-kernel scheduling

Hummingbird intercepts low-level CUDA Driver APIs and combines three ideas:

- **Split-kernel execution.** It profiles an efficient sub-grid size, then rewrites PTX with `blockIdx` offsets so one logical grid becomes a sequence of smaller launches.
- **Bubble-aware admission.** It recognizes short synchronization/communication bubbles from CUDA and NCCL API patterns, while longer idle periods are detected by queue inactivity and optionally predicted from request intervals.
- **Kernel-tick scheduling.** It keeps at most one BE split-kernel in the device queue and launches the next near the previous split's completion. When HP work arrives, a **host-side control flag** stops further launches; the running split drains normally.

This last point is important: Hummingbird does **not** turn the BE kernel into a persistent worker or terminate its current split early. Its preemption bound comes from making each ordinary launch short and limiting queue depth. For long bubbles, **split consolidation** restores larger grids to reduce launch overhead and improve BE throughput.

Hummingbird also provides **priority-aware memory management**, retaining local HBM for HP data while offloading BE pages to NVLink-connected HBM or host DRAM. The paper reports **9.7x/3.5x** higher HP SLO attainment than representative spatial/temporal approaches, less than **1%** loss versus exclusive HP execution, and up to **2.4x** higher BE throughput than REEF.

### Tally: profile-guided slicing or persistent-worker preemption

Tally uses a centralized **client/server CUDA virtualization** layer to intercept operations, preserve CUDA semantics, profile kernels, and dispatch HP work before BE work. It creates bounded BE units using one of two per-kernel mechanisms:

- **Kernel slicing** rewrites block indices and issues small sub-grids one at a time. It is simple but pays repeated launch overhead.
- **Persistent-worker preemption** launches a fixed set of physical worker blocks. Each worker obtains a logical block ID from a global counter, executes that block, and checks a device-visible **preemption flag** before claiming another. On HP arrival, workers drain at logical-block boundaries and later resume from the saved counter.

Persistent conversion can deadlock if some threads return while others wait at a block-wide barrier. Tally's **unified-synchronization transformation** rewrites return and barrier paths so threads exit together at safe boundaries. Tally then profiles both primitives and their parameters, choosing the configuration that maximizes BE performance under a turnaround bound.

Unlike Hummingbird, Tally does not require semantic CUDA/NCCL bubble patterns: generic HP inactivity is sufficient. Its adaptivity concerns **how each kernel yields**, rather than **how long the current bubble will last**. Tally reports **7.2%** average HP P99 latency overhead while retaining more than **80%** of TGS's aggregate throughput.

### Direct comparison between the two systems

The systems belong to the same design family. Their differences are mainly in **opportunity detection** and **yield implementation**:

| Dimension | Hummingbird | Tally |
|---|---|---|
| Opportunity | CUDA/NCCL hints plus large-idle detection/prediction | Generic HP inactivity |
| BE primitive | PTX-rewritten split-kernel launches | Per-kernel slicing or persistent workers |
| HP arrival | Stop the host launch loop; current split drains | Stop slices, or set a device flag and drain logical blocks |
| Adaptation | Profile split size; consolidate in long bubbles | Profile primitive and parameters under a turnaround bound |
| Extra emphasis | Bubble structure and memory offloading | Semantics-safe, general block-level yield |

In short, **Hummingbird controls a sequence of ordinary short launches**, whereas **Tally can additionally stop inside a persistent kernel at logical-block boundaries**. Hummingbird invests more in finding and sizing bubbles; Tally invests more in selecting a safe yield mechanism.

### Comparison with Orion

All three intercept unmodified applications, but **Orion spatially co-executes compatible whole kernels**, whereas Hummingbird and Tally primarily perform **fine-grained temporal borrowing**. Once Orion admits a BE kernel, it normally runs to completion and may interfere with an HP kernel; Hummingbird and Tally pay transformation overhead to bound withdrawal at a split or logical-block boundary. They therefore offer stronger tail-latency isolation when kernels are long or interference is difficult to predict.

Orion can still achieve higher utilization by exploiting complementary resources while HP work is active, and it can schedule opaque library kernels without transformable PTX. The tradeoff is therefore **overlap and lower mechanism cost** in Orion versus **bounded yield and stronger isolation** in Hummingbird/Tally.

### Summary and research implications

The shared recipe is **intercept, divide, harvest, yield, and resume**. Hummingbird contributes workload-aware bubble detection, paced split launches, consolidation, and memory management; Tally contributes a profile-guided choice between slicing and semantics-safe persistent workers.

A natural combined design would use Hummingbird to decide **when and for how long BE may run**, Tally to decide **how the current kernel should yield**, and Orion to decide **when controlled spatial overlap is preferable to temporal withdrawal**.

## 2. Bless: Quota-Aware Reclamation of Spatial GPU Bubbles

> **In brief:** Bless targets multiple GPU tenants with explicit SM quotas. It transparently groups their kernels into short **kernel squads**, selects a profiled MPS allocation for each squad, and lets one tenant reclaim capacity that another tenant cannot currently use - while preserving every tenant's quota-equivalent progress.

### Motivation and background

Bless assumes that **multiple independent workloads share one GPU and each has a predefined compute quota**, expressed as a fraction of SM capacity. For an application with quota *n%*, its performance target is the isolated latency measured while MPS restricts it to *n%* of the GPU.

Static spatial sharing wastes capacity whenever a kernel has too little parallelism, saturates memory bandwidth before using its assigned SMs, or has no successor ready. Bless calls this unused quota a **spatial bubble**. Unrestricted sharing can reclaim it, but hardware-controlled overlap makes latency unpredictable; MIG is too coarse for arbitrary quotas and kernel-scale reconfiguration. Bless therefore seeks **unbiased, work-conserving sharing**: reduce every active request's latency without letting it fall behind its quota-derived target.

![Bless motivation: temporal and static spatial sharing both leave bubbles that ideal sharing could reclaim.](assets/paper_figures/bless_motivation.png)

*Motivation (Figure 1): a nominal spatial allocation can contain unused capacity because kernels do not continuously consume their full quota. Bless seeks the bubble-free schedule while retaining quota-derived progress guarantees.*

### System abstraction and adaptive spatial-temporal sharing

Bless uses a **kernel squad** - a short group of kernels from several active requests - as its scheduling epoch. Squads last roughly **0.7-10 ms**. Bless never splits or preempts a running kernel; it changes the configuration only for subsequent launches.

![Bless mechanism: runtime components for forming kernel squads and selecting execution configurations.](assets/paper_figures/bless_mechanism.png)

*Main mechanism (Figure 5): wrapped CUDA requests enter the multi-task scheduler; the configuration determiner chooses a profiled allocation; and the concurrent-kernel manager dispatches the resulting squad through resource-configured GPU contexts.*

Its control path is **PROFILE → TRACK/GROUP → CONFIGURE → RECLAIM**:

- **PROFILE.** Bless records quota-isolated latency and per-kernel duration/SM usage under different MPS allocations; its A100 setup profiles 18 allocations in 6% increments.
- **TRACK/GROUP.** The scheduler compares each request's real-time progress with its profile and repeatedly adds a kernel from the request with the smallest relative progress to the next squad.
- **CONFIGURE.** Two estimators predict squad duration under candidate strict SM partitions and unrestricted overlap; Bless selects the predicted fastest configuration.
- **RECLAIM.** In **semi-spatial execution**, the first fraction of a squad uses SM-restricted contexts. After those kernels finish, the remainder launches through unrestricted contexts and may reclaim the whole GPU. The testbed uses a 50% split ratio.

Dynamic SM allocation is implemented with **pre-created CUDA/MPS contexts**, not Green Contexts. Each client receives one unrestricted context and several `cuCtxCreate_v3` contexts with different SM-count affinities. Kernel functions are injected into these contexts, and client memory is allocated and mapped in advance. The runtime then routes each kernel to the context selected for its squad.

To preserve correctness across context-local queues, the manager waits for a client's restricted-context kernels to complete before submitting its later kernels to the unrestricted context; it monitors completion and uses `cudaDeviceSynchronize()` for cross-context coordination. Thus, dynamic sharing means **switching the launch target at complete-kernel boundaries**, not migrating a resident kernel or changing its affinity mid-execution.

The innovation is the combination of **progress-aware squad formation**, **profile-guided SM configuration**, and **fast restricted-to-unrestricted context switching**.

### Evaluation and limitations

The 5,000-line C++ system wraps CUDA APIs and reports a **21.1%-37.3% average latency reduction** over prior sharing systems while preserving quotas. Its main limitations are reliance on stationary, offline-profiled workloads, a finite set of MPS configurations, memory overhead from multiple contexts, and inability to preempt an unusually long kernel already admitted to a squad.

### Comparison with Orion

Orion classifies and overlaps compatible whole kernels to improve throughput. Bless instead forms progress-aware **kernel squads** and executes them through profiled MPS configurations to enforce quota-derived latency targets. Orion is a leaner mechanism for beneficial pairing; Bless adds proportional-allocation guarantees and explicit spatial control at the cost of offline profiling and multiple contexts.

### Summary

Bless turns the gap between **allocated SM quota** and **useful execution** into reclaimable capacity at millisecond-scale squad boundaries while keeping every request on its quota-derived progress trajectory.

## 3. LithOS: TPC-Level Scheduling with Transparent Kernel Atomization

> **In brief:** Like Hummingbird and Tally, LithOS transparently colocates high-priority (HP) and best-effort (BE) workloads by intercepting CUDA launches and making long BE kernels yield at sub-kernel boundaries. Its distinctive contribution is to combine this temporal control with **per-atom TPC allocation**, so BE work can borrow idle physical compute units and return them when HP work arrives.

### Motivation and limitations of existing sharing

LithOS starts from the same HP/BE setting as Hummingbird and Tally. HP inference requires latency close to isolated execution, while BE inference, training, or finetuning should use otherwise idle resources. Whole-kernel temporal sharing leaves GPU bubbles; direct MPS colocation fills those bubbles but can let a long BE kernel delay subsequent HP kernels because stream priority cannot evict work already resident on the GPU.

LithOS adds a spatial problem to this familiar temporal one. Applications may receive quotas in **TPCs** - physical compute clusters containing a small number of SMs - but a fixed quota is often underutilized. BE work should be able to borrow those idle TPCs, yet the system must return them promptly when their owner becomes active. Once a kernel is submitted to the hardware device queue, neither its priority nor its physical TPC allocation can be changed. LithOS therefore needs both **short, controllable BE execution units** and **launch-time control over their eligible TPCs**.

![LithOS motivation: MPS concurrency still produces head-of-line blocking and idle GPU capacity.](assets/paper_figures/lithos_motivation.png)

*Motivation (Figure 3): whole-kernel execution and fixed spatial shares create both idle capacity and long blocking intervals. LithOS seeks work-conserving TPC allocation with bounded sub-kernel scheduling units.*

### System abstraction and key mechanisms

LithOS interposes on the CUDA Driver API through `LibLithOS`, following the same basic interception pattern as the earlier kernel schedulers. An unmodified application's asynchronous CUDA calls return normally, but its kernels first enter per-stream software **launch queues** rather than immediately filling hardware device queues. Dispatcher threads decide when to forward each kernel or atom to the GPU, while sync queues and a tracker thread bound outstanding work and record completion.

![LithOS mechanism: a GPU OS layer exposing TPC scheduling, kernel atomization, right-sizing, and power management.](assets/paper_figures/lithos_mechanism.png)

*Main mechanism (Figure 7): LithOS sits below unmodified ML frameworks and jointly controls when work is dispatched, which TPCs execute it, how a kernel is atomized, and how much hardware capacity it receives.*

Its main execution path is **INTERCEPT → ATOMIZE → ASSIGN TPCs → DISPATCH → STEAL/RETURN**.

#### TPC scheduling and stealing

The TPC scheduler maintains each application's quota, the set of currently available TPCs, and predicted completion timers for work already assigned to every TPC. Before dispatch, it chooses an allowed TPC set for the kernel or atom. At the scheduling interface, this set acts as a **per-launch TPC mask**: it constrains the GPU's thread-block distributor so blocks from that launch may be assigned only to the selected TPCs. It masks physical dispatch destinations, not `blockIdx`, CUDA threads, or parts of the kernel code.

If a quota owner has no ready work, another application may temporarily use its TPCs through **TPC stealing**. LithOS limits the amount of outstanding stolen work, avoids TPCs predicted to remain busy for too long, and submits stolen work at lower stream priority. When the owner becomes active, the scheduler removes those TPCs from subsequent BE atoms. It cannot change the mask of an atom already running, so the owner still waits for that atom's residual execution time.

The paper clearly exposes per-launch TPC assignment as its enforcement abstraction, but defers the lowest-level mechanism for injecting the TPC mask into NVIDIA launch metadata to a separate technical report. It is not implemented by repeatedly changing an MPS percentage or by Green Context reconfiguration.

#### Kernel atomization: LithOS-style kernel slicing

LithOS's **Kernel Atomizer** is its version of kernel slicing. A long logical kernel is divided into atoms covering non-overlapping ranges of the original grid's thread blocks. Rather than rewriting PTX, LithOS launches a Prelude kernel with the original grid. Every block flattens its `blockIdx`; blocks within the atom's range call the original kernel entry point, while all others return immediately. For a 64-block kernel, two atoms may cover `[0,32)` and `[32,64)`, ensuring each original block body executes exactly once.

Because this mechanism needs neither source nor PTX, it works with arbitrary frameworks and closed-source libraries such as cuDNN. The tradeoff is that every atom launches the original grid shape and pays for range checks and early-exiting blocks. LithOS therefore predicts kernel duration, chooses an `atom_duration`, and disables or coarsens atomization when its overhead exceeds the scheduling benefit.

Each atom receives its own TPC mask. Consequently, a logical BE kernel can use borrowed TPCs in one atom and a smaller guaranteed set in its next atom after HP work arrives. LithOS still provides cooperative software preemption: it stops future atoms but does not interrupt the current one.

Although all three systems divide a long BE kernel into shorter scheduling units, they construct those units differently:

| System | Division mechanism | Grid actually launched for each unit | How the blocks to execute are selected |
|---|---|---|---|
| Hummingbird | PTX rewriting + sub-grid slicing | Only the smaller grid required by the current slice | A block-index offset maps the slice-local `blockIdx` back to its position in the original grid |
| Tally slicing | PTX rewriting + sub-grid slicing | Only the smaller grid required by the current slice | Like Hummingbird, an offset reconstructs the original block coordinates |
| Tally persistent | Persistent kernel | A fixed number of long-lived worker blocks | Workers dynamically acquire logical blocks and check a yield flag at logical-block boundaries |
| LithOS atomization | Prelude wrapper + block-range filtering | The complete original grid is launched for every atom | Every block checks whether it belongs to the atom's range; selected blocks call the original kernel, while the others immediately exit |

A **Prelude wrapper** is a small entry function placed in front of the original kernel entry point. LithOS launches this wrapper with the original grid dimensions. Each physical block flattens its original `blockIdx`, compares it with the atom's interval `[begin, end)`, and either invokes the unchanged original kernel body or returns immediately. The wrapper therefore changes **which blocks execute**, without rewriting the kernel's PTX or changing the `blockIdx` and `gridDim` values observed by selected blocks. Its cost is that every atom instantiates the full grid, including blocks that perform only the range check and exit.

Thus, Hummingbird primarily asks **how to fit split kernels into known bubbles**, Tally asks **which yield implementation is best for each kernel**, and LithOS asks **how to jointly schedule a kernel slice and the physical TPCs on which it may execute**.

#### Auxiliary mechanisms

LithOS also learns per-kernel scaling with TPC count and assigns the smallest width within a configurable latency-slip bound (**right-sizing**). A similar online model lowers frequency for insensitive kernel sequences (**DVFS**). These extend the same OS control plane to capacity and energy, but atomization plus TPC scheduling form the core HP/BE sharing mechanism.

### Evaluation and limitations

The Rust prototype supports PyTorch, TensorFlow, JAX, TensorRT, Triton, and closed-source cuDNN workloads. In inference-training stacking, HP latency averages **1.19x** isolated execution and aggregate throughput improves by roughly **1.35x** over TGS. Right-sizing saves **26%** GPU capacity on average, while DVFS saves **26%** energy.

The main atomization costs are Prelude/extra-launch overhead (about 10% BE throughput in the reported experiment) and prediction error when selecting atom size for unseen or dynamic operators. TPC masking introduces additional limitations. First, ordinary CUDA does not expose an official per-kernel TPC-mask API; LithOS depends on GPU- and driver-specific launch metadata whose lowest-level injection mechanism is not documented in the paper, creating portability and maintenance risks across GPU and CUDA generations. Second, allocation is quantized at TPC granularity, often grouping multiple SMs, and logical TPC IDs do not necessarily reveal their physical GPC placement. Third, changing a mask affects only later atoms: it cannot evict CTAs already resident on a borrowed TPC, so HP still waits for the current atom to drain. Finally, disjoint TPC masks isolate compute placement but not shared resources such as L2 cache, HBM bandwidth, memory controllers, copy engines, or power domains; they therefore provide weaker performance and security isolation than MIG.

### Comparison with Orion

Orion intercepts and pairs compatible whole kernels but leaves thread-block placement to the GPU. LithOS changes both dimensions: it exposes sub-kernel atoms as scheduling units and assigns each atom a physical TPC set. This provides explicit quotas, TPC stealing, and a shorter withdrawal boundary, at the cost of substantially more runtime machinery and architecture-specific enforcement.

### Summary

LithOS can be summarized as **the same transparent HP/BE interception loop, extended with Prelude-based kernel slicing and per-atom TPC masks**. Its central innovation is making logical sub-kernel work and physical GPU width jointly schedulable.

## 4. MMK: A Hybrid Scheduling Framework for Fine-Grained GPU Sharing for Deep Learning Applications

> **In brief:** MMK manages latency-sensitive **online jobs** and throughput-oriented **offline jobs** at three different time scales. MIG creates coarse hardware-isolation domains, MPS multiplexes jobs and oversubscribes compute inside each domain, and an intercepted whole-kernel scheduler uses online-job slack to control short-term contention.

### Motivation and limitations of single-level GPU sharing

**Problem space.** MMK colocates latency-sensitive **online inference jobs** with long-running **offline training or throughput jobs**. It seeks to preserve online QoS while reducing offline completion time, makespan, and stranded GPU capacity.

**Why multiple levels are necessary.** Hummingbird, Tally, and LithOS start with HP and BE applications already sharing an execution domain and focus on fast BE withdrawal. MMK must first decide **which jobs should share**, then **how much isolated hardware each group receives**, and finally **which kernels may run concurrently**. These decisions operate at different resource scopes and time scales.

**MIG alone.** MIG separates compute, L2, memory controllers, bandwidth, and memory capacity, providing strong isolation between sharing groups. However, its few legal partition shapes cause overprovisioning and fragmentation, idle capacity is difficult to borrow across instances, and reconfiguration requires stopping affected execution.

**MPS alone.** MPS enables process concurrency and assigns active-thread percentages inside a MIG instance. These percentages limit compute capacity but do not isolate L2 or memory bandwidth; the same percentage can also produce different slowdowns across jobs and MIG sizes. Reconfiguring an existing client is too disruptive for per-kernel control.

**Kernel interception alone.** Launch-level scheduling can follow short phases, but cannot create memory/cache isolation, change the outer physical allocation, or optimize global job placement. Conversely, MIG and MPS cannot react at kernel time scales. MMK therefore composes all three mechanisms:

**MIG isolation envelope → MPS compute-sharing envelope → whole-kernel admission and ordering**

![MMK motivation: SM utilization and execution time vary substantially across MIG and MPS allocations.](assets/paper_figures/mmk_motivation.png)

*Motivation (Figures 3-4): different MIG sizes leave different amounts of stranded capacity, while the slowdown caused by an MPS limit changes with both the job and its enclosing MIG partition. No single fixed spatial configuration consistently provides isolation, utilization, and online QoS.*

### Runtime execution chain: from profiles to repartitioning

**1. Workload assumptions and profiles.** MMK assumes jobs can be classified in advance as **online** (latency-sensitive, with a QoS target) or **offline** (training/throughput work without a strict deadline). Before deployment, it profiles representative jobs and kernels under multiple MIG sizes, MPS limits, and colocation conditions. The profile contains job-level SM, memory-bandwidth, and L2 utilization; kernel grid/block dimensions, register count, and shared-memory use; and the resource features of potential colocated jobs. These samples train a random-forest predictor. At runtime, the current job features, candidate MIG/MPS configuration, and candidate colocated set are its inputs. Its outputs are predicted execution time for an online job or kernel and predicted throughput for an offline job. This assumes the production models and shapes are sufficiently similar to the profiled set.

**2. First GPU partition.** When the scheduler first allocates a set of pending jobs, it sorts them by memory demand and forms a resource-demand vector for each. It computes **Partition Entropy** over possible partition granularities: the heuristic prefers a number of MIG instances that both avoids over-fragmenting the GPU and can separate jobs with different resource behavior. This selects a small set of legal MIG layouts to examine, such as `[4g, 3g]`, instead of exhaustively testing every layout, placement, and MPS percentage.

**3. Assigning jobs to the candidate MIG instances.** For each retained layout, MMK groups the jobs into its MIG instances so that jobs within a group have relatively similar resource demand and heterogeneous groups are separated. It then constructs an MPS candidate for every group and uses the predictor to estimate the resulting performance. The design favors the candidate with the best estimated system throughput while protecting online QoS through the later MPS and kernel-scheduling rules. One important paper detail: MMK clearly specifies memory-demand sorting and the entropy objective of reducing within-group resource heterogeneity, but does **not** provide a fully explicit per-job `choose MIG instance` placement algorithm. It is best viewed as entropy-guided heuristic grouping, followed by predictor-based candidate evaluation, rather than an exact bin-packing solver.

**4. Set MPS ceilings inside each MIG instance.** MMK separates the colocated jobs into online and offline sets. It starts with baseline MPS shares based on SM utilization, then gives a job an additional **kernel-aware oversubscription** allowance when the *other* jobs have variable aggregate kernel use. Their low-utilization periods are treated as temporary slack. As a result, the configured MPS percentages may sum to more than 100%; they are ceilings for opportunistic sharing, not simultaneous reservations of more SMs than exist. The paper uses predicted aggregate **offline** throughput to compare these MPS candidates, while online jobs are handled as QoS-constrained clients whose configuration should remain stable.

![MMK mechanism: hierarchical predictor, MIG partition scheduler, and per-partition kernel scheduler.](assets/paper_figures/mmk_mechanism.png)

*Main mechanism (Figure 7): profiling supplies performance estimates; the macro scheduler establishes MIG isolation domains and MPS ceilings; the micro scheduler controls whole-kernel release inside those envelopes.*

**5. Intercept and release whole kernels.** An `LD_PRELOAD` layer intercepts CUDA API calls from both online and offline jobs and holds kernel launches in software queues before forwarding them to CUDA. It does not choose the MIG layout, change MPS ceilings, rewrite PTX, or split a kernel; its unit is a **whole kernel**. Its role depends on the current workload mix:

- **Online vs. online:** allow concurrent online kernels only when predicted contention and execution fit within the minimum remaining online QoS slack.
- **Online vs. offline:** use remaining online slack to form an offline execution budget; release offline kernels only until their accumulated predicted duration reaches that budget, then delay later offline kernels.
- **Offline vs. offline:** with no online QoS constraint, package kernels from multiple offline jobs whose predicted aggregate resource use fits within the partition budget, then launch a package concurrently.

Thus interception applies to **both** job classes, but its priority policy is asymmetric: online kernels are protected, while offline kernels are delayed or batched to exploit safe capacity. Since already-released kernels are not preempted, it cannot remove the residual blocking time of a long offline kernel.

**6. Reconsider MIG partitioning only at a safe time.** Job arrivals, completions, and resource-mix changes can make the current outer layout inefficient, so MMK may rerun the entropy-pruning and candidate-evaluation procedure. It does **not** immediately repartition on every event. MIG reconfiguration is applied only when a safe window exists for the affected area - in the paper, when no online job is running in that partition or its run queue is empty. MPS reconfiguration is also kept off the online critical path and is mainly triggered by offline-job arrival or completion. Kernel interception is therefore the fast path; MPS adjustment is the medium-frequency path; MIG repartition is the rare structural path.

The central idea is **MIG for long-lived isolation, MPS for medium-term opportunistic compute sharing, and intercepted whole-kernel scheduling for short-term online-QoS protection.**

### Evaluation and limitations

The prototype contains about 3,000 lines of C++ and Python and supports unmodified PyTorch and TensorFlow applications through CUDA API interception. Experiments use two A100 80 GB GPUs and include CNNs, BERT/Transformer workloads, and several small LLMs. Relative to MIGER, MMK reports 28% lower average JCT, 32% lower makespan, and 35% higher system throughput. Kernel scheduling contributes only about 0.08% average execution-time overhead, although the less frequent MIG and MPS reconfiguration paths account for approximately 1.5% and 3.2% of the workload lifecycle, respectively.

**What MMK cannot do and its limitations.** MMK cannot preempt or atomize a kernel that has already been released, so an online arrival can still wait for the residual duration of a long offline kernel; its finest control is whole-kernel admission, not the bounded slice/atom boundary provided by Hummingbird, Tally, or LithOS. MPS percentages are capacity ceilings rather than explicit physical placement and do not isolate L2 or memory bandwidth, while MIG provides stronger isolation only through coarse, generation-specific shapes that remain expensive to change. The three-level controller also depends on an offline-trained predictor generalizing to new models, input shapes, kernels, interference patterns, and GPU architectures; prediction error can produce either QoS violations or conservative underutilization. Finally, the evaluation is centered on two A100 GPUs and generic DL jobs: including several LLMs does not make MMK a token-, prefill/decode-, KV-cache-, or communication-aware LLM serving scheduler.

### Comparison with Orion

Orion operates primarily at one fine-grained software layer: it intercepts launches, predicts kernel resource behavior, and pairs compatible whole kernels. MMK places a related interception layer below a macro allocator. MIG first limits the interference domain, MPS supplies per-process compute ceilings, and MMK then gates whole kernels according to online slack or packages offline kernels according to aggregate resource demand. Both retain whole kernels, but Orion emphasizes kernel compatibility and overlap, whereas MMK emphasizes coordination between job placement, spatial configuration, and QoS-aware admission.

The hierarchy addresses interference that Orion cannot eliminate through launch ordering alone: MIG separates several cache and memory resources before software attempts concurrency. The cost is much greater operational complexity and less agility. Orion can reconsider every launch directly, while MMK must coordinate profiler predictions, legal MIG layouts, MPS process state, safe reconfiguration windows, and kernel queues.

### Summary

MMK can be summarized as **offline-learned performance modeling plus online hierarchical control**. It first chooses who should share through MIG, then how much compute they may opportunistically use through MPS, and finally when their whole kernels may enter the GPU. Its novelty is not a new preemption primitive, but a coordinated policy that assigns existing mechanisms to the resource scope and time scale where each is most useful.

## 5. Interference-Aware Workload Co-location: SMore and Usher

> **In brief:** After MMK decides a resource envelope and a lower-level runtime decides when kernels may run, a remaining question is **which workloads should share a GPU in the first place**. SMore and Usher address that question through interference prediction and workload-level placement, rather than slicing, intercepting, or physically placing individual CUDA kernels.

### Common problem and relation to the kernel schedulers

Both systems begin from the same observation behind MMK: apparent spare SM capacity is not automatically safe capacity. Two workloads can have complementary average compute demand yet still hurt one another through GPU memory capacity, HBM bandwidth, L2/cache behavior, or transient resource peaks. The key question is therefore not only *can this GPU fit another workload?*, but *can it do so without violating an SLO or causing unacceptable degradation?*

Their control unit is above the CUDA runtime: a serverless function in SMore, and a model configuration/replica in Usher. They decide **whether and where to colocate**; an Orion-, Hummingbird-, or LithOS-like runtime could subsequently control the actual kernel execution inside the selected GPU. In contrast to MMK, neither system dynamically partitions an individual GPU with MIG/MPS and then intercepts whole-kernel launches to protect an online job.

### SMore: admit serverless inference into training slack

SMore places short, deadline-constrained **serverless inference functions** alongside long-running **serverful training jobs**. The training job is the stable background allocation, while the serverless functions arrive in bursts and need a latency SLO. SMore's target problem is therefore to harvest training-side slack without violating either the inference SLO or an allowed training-degradation bound; it must also avoid cold starts caused by loading an infrequently invoked function model into GPU memory.

Its execution chain is: **profile/predict degradation → check SLO and training impact → select a GPU → prewarm the model when demand is expected**. A pairwise model takes features from a training model and an inference model - including SM/memory use, FLOPs, footprint, depth, and operator composition - and predicts their mutual degradation. When several functions would share one training job, an online-updated multi-way model combines those pairwise estimates rather than profiling every workload combination.

For every arriving function request, SMore checks available memory, predicted function completion relative to its SLO, and predicted degradation of the colocated training job. It admits the request only if these constraints are met, then chooses the GPU with low predicted interference; under high cluster load it bounds the placement search to keep scheduler overhead low. Separately, an LS-LSTM predicts future request volume and preloads or offloads function models to trade some idle memory time for fewer cold starts. Thus SMore's core contribution is **degradation-aware admission and placement of bursty inference into stable training allocations**, not a lower-level GPU sharing mechanism.

### Usher: jointly pack compute and memory for multi-model inference

Usher targets a different workload mix: many latency-SLO-constrained inference models, rather than foreground inference plus background training. Its key observation is that batch size is an inadequate single knob: increasing it can increase compute utilization but also increases latency and memory footprint. Meanwhile, naive model colocation can create cache interference even when the models appear complementary. Usher therefore jointly chooses each model's batch size, replica count, GPU type, and placement to maximize goodput in a fixed cluster or minimize cost in an elastic one.

Usher's execution chain is: **estimate model requirements → choose configuration and placement → mitigate residual cache interference**. Its GK-Estimator expands an ONNX operator graph into a GPU-kernel graph using a reusable operator-to-kernel mapping. Learned kernel-time and memory regressors estimate which kernels overlap and the resulting peak compute and intermediate-memory requirement of a new model, avoiding expensive exhaustive profiling of every new model/batch-size/GPU configuration.

The scheduler classifies models as compute-heavy or memory-heavy, groups complementary models, and jointly searches batch size and replica degree before packing replicas with a multidimensional best-fit heuristic. Replication is part of the placement decision: more, smaller replicas can sometimes use fewer GPUs overall because smaller batches reduce footprint and pack better with another model. Finally, if colocated models have structurally similar graphs and sufficiently similar weight submatrices, Usher merges relevant graph operations so shared weight regions are reused in cache. Usher therefore attacks interference through **configuration, complementary packing, and graph-level cache reuse**, not runtime kernel launch control.

### Takeaway and limitations

SMore is an online admission/placement system for a training-plus-serverless cluster; Usher is a slower control-plane optimizer for multi-model inference. Both predict interference, but neither can react to a newly arrived HP kernel by preempting, slicing, atomizing, or masking an already-running BE kernel. SMore's multi-way degradation model approximates higher-order interference from pairwise observations, while Usher's estimates and graph merging depend on known operator mappings and useful structural/weight similarity. Neither is specifically designed around LLM prefill/decode, KV-cache dynamics, or distributed tensor-parallel communication.

## 6. Cross-Paper Comparison

| Paper | Primary scheduling object | Control layer | CUDA/kernel interception | Kernel transformation | Main objective | Fit to Orion-style kernel scheduling |
|---|---|---|---|---|---|---|
| Hummingbird | Split-kernel sub-grids | CUDA Driver interposition runtime | Yes | Yes, PTX splitting and offset injection | Bubble-aware microsecond preemption and SLO protection | Very high |
| Tally | Kernels and logical thread blocks | CUDA virtualization server | Yes | Yes, slicing and persistent preemption | Non-intrusive performance isolation | Very high |
| Bless | Kernel squads and SM configurations | Host runtime + multiple GPU contexts | Yes, CUDA runtime wrapping | No general PTX rewrite; controls launches/configured contexts | Reclaim bubbles while enforcing quotas | Very high |
| LithOS | Kernel atoms and TPC allocations | GPU OS/runtime layer | Transparent low-level submission control | Yes, kernel atomization | Isolation, work conservation, right-sizing, energy | Very high |
| MMK | MIG groups, MPS shares, and kernels | Hierarchical cluster/GPU runtime | Yes at fine-grained layer | Limited/implementation-dependent | Combine isolation and utilization across three mechanisms | High |
| SMore | Serverless function admission and GPU placement | Cluster/serverless scheduler | No | No | Harvest idle training capacity | Low; complementary upper layer |
| Usher | Models, batches, replicas, and GPU placement | Inference-serving control plane | No runtime launch scheduling | Operator-graph merging, not scheduling transformation | Multi-model goodput and cost efficiency | Low; complementary upper layer |

## 7. Synthesis and Research Opportunities

The five low-level systems reveal a common architecture: transparent interception creates a global observation and control point; profiling or prediction estimates kernel behavior; a policy selects priorities or resource shares; and a mechanism translates policy into enforceable execution units. Their key difference is the unit of control. Bless uses kernel squads and context configurations. Hummingbird schedules PTX-rewritten sub-grids, while Tally adds slicing and persistent-worker yield at logical-block boundaries. LithOS generalizes fine-grained execution into kernel atoms scheduled onto TPCs. MMK composes kernel scheduling with coarse MIG isolation and intermediate MPS partitioning.

Three unresolved problems recur across the papers.

First, **preemption granularity and overhead are inseparable**. Smaller slices or atoms reduce blocking but increase launch, synchronization, counter, and scheduling overhead. Current systems choose granularity through profiling or heuristics. A promising direction is an online controller that predicts the marginal SLO benefit and overhead of the next finer granularity under changing shapes and request mixes.

Second, **SM isolation does not isolate shared resources**. L2 cache, HBM bandwidth, power limits, copy engines, PCIe/NVLink, and collective communication remain sources of interference. Bless and Usher explicitly model some interference; LithOS right-sizes compute; MMK uses MIG where stronger isolation is needed. A unified scheduler should jointly reason about compute allocation and bandwidth/cache pressure rather than treating SM count as the complete resource state.

Third, **upper- and lower-layer schedulers are disconnected**. SMore and Usher make workload-level placement decisions using predicted degradation, while Tally, Hummingbird, Bless, and LithOS control actual kernel execution. A hierarchical system could use workload-level models to choose colocations and reserve SLO budgets, then use a kernel scheduler to enforce those budgets and return online interference measurements. This feedback loop could correct prediction errors and adapt to phase changes without exhaustive pairwise profiling.

For LLM systems specifically, the most important extension is phase awareness. Prefill, decode, attention, MoE routing, KV-cache movement, and collectives have different compute, bandwidth, and latency behavior. Existing generic kernel schedulers can intercept them, but do not automatically understand TTFT/TPOT semantics or distributed parallelism dependencies. Hummingbird begins to bridge this gap through API-pattern bubble detection across vLLM, SGLang, llama.cpp, DeepSpeed, and Megatron. A strong next step would combine phase-aware request scheduling with portable kernel/thread-block control and explicit modeling of HBM and interconnect contention.

## 8. Overall Classification

For a literature review centered on transparent CUDA interception and fine-grained GPU scheduling, the primary papers are **Tally, Hummingbird, Bless, LithOS, and MMK**. Tally and Hummingbird are the closest methodological matches because both intercept CUDA execution and transform kernels to create software-controlled preemption points. Bless is particularly relevant for quota-aware kernel-squad scheduling and bubble reclamation. LithOS offers the broadest OS-level abstraction, while MMK provides the clearest argument for combining coarse hardware isolation with fine software scheduling.

SMore and Usher should be treated as adjacent rather than central work. Their value lies in admission, placement, and interference prediction above the kernel layer. They provide useful mechanisms and objective functions for deciding *what should share a GPU*, whereas the primary kernel-scheduling papers decide *how that sharing should be executed safely and efficiently*.
