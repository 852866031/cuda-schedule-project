# From measuring to changing the tenant and its pauses: what was done on Nov 21–22 and what it shows

This document tells the round of work from the start and does not assume the reader saw any of it. It states the
problem first and then proceeds in the order "what was measured → so what was done → what exactly that is → the
result, and against what it is compared".

**Setting.** One RTX 5090 runs two jobs at the same time:

- **decode**: vLLM serving Llama-3-8B to real requests. It is the job to protect.
- **tenant**: a training job (a LoRA fine-tune of Qwen2.5-0.5B) that uses what decode leaves. Its training data are
  randomly generated token sequences; its role is to produce the computing load of a real fine-tune.

**One aim**: the tenant keeps training and decode slows as little as possible.

**Words used throughout.**

| Word | Meaning |
|---|---|
| A step (of the tenant) | One parameter update: 8 sequences of 512 tokens, 4,096 tokens. GPU memory does not allow all 8 at once, so they are computed in several batches (the measurements below use 4 batches of 2 sequences; while running, the tenant chooses the number of batches from the free GPU memory, see 3.4); below, "a batch" |
| Tenant progress | Tokens the tenant trains per second (tok/s) |
| Moving data | Bytes read and written between GPU memory and the compute units (SMs) during computation, in GB. The nominal limit of this path on this card is 1,780 GB per second |
| Budget | A rule for the tenant: "at most so many GB per second on average" |
| A decode step | Decode emits one token for every request being generated; about 12 milliseconds |
| Token gap | The time between two consecutive tokens of one request, i.e. the time of one decode step |
| TPOT | For one request, the average time per token |
| p90, p99 | Sort all values; the value at the 90 % / 99 % position. p99 speaks about the slowest 1 % |

**Every "x % slower" is computed the same way**: a statistic measured with the tenant, divided by **the same statistic
measured with no tenant on the same night in the same setting**, minus 1. For example "p99 +19 %" means: the p99 of
decode's single-step time with the tenant is 19 % above the p99 of decode's single-step time without it.

**The conclusions first** (how each was obtained, and the data, are in the sections named).

1. **The data moved by each part of a tenant step was measured.** A step of the usual tenant moves 57.2 GB, and 55 % of
   it is in the output layer and the loss at the end: four blocks per step of 7.97 GB each, with no place to pause
   inside, and a momentary traffic close to the limit of this card. (Section 2)
2. **From that, three changes were made to the tenant's computation**: the output layer and the loss computed over
   blocks of the vocabulary; the 24 layers of the model compiled one by one; the small operations inside each vocabulary
   block merged into one fused kernel. The training is unchanged. The tenant with all three (the "new tenant" below)
   moves 21.9 GB per step, 62 % less; runs as fast alone; can pause after every layer; needs less than half of the GPU
   memory for the same way of splitting a step into batches. (Section 3)
3. **The rule for pausing was changed to follow the data actually moved.** (Section 4)
4. **On vLLM next to decode, with the tenant doing the same amount of training (about 3,500–3,700 tokens per
   second)**: decode's mean token gap, against the runs without a tenant, is <<V:c1_fix:mean>> next to the usual tenant
   and <<V:cvfm_step:mean>> next to the new tenant; the p99 of decode's single-step time, against the p99 without a
   tenant, is <<V:c1_fix:s99>> and <<V:cvfm_step:s99>>. The best method of Nov 16 (the tenant compiled as a whole): mean
   <<V:c2a:mean>>, p99 <<V:c2a:s99>>. (Section 5.3)
5. **"Cutting the pause into small ones" now works, with a limited effect.** The tenant loses almost no progress; the
   p90 of decode's single-step time falls from <<V:cvfm_step:s90>> above the no-tenant value to <<V:cvfm_w12:s90>>;
   decode's mean does not change (<<V:cvfm_step:mean>> against <<V:cvfm_w12:mean>>) and the TPOT per request does not
   improve. (Section 5.4)
6. **Decode's cost follows the data the tenant moves.** With half and with twice the budget for the new tenant,
   decode's mean is <<V:cvfm_w12_half:mean>> and <<V:cvfm_w12_x2:mean>> (tenant progress <<V:cvfm_w12_half:tok>> and
   <<V:cvfm_w12_x2:tok>> tokens per second); with the tenant on 16, 32 or 64 SMs the cost is almost the same.
   (Sections 5.5, 5.6)
7. **Not done**: the data per part with the tenant next to decode was not measured, the values of the tenant alone are
   used; the cautious automatic budget was not fixed; only the two lighter loads (0.5 and 1 request per second) are
   counted. (Section 7)

---

## 1. What was known before this round, and where it was stuck

Conclusions of the night of Nov 16 (details in `3_c1_c2.md`):

1. How much decode slows follows mainly the data the tenant moves per second.
2. So the tenant gets a budget (50 GB per second on average): on 32 SMs it computes a step at full speed and then pauses
   until its average is back under the budget. Decode's mean cost halves against the earlier method.
3. Compiling the whole tenant with `torch.compile` makes it move about four tenths less data per token, and decode's
   cost falls a little further.

Three things were left unsolved:

- **A. Pausing in small slices did not help.** The idea: instead of one long pause after a whole step, pause briefly
  every few milliseconds, so that every decode step is affected alike. Decode did not improve and the tenant lost 9 %
  of its training.
- **B. The slowest 1 % of decode steps were more than 20 % slower, with no known cause.**
- **C. Only the total data of a tenant step was known (13.47 MB per token on average), not which part of the step moves
  it.** The budget and "moving less data" both turn on this quantity, and there was only one average.

This round starts from C.

---

## 2. First: measure how much data each part of a tenant step moves

### 2.1 How

- The tenant's code has fixed places where a pause can be inserted: the start of each batch, and before the forward and
  before the backward computation of each of the model's 24 layers. The computation between two neighbouring places is
  called **a segment** below. The tenant can pause only between segments, so a segment is the smallest unit that can be
  scheduled.
- The tenant runs **alone** on the GPU (no decode), set up as in the experiments (32 SMs). At every place where it could
  pause, it first waits for the GPU to finish the work issued so far and then emits a named time mark.
- At the same time NVIDIA's Nsight Systems reads the card's memory read and write traffic 10,000 times a second.
- The traffic between two marks is the data that segment moved. It is averaged over many steps.
- Calibration: the GPU copies a 1 GiB tensor several times; the data moved is known and turns the tool's percentage
  into GB.

How reliable: two rounds on the same evening agree within 0.1 % on a step's total (57.13 and 57.17 GB); a second tool
(Nsight Compute, hardware counters kernel by kernel) gives 7.98 GB for the largest segment against 7.97 GB.

### 2.2 What it shows

A step of the usual tenant moves **57.2 GB** in 356 milliseconds (13.95 MB per token; the value of Nov 16, measured in another way, was 13.47 MB).

| Segment | Per batch | Data each | Time each | Data per second while it runs |
|---|---|---|---|---|
| One layer, forward | 24 | 0.11 GB | 1.3 ms | 82 GB |
| One layer, backward | 24 | 0.16 GB | 1.6 ms | 99 GB |
| Output layer and loss (forward and backward) | 1 | **7.97 GB** | 19.3 ms | 413 GB; for about 6 ms of it 500–1,500 GB |

![traffic of one batch](../coloc_adjust/report/figures/prof_trace.png)

*Figure 1 — top: memory traffic over one batch (2 sequences) of the usual tenant, every sample drawn; the shaded stretch
is the segment that holds the output layer and the loss. Bottom: the shaded stretch enlarged, split into its seven
operators.*

Three findings:

1. **55 % of a step's data is the output layer and the loss**, in 21 % of the time. The vocabulary has 151,936 entries:
   the model ends by computing 150 thousand scores (logits) for each of 1,024 positions. That tensor is 311–622 MB and is
   read and written about a dozen times while the loss is computed.
2. **There is no place to pause inside this segment.** Each batch has one block of 7.97 GB; a step has four.
3. **This segment peaks very high.** For about 6 ms in each batch the tenant moves 500–1,500 GB per second, close to the
   card's limit (1,780 GB), while decode itself normally uses about 1,300–1,400 GB per second.

### 2.3 What the three findings mean

- **For problem A (slices did not help)**: the old slicing code assumed every segment was the same amount of work. The
  largest segment is 70 times the smallest. The four blocks of 7.97 GB cannot be cut and upset the timetable; the 9 % the
  tenant lost are exactly these four blocks.
- **For problem B (the slowest 1 %)**: four times per step, for about 6 ms each, the tenant takes nearly the whole memory
  path. The decode steps that meet these milliseconds are the likely slowest 1 %. At this point that was an inference;
  section 5 checks it on vLLM.
- **For where to move less data**: change the output layer and the loss first; they are more than half.

So two things follow: **change the tenant's computation so that it moves less data and can pause everywhere** (section
3), and **change the rule for pausing so that pauses follow the data actually moved** (section 4).

---

## 3. Second: three changes to the tenant's computation

None of the three changes what is trained: the loss is the same and the gradients of the parameters are the same within
numerical error (3.4). All numbers in this section are measured with the **tenant alone** and compared with the "usual
tenant".

### 3.1 Change one: the output layer and the loss computed in blocks of the vocabulary

**Before**: all 1,024 × 151,936 scores are computed at once into one large tensor, and the loss and its gradient are
computed on that tensor.

**Now**: the 150 thousand vocabulary entries are cut into small blocks (4,096 entries each) and processed block by
block, in two passes:

- Pass 1: for each block, compute the scores of all positions on this block; add up only the sum that the normalisation
  needs and note the score of the correct answer; then drop the block's scores. After pass 1 the loss is known.
- Pass 2: for each block, compute the scores again, turn them into probabilities, and add up the gradient that goes back
  into the model.

The large tensor never exists; a block's intermediate results are a few MB. The price is one more matrix product (three
instead of two). It requires that the output layer's weights are not trained, which holds for a LoRA fine-tune.

On Nov 16 another split was tried, grouping the **positions**; it tripled the data, because every group read the 272 MB
of output weights again. Splitting along the **vocabulary** reads the weights twice per batch.

Result (the block size matters):

| Entries per block | Data per step | Against the usual tenant | Largest piece that cannot be paused | Tenant speed alone |
|---|---|---|---|---|
| Usual tenant (no blocks) | 57.2 GB | – | 7.97 GB | 11,511 tok/s |
| 32,768 | 71.4 GB | 25 % more | 1.47 GB | 9,996 |
| 8,192 | 40.4 GB | 29 % less | 0.17 GB | 10,184 |
| 4,096 | 35.0 GB | 39 % less | 0.18 GB | 9,975 |
| 2,048 | 30.4 GB | 47 % less | 0.18 GB | 9,533 |
| 1,024 | 29.6 GB | 48 % less | 0.18 GB | 8,687 |

![block sizes](../coloc_adjust/report/figures/prof_variants.png)

*Figure 2 — data per step by block size. Blue: the 24 layers; orange: the output layer and the loss.*

- Blocks that are too large make it worse; savings start at 8,192 and stop growing below 2,048.
- The largest piece that cannot be paused goes from 7.97 GB to 0.18 GB (one layer of the model). The four large blocks
  are gone.
- The price: the tenant alone is 13–25 % slower, from the extra matrix product and many very small computations.

### 3.2 Change two: each of the 24 layers compiled on its own

`torch.compile` merges chains of small computations into a few, so that intermediate results are no longer written to
GPU memory. On Nov 16 the **whole model** was compiled as one piece: four tenths less data, but no place to pause inside
the compiled model (a batch is one block of 8.6 GB), so slicing could not be used at all.

Now **every layer is compiled separately**: the places to pause between layers stay.

One trap on the way: after the first attempt neither the data nor the speed had changed. PyTorch limits how often the
same piece of code may be recompiled (8 times); only the first 8 of the 24 layers were really compiled and the others
fell back to uncompiled without any message. With the limit raised all 24 are compiled.

Result: the data of the 24 layers goes from 25.5 GB per step to 18.7 GB (27 % less), and the tenant can still pause
after every layer.

A fact that was not known before follows from this: Nov 16's whole-model compilation moves 22.8 GB less per step
(57.2 → 34.4 GB), and compiling the 24 layers can save only 6.8 GB. So about seven tenths of those "four tenths less"
came from the output layer and the loss and only about three tenths from the 24 layers (estimated from what the
per-layer compilation saves).

### 3.3 Change three: the small computations of each vocabulary block merged into fused kernels

In change one every vocabulary block runs about ten small computations (type conversion, exponential, sums, subtracting
the correct answer, …). The ones between the two matrix products are compiled into one merged kernel that keeps its
intermediate results in registers instead of GPU memory.

Result: the output layer and the loss move 3.2 GB per step instead of 8.9 GB (change one, blocks of 4,096; the usual
tenant: 31.4 GB), and the speed comes back.

### 3.4 The three changes together

| Tenant | Data per step | Against usual | Largest piece that cannot be paused | Speed alone | GPU memory needed with this way of splitting (4 batches of 2) |
|---|---|---|---|---|---|
| Usual | 57.2 GB | – | 7.97 GB | 11,511 tok/s | 5.0 GiB |
| Change one only (blocks of 4,096) | 35.0 GB | 39 % less | 0.18 GB | 9,975 (13 % slower) | not measured |
| Changes one + two | 28.1 GB | 51 % less | 0.13 GB | 10,984 (5 % slower) | not measured |
| **Changes one + two + three** | **21.9 GB** | **62 % less** | **0.12 GB** | **11,555 (the same)** | **2.3 GiB** |
| For reference: the whole-model compilation of Nov 16 | 34.4 GB | 40 % less | 8.58 GB | 14,294 (24 % faster) | not measured |

![all tenants](../coloc_adjust/report/figures/night_tenants.png)

*Figure 3 — data per step of each tenant. Purple: after whole-model compilation the output layer cannot be told apart.
Red figures: a piece of more than 1 GB that cannot be paused.*

Below, the tenant with changes one + two + three is called **the new tenant**, and the one with change one only **the
tenant with only the output layer changed**.

Against the usual tenant, the new tenant moves 62 % less data, is as fast, can pause after every layer and needs less
than half of the GPU memory for the same way of splitting a step into batches.

**What exactly "the new tenant" is.** The model (Qwen2.5-0.5B), the LoRA (rank 16, on the q, k, v and o matrices of the
attention), the training data and the step of 8 sequences × 512 tokens are those of the usual tenant. The only
difference is these start-up options:

| Option | What it does |
|---|---|
| `--head vocab --vocab-block 4096` | Change one: the output layer and the loss over blocks of the vocabulary, 4,096 entries per block |
| `--compile-blocks` | Change two: the 24 layers compiled one by one |
| `--head-fused` | Change three: the small computations of each vocabulary block merged into fused kernels |
| `--no-slow-shapes` | The ways of splitting a step that keep part of the data in host RAM are not used (the per-layer compilation does not support them); if not even the smallest way fits in GPU memory, the tenant pauses |
| `--tiers-json results_prof/tiers_cb_vocab4096f.json` | Its own measured table of GPU-memory needs instead of the usual tenant's |

**Into how many batches a step is split is chosen by the tenant while it runs, from the GPU memory free at that
moment**: fewer batches if memory allows (faster), more if not. The tables above were all measured with "4 batches of
2 sequences", so that both tenants are compared in the same way of splitting. The new tenant in each way (measured
alone):

| How a step is split | GPU memory needed | Data per step | Largest piece that cannot be paused | Speed alone |
|---|---|---|---|---|
| 1 batch of 8 | 5.01 GiB | 26.9 GB | 0.68 GB | 15,243 tok/s |
| 2 batches of 4 | 3.25 GiB | 20.5 GB | 0.24 GB | 13,741 tok/s |
| 4 batches of 2 | 2.32 GiB | 21.9 GB | 0.12 GB | 11,555 tok/s |
| 8 batches of 1 | 1.84 GiB | 29.1 GB | 0.08 GB | 9,225 tok/s |

"1 batch of 8" moves more data, so it is switched off for the new tenant. **In the vLLM runs of section 5 the new
tenant computed about 90 % of its steps as "2 batches of 4"** (it needs little memory, so this larger way fits most of
the time) and 7–8 % as "4 batches of 2". The usual tenant computed about 67 % of its steps as "4 batches of 2" and about
22 % as "8 batches of 1". The table of data per part that slices use was measured separately for each way of splitting
the new tenant's step.

**Is the training the same?** Checked on one batch with the real model:

- Loss: the same value for every algorithm (to five decimals).
- Gradients on the LoRA parameters: the new algorithm differs from the original by about 0.5 %; the original run twice
  differs from itself by 0.4 % (the error of half-precision arithmetic).
- Training for 1,500 steps (same data, same random seed, tenant alone): the loss of the new tenant and of the usual tenant
  differs by 0.0009 per step on average and by 0.06 at most; the usual tenant run a second time differs from its first
  run by 0.0014 on average and 0.11 at most. So the new tenant is no further from the usual one than the usual one is
  from itself. The loss at step 1,500 is 11.9357 against 11.9359.

So "the same within numerical error", not bit for bit.

![a batch of the changed tenant](../coloc_adjust/report/figures/prof_trace_vocab2048.png)

*Figure 4 — traffic over one batch with only the output layer changed (blocks of 2,048, blue), on the axes of the usual
tenant (grey line). The stretch that reached 1,500 GB per second is gone.*

---

## 4. Third: a rule for pausing that follows the data actually moved

### 4.1 What it does

The budget is "at most so many GB per second on average". The new rule uses the table of section 2 directly:

1. The table holds the data of every segment of a step, so at any place where the tenant could pause, "how much this
   step has moved so far" is known.
2. By the budget, the time by which that much data "should" have been moved is known too: data so far ÷ budget.
3. Choose a time window (for example 12 ms, about one decode step). Budget × window is the size of one chunk of data.
   Every time a chunk is full, the tenant stops at the nearest place where it can pause and waits until the time
   computed in 2.
4. If a segment ran slower than the budget allows (it is late), the time owed is made up by shorter pauses later, by at
   most one window.
5. At the end of the step the account is settled once more, so that the step's average stays within the budget.

"Compute a whole step, then pause" is a special case: an infinite window, settled only at the end of the step. So the
two ways of pausing below run the same code and differ only in the window:

- **Pause after the step**: compute a step at full speed, then pause until the average is within the budget.
- **Slices**: pause briefly every time "budget × 12 ms" of data has been moved.

### 4.2 Checking the rule by itself first

The tenant runs **alone** under a budget of 50 GB per second (no decode). Two questions.

**Does the tenant get all the progress the budget allows?**

| Way of pausing | Progress the budget allows | Progress obtained |
|---|---|---|
| The old slices of Nov 16 | 3,582 tok/s | 3,522 tok/s (98.3 %; next to vLLM that night only 91 %) |
| The new rule, slices | 3,582 tok/s | 3,581 tok/s (100 %) |

**Is the traffic even?** In any 12 ms (the length of a decode step), how much does the tenant use at most:

| Tenant and way of pausing | Data per second in the worst 12 ms | Times the budget |
|---|---|---|
| Usual tenant, pause after the step | 610 GB | 12 |
| Usual tenant, slices | 610 GB | 12 |
| Only the output layer changed, pause after the step | 107 GB | 2.1 |
| Only the output layer changed, slices | 72 GB | 1.4 |

Conclusion: the new rule no longer loses progress. For the usual tenant slices still cannot even out the traffic,
because the four blocks of 7.97 GB cannot be cut; only once the tenant is changed do slices flatten it. **The two go
together: the tenant must be able to pause everywhere before slicing means anything.**

This section shows only that the rule itself is right. It says nothing about how much less decode slows; that is the
next section.

---
## 5. Fourth: on vLLM, next to decode

The numbers of the three sections above were measured with the tenant alone. This section is where the conclusions
are: both jobs run on one card at the same time, and what counts is how much decode slows and how much the tenant
trains.

### 5.1 How it was measured

- **Decode**: vLLM serves Llama-3-8B and receives an 11-minute sequence of requests. The arrival rate changes in
  phases, from 0.5 to 4 requests per second. Each request carries a prompt of about 6,300 tokens and generates 128
  tokens.
- **Tenant**: on the same card as decode, confined to 32 SMs.
- **Every setting is run once in full, 11 minutes.** Only the phases with 0.5 and 1 request per second are counted
  below (about four and a half minutes together). These two loads are light, decode does not queue by itself, and the
  tenant's effect shows most clearly. The first 10 seconds of each phase and the first 30 seconds of the run are left
  out (the tenant is still starting).
- **Baseline**: runs on the same night with the same setting and no tenant. There were several tonight; they are
  pooled. Among them the mean and the median differ by less than 0.2 %; the p99 of the single-step time differs by
  about 4 %, so for the p99 numbers below a difference within 3–4 points should not be taken as real.
- **How the budget is chosen**: so that every tenant progresses at about the same rate (about 3,500 tokens per
  second); what is compared is then "for the same amount of training, what decode pays". Tenants move different
  amounts of data per token, so the budgets differ in GB: 50 GB per second for the usual tenant, 19.2 GB per second for
  the new tenant.

**The quantities** (each is "the value with the tenant ÷ the baseline's value of the same quantity − 1"):

| Quantity | What it is |
|---|---|
| Mean token gap | The mean of all gaps between consecutive tokens of all requests. How much slower decode is as a whole |
| TPOT median, TPOT p90 | First "average time per token" for each request, then the median and the p90 over all requests. The speed a user sees |
| Single-step time: median, p90, p99 | Token gaps taken one by one, without two kinds that have nothing to do with the tenant: gaps during which another request arrived, and the engine's own occasional gaps of double length or near zero. The rest is grouped by "how many requests were being generated", each group is compared with the same group of the baseline, and the groups are averaged |
| Tenant progress | Tokens the tenant trains per second in the counted phases; the average over the whole 11 minutes is given too |

**Why the single-step time leaves those two kinds out.** With no tenant, the longest 1 % of all token gaps are
already 66 milliseconds, more than 5 times a normal step (about 12 milliseconds). They happen when another request
arrives and the engine first processes its prompt; every request being generated has to wait. This has nothing to do
with the tenant, but it is so large that it hides what the tenant does: whichever tenant is there, "the p99 of all
gaps" changes by only about 1 %. To see the tenant's effect on each decode step these gaps have to be taken out
first. So that not only filtered numbers are given, the tables below also give the p90 and p95 of "all gaps", with
nothing left out.

