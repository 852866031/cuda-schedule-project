
## 2026-08-18 20:15 — Finetuning sweep done; then the user asked the right question

**9/9 points complete** for the fetch-on-demand arm. The cost model predicts every point to
within 0.4%, using only the PCIe bandwidth measured in the *inference* study's Phase 0:

| layers | GiB | measured | predicted | tok/s | MFU | peak VRAM |
|---|---|---|---|---|---|---|
| 0 | 0.00 | 1.144s | 1.144s | 3580 | 55.0% | 23.04 G |
| 16 | 6.50 | 2.103s | 2.109s | 1947 | 29.9% | 16.54 G |
| 32 | 13.00 | 3.063s | 3.074s | 1337 | 20.5% | 10.04 G |

Each GiB freed costs **0.1476 s/step** measured against 0.1484 predicted. Loss is flat across
all points, and `layer_fetches` is exactly `n_offload x 2 x steps`, confirming two crossings
per layer per step.

**Then the user asked: did we overlap compute with transfer? We had not.** The hooks fetched
on demand on the compute stream, so transfer and compute serialised. The evidence was sitting
in the data and I had not read it that way: measured step time matched the *fully additive*
prediction to within 0.4%, which is only possible if overlap is zero. What I had measured was
arm A2 from the plan (the pessimistic bound), and I had been presenting it as the cost of
offloading rather than the cost of *naive* offloading.

The gap is not small. With perfect overlap the step is `max(compute, transfer)` rather than
their sum, so up to **19 layers (7.7 GiB) would be free** -- transfer hides entirely behind the
1.144 s of compute -- and beyond that degradation restarts from a much better base.

Implemented prefetch (arm A1): a side CUDA stream stages layer N+1 while layer N computes,
with events for ordering and `record_stream` so the caching allocator does not recycle a
buffer that is still in flight on the compute stream. Direction flips in backward, where the
recomputed forward runs descending.

First result at 16 layers offloaded: **2.101s -> 1.350s per step, 1949 -> 3034 tok/s, 1.56x**,
identical loss, +0.4 GiB for double-buffering. That recovers two-thirds of what naive
offloading gave away. Full prefetch sweep running now.

Lesson worth keeping: a cost model matching measurement to 0.4% felt like a triumph, but a
*perfectly additive* fit was actually evidence of a missing optimisation. Agreement with a
model is only as good as the model's assumptions.
