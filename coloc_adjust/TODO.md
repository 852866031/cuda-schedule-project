# TODO: colocation interference (what happens, why, how to improve)

Started Nov 9 2026. From here on the reports (`report/REPORT.md`, `report/REPORT_zh.md`) concentrate
on three questions about colocation, and every item below feeds one of them:

- **A. What happens** when the decode step and the tenant share the GPU.
- **B. How to explain it.**
- **C. How to improve it.**

Nothing below has been run unless it is listed under "Evidence so far". Results go into both reports
when an item is done (next free figure number: Figure 7 / 图 7). Background reading:
`report/RELATED_bandwidth.md`.

Rules for every run: either GPU may be used (GPU 1 only was lifted on Nov 9); check that the GPU
is free first (another project's vLLM tests used both on Nov 9); one MPS daemon and pipe directory
per GPU (`GPU=0 ./run_knee_corun.sh …`); weights under `/mnt/storage/huggingface`; 500 W cap as in the earlier runs;
rerun any point that looks odd before reporting it.

Terms: *decode* = the synthetic Llama-3-8B decode step (15.8 GB read per step); *tenant* = the
LoRA fine-tune worker; *bandwidth* = GPU memory bandwidth (SMs ↔ GPU memory), measured with the
GPM counter `DRAM_BW_UTIL` (1 % ≈ 17.8 GB/s, calibrated on the decode step).

## Results so far (Nov 9, GPU 1)

Run `results/knee_bw2.csv` (all series, counter on), short reruns `knee_bw2_chkA/B.csv`, tables in
`results/knee_bw2_summary.md` (byte accounting) and `results/knee_bw2_crossrun.md` (five runs side by
side). Script: `analyze_knee_bw.py`. Not in the reports yet; no figure yet.

1. **The total does not grow when the tenant joins.** Decode alone moves 1.33–1.40 TB/s (75–79 % on
   the counter). Decode plus tenant moves 1.26–1.41 TB/s (71–79 %). So the 23 % the tenant uses alone
   is not added on top; it comes out of decode's share. The counter never went above 79 %.
2. **On disjoint SMs decode loses about one GB/s per tenant GB/s** when the tenant has 64 or 96
   SMs: 0.85–1.14 over five runs. With a small tenant (8–16 SMs) and on shared SMs decode loses
   more than the tenant moves: 1.2–2.5×.
3. **Sharing SMs costs the tenant.** Its throughput next to decode is 34–73 % of its solo rate on
   shared SMs and 68–88 % on disjoint SMs.
4. **The tenant's bandwidth is not proportional to its SMs** (table under H3): 170 → 8 SMs is 21×
   fewer SMs and 8× less bandwidth; traffic per SM rises from 2.4 to 6.3 GB/s.
5. **Counter check.** Calibration over 54 decode-alone points: 1 % = 17.79 GB/s, median error 0.5 %.
   In co-runs, decode GB/s + tenant GB/s matches the counter within 1 % at B = 1 and is 3–4.5 % below
   it at B = 4 and 8 (40–60 GB/s not accounted for; cause open).
6. **Repeatability.** Shared-SM points with the tenant on a green context repeat within 0.1 % over
   three runs. Points where decode runs on a "remaining" green context fall into two levels 1–3.6 %
   apart, fixed for a whole run (also seen for decode alone at B = 4). The extra 8-SM context is
   not the cause (checked with reruns A and B). Shared, tenant on the full device: 13.9 ms in three
   long runs, 14.6 ms in the two short reruns. Causes open (E2.6).

### E1.2 results: synthetic aggressor in place of the tenant (Nov 9, GPU 1)

Runs `results/aggr_corun.csv` and `aggr_corun_r2.csv` (two restarts, 155 points each: step times
agree within 0.02 % in the median, slowdowns within 2 points), controls `aggr_corun_chunk32.csv`
and `aggr_corun_chunk256.csv`, tables in the `*_summary.md` next to each. Scripts: `bw_aggr.py`
(the aggressor: a float32 matrix-vector product that reads a 2 GiB buffer in 64 MB calls, paced to
a target GB/s, on the first M SMs), `aggr_corun.py` (driver), `analyze_aggr.py`. Decode always runs
on SMs the aggressor does not use. Not in the reports yet; no figure yet.

7. **An SM limit does not stop the harm.** Unpaced, next to decode:

   | Aggressor SMs | Aggressor alone | Next to decode (B = 1 / B = 8) | Decode slowdown (B = 1 / B = 8) |
   |---|---|---|---|
   | 8 | 491 GB/s | 205 / 298 GB/s | +8 % / +16 % |
   | 16 | 894 | 382 / 506 | +15 % / +32 % |
   | 32 | 1,263 | 585 / 732 | +33 % / +57 % |
   | 64 | 1,335 | 858 / 972 | +76 % / +100 % |
   | 96 | 1,342 | 955 / 1,059 | +105 % / +124 % |

   With 256 MB calls the 96-SM aggressor takes 1,231–1,307 GB/s and decode runs 3.1–3.5× slower.
8. **At a fixed SM count the slowdown is linear in the aggressor's GB/s** (50–400 GB/s; largest
   residual of a straight-line fit 2 points, most under 1).
9. **It is not flat in SMs.** The same GB/s costs decode less when it comes from fewer SMs.
   Slope of the fit, % decode slowdown per 100 GB/s:

   | Aggressor SMs | B = 1, decode on the rest | B = 1, decode on 106 SMs | B = 8, decode on the rest | B = 8, decode on 106 SMs |
   |---|---|---|---|---|
   | 96 | 7.0 | | 6.7 | |
   | 64 | 6.8 | 6.8 | 7.2 | 7.2 |
   | 32 | 5.3 | 5.7 | 6.4 | 6.1 |
   | 16 | 3.8 | 4.4 | 5.7 | 5.2 |
   | 8 | 3.8 | 4.0 | 4.6 | 4.5 |

   Byte for byte would be 7.5. Holding decode on 106 SMs gives the same picture, so the change
   comes from the aggressor's SM count, not from decode's. GB/s is still the larger variable
   (8× across the grid against 1.5–1.8× from the SM count).
10. **The total grows.** The counter goes from 68–79 % (decode alone) up to 91 % in the co-run.
    Decode loses 0.46–0.78 GB/s per aggressor GB/s at B = 1 and 0.57–0.84 at B = 8 (200–800 GB/s
    points). This is unlike the tenant, where the total stayed at 71–79 % and decode lost 0.85–1.14
    GB/s per tenant GB/s at 64–96 SMs and 2.0–2.4 at 8 SMs. Per byte, the tenant costs decode
    1.1–1.8× what the reader costs at 64–96 SMs and 3–5× at 8 SMs (one tenant run at 8 SMs;
    tenant GB/s taken from its bytes per token alone). So bytes read do not explain all of the
    tenant's harm at few SMs (E2.7).
11. **The aggressor is slowed too, and its SM count sets its ceiling.** Unpaced next to decode
    (B = 1) it reaches 42 % of its solo rate on 8 SMs, 43 % on 16, 46 % on 32, 64 % on 64, 71 %
    on 96. Per SM that is 18–37 GB/s at 8–32 SMs. The tenant moves 4.9–6.3 GB/s per SM there,
    far under this ceiling, so the ceiling does not bind for it.
12. **The byte accounting closes.** Decode GB/s + aggressor GB/s is within 1.2 % of the counter in
    248 of 252 points, B = 1 and B = 8 (the other four are 1.3–3 % low, all with the aggressor idle,
    gone, or at 50 GB/s). Aggressor alone: 1.007–1.008 at every rate. So the 3–4.5 % gap seen with the tenant
    at B = 4 and 8 (result 5) is not on the decode side or in the counter; the likely cause is the
    tenant's bytes per token differing between alone and co-run (not checked).
13. **Call size control.** The same GB/s in 256 MB calls (a quarter of the calls): decode step
    within 0.8 %. In 32 MB calls (twice the calls, up to 11.9 k/s): decode 0–3 % slower. The
    slowdown follows GB/s; the call rate adds little.
14. **A fixed step at B = 8.** With decode on 138–162 SMs the fit has an intercept of +1.1 to +1.6 %:
    decode pays this as soon as the aggressor issues any kernels, before the per-GB/s part. Baselines
    for decode on a "remaining" context also moved by about 1 % within a run (two levels, as in
    result 6), so ratios at 50–100 GB/s are less certain than at 200 GB/s and above. Feeds E2.6.
15. **A reader alone reaches 88–90 % on the counter** (1.58–1.61 TB/s, 256 MB calls, 96–170 SMs);
    decode alone stops at 68–79 %. So decode does not fill the GPU memory bandwidth (first answer
    to E2.2).

### E2.7 results: aggressor modes (Nov 9, GPU 1)

`aggr_corun.py --modes`, `analyze_modes.py`; `results/aggr_modes*.{csv,md}`, rerun of 20 points in
`results/aggr_modes_r2*` (step times within 0.5 % of the first run). Aggressor on 8 or 96 SMs,
decode on the rest, B = 1 and 8, 4 s windows.

16. **Per byte, reads, writes and copies cost about the same from 96 SMs** (5.4–8.2 % per 100 GB/s
    at 100–400 GB/s). **From 8 SMs writes and copies cost more than reads**: 6.1–6.6 against 4.0 %
    per 100 GB/s at B = 1, 7.1–7.8 against 5.5–6.3 at B = 8. Unpaced on 96 SMs a writer or copier
    (1.0–1.15 TB/s) slows decode +197–211 %, a reader (0.93–1.04 TB/s) +105–120 %.
17. **Kernel launches with no bytes cost 1–6 %**: 2–5 k/s +0.7–2.1 %, 20–50 k/s +2.1–3.8 %,
    100–132 k/s +3.0–5.8 %; the same from 8 and 96 SMs; power and clock unchanged.
18. **Arithmetic with no counted bytes** (512×512 matmul): from 8 SMs +1.4–2.6 % (B = 1) and
    +3.8–6.9 % (B = 8), clock unchanged. From 96 SMs unpaced +8.6–9.8 % with the board at the 500 W
    cap and the clock down 8–9 %. At B = 8 the counter shows 2.5–6.6 % more bytes than decode's own
    (8 SMs, and 96 SMs at 25 % speed where the slowdown is +6.9 %, reproduced): the job may stop
    being L2-resident next to a B = 8 decode. Hypothesis only.
19. **The byte results are not a power-cap effect**: 379–473 W, clock within 1 % at all read,
    write and copy points. A writer alone reaches 86.6 % on the counter (1.55 TB/s on 96 SMs), a
    copier 77.3 %, a reader 73–75 % in 32 MB calls (second answer to E2.2).
20. **Reading for the tenant gap** (result 10): on 8 SMs the tenant's bytes as reads predict
    1.4–2.6 %, as writes 2.1–3.2 %, against 5.4–6.4 % measured. Launches at 5–20 k/s (1.4–2.8 %)
    and arithmetic on 8 SMs (1.4–6.9 %) are of the size of the rest. Not proven: the tenant's
    launch rate is unmeasured and the parts were not run together (needs E2.4, E2.5).

## The three hypotheses

### H1. Limiting the tenant's SMs does not limit the bandwidth harm to decode

Claim: what slows decode is the tenant's memory traffic, and an SM limit acts on it only through
the tenant's throughput. At the same tenant throughput, few SMs and many SMs hurt decode alike.

Status after E1.2: the first half holds (traffic from 8 disjoint SMs slows decode; at a fixed SM
count the slowdown is linear in GB/s). The second half needs a correction: the same GB/s costs
decode about half as much from 8–16 SMs as from 64–96 SMs (results 7–9), and with the real tenant
the trend is the opposite (result 10).

Evidence so far
- Decode slowdown per 1 k tenant tokens/s: 1.3–2.4 % on shared SMs, 1.1–2.3 % on disjoint SMs
  (`results/knee_corun*.csv`, two runs).
- Published: Llama-3.1-8B decode on its own 64 SMs slows up to 1.3× next to a copy kernel on the
  other SMs (Elvinger et al.); see `report/RELATED_bandwidth.md` §2.4.

Experiments
- [ ] **E1.1 Same traffic, two knobs.** Bring the tenant to the same tokens/s (so the same GB/s)
  in two ways: (i) an SM limit, (ii) all SMs with pauses between micro-batches. Compare decode
  step time, mean and p99, at 3 traffic levels × B = 1, 4, 8. If H1 holds the two curves coincide.
  About 15 min GPU. Figure: decode slowdown against tenant GB/s, one line per knob.
- [x] **E1.2 Controlled aggressor** done (results 7–15). Grid run: GB/s ∈ {50, 100, 200, 400, 800,
  unpaced} × SMs ∈ {8, 16, 32, 64, 96} × B ∈ {1, 8}, decode on the remaining SMs, plus decode held
  on 106 SMs; two restarts and two call-size controls, about 30 min GPU. Slowdown follows GB/s;
  it is not flat in SMs. Still to do: the figure (decode slowdown against aggressor GB/s, one line
  per SM count).
- [x] **E1.3 Co-run bandwidth series** done (`results/knee_bw2.csv`), tenant on 8–170 SMs,
  shared and disjoint. Decode slowdown per 1 k tenant tokens/s: 1.3–2.8 % shared, 1.2–2.7 %
  disjoint (against decode alone on the same SMs).

### H2. Bandwidth is the bottleneck

Claim: decode alone already uses most of the bandwidth it can reach (1.34–1.40 TB/s, 75–79 % on
the counter), so each GB/s the tenant moves is taken from decode.

Status after E1.2: a pure reader reaches 88–90 % alone and the co-run total reaches 91 %, so decode
alone does not fill the bandwidth; still, every GB/s a reader adds costs decode 0.46–0.84 GB/s.
One reading, not tested: decode fills the memory path while its large kernels run and leaves it
idle between them; aggressor bytes that land inside those kernels displace decode's, the rest are
free, and an aggressor on few SMs gets a smaller share while both compete. For 64–96 aggressor SMs
the loss per GB/s (0.63–0.78) is close to decode's own counter level (0.68–0.75). Test: E2.8.

Evidence so far
- Solo knee: decode read rate grows about 32 GB/s per SM up to 32 SMs and is flat from 80 SMs
  (`results/knee_probe.csv`, report §2.6).
- Co-run total stays at decode's solo level (71–79 % of the counter); see results 1 and 2 above.
  The earlier reading "77 % + 23 % ≈ 100 %" was from separate runs and does not describe the co-run.
- Decode's lost GB/s over the tenant's GB/s: about 1 on disjoint SMs with a 64–96-SM tenant,
  1.2–2.5 on shared SMs or with a small tenant. The part above 1 is open.
- Tenant shape changes the cost per token 7× (1.29 → 9.2 % per 1 k tokens/s,
  `results/probe_probe.md`).

Experiments
- [x] **E2.1 Byte accounting during co-run**: first pass done (results 1, 2, 5). Still to do:
  the figure (stacked GB/s, decode and tenant, alone and together). The 3–4.5 % gap at B = 4, 8 is
  not on the decode side or in the counter (result 12).
- [x] **E2.7 Why the tenant costs more per byte than a reader** (result 10), most of all on few
  SMs. Done, results 16–20: no single cause; writes, launches and arithmetic each add 1–3 %.
  Still open: the tenant's launch rate, the L2 hypothesis, whether the parts add up. Original plan: Candidates: many small kernels (launch and MPS path; the call-size control showed a small
  call-rate effect), writes, L2 eviction, tenant bytes per token changing in co-run. First step: a
  second aggressor mode with many tiny kernels and almost no bytes, calls/s swept on 8 and 96 SMs;
  then a write and a copy mode at equal GB/s. About 10 min GPU. Overlaps E2.3.
- [ ] **E2.8 Does the loss per GB/s follow decode's busy fraction?** Put idle gaps between decode's
  kernels (or use B and context length to change how full decode keeps the memory path) and see
  whether decode's lost GB/s per aggressor GB/s moves with decode's counter level. About 10 min GPU.
- [ ] **E2.2 What is the reachable maximum?** First answer in result 15 (a reader reaches 88–90 %
  in 256 MB calls); write and copy alone on 96 SMs in result 19 (86.6 %, 77.3 %); all-SM write and
  copy still to run. Decode stops at 75–79 % of the counter. Run
  read-only, write-only and copy kernels on all SMs and record the highest counter value. Tells
  us whether "100 %" is reachable and how much headroom decode really leaves. About 5 min GPU.
- [ ] **E2.3 Rule out the other shared resources**, each with an aggressor on disjoint SMs
  (compute-only and read against write are answered by results 16, 18, 19: power matters only for
  an unpaced compute job on 96 SMs; a register/L1-only job and a clean L2 test are still to do):
  - compute-only (same SMs and power, data stays in registers or L1): tests the 500 W cap and
    clock frequency path;
  - L2-resident (working set smaller than the L2): tests L2 contention without GPU memory traffic;
  - read-heavy against write-heavy at the same GB/s (Mosaic reports reads hurt more).
  About 15 min GPU. Figure: slowdown per aggressor type at equal SMs.
- [ ] **E2.4 Slowdown by tenant phase.** Time-stamp the tenant's forward, backward and optimizer
  phases and split decode step times by the phase they overlapped. Explains run-to-run variation
  ("the colocated kernels are not the same each time") and which phase costs most. No new
  mechanism needed; about 10 min GPU.
- [ ] **E2.6 Run-to-run levels.** Restart the co-run 10 times on a fixed point (disjoint, 64-SM
  tenant) and record which level each run lands on; then re-create only the decode context, only
  the buffers, only the tenant, to see which one sets the level. Also measure shared/full-device
  at 10 s and at 3 min after tenant start. About 15 min GPU.
- [ ] **E2.5 Per-kernel profiles** of the tenant's and decode's main kernels with Nsight Compute
  (alone; it cannot profile during colocation). Bandwidth, L2 hit rate, read/write split. Feeds
  E2.3 and section C. Feasibility under green contexts not checked yet.

### H3. Fewer SMs make the tenant put more blocks on each SM, so bandwidth is not linear in SMs

Claim: with fewer SMs the tenant's kernels pack their blocks onto the SMs that remain; each SM
then moves more bytes, so the tenant's bandwidth falls much less than its SM count.

Evidence so far (tenant alone, `results/knee_bw2.csv`; the 170/128/96 rows repeat `knee_bw.csv`
within 2 %)

| Tenant SMs | tokens/s | Bandwidth | GB/s | GB/s per SM | MB per token | SM busy, per allowed SM |
|---|---|---|---|---|---|---|
| 170 | 29,006 | 23.3 % | 414 | 2.4 | 14.3 | 59 % |
| 128 | 24,516 | 19.1 % | 339 | 2.7 | 13.8 | 59 % |
| 96 | 21,942 | 16.8 % | 298 | 3.1 | 13.6 | 63 % |
| 64 | 17,947 | 13.7 % | 244 | 3.8 | 13.6 | 69 % |
| 32 | 11,642 | 8.8 % | 156 | 4.9 | 13.4 | 83 % |
| 16 | 6,543 | 5.1 % | 90 | 5.7 | 13.8 | 91 % |
| 8 | 3,490 | 2.8 % | 50 | 6.3 | 14.3 | 96 % |

The knob is weak at the top (170 → 64 SMs: −62 % SMs, −41 % bandwidth) and close to proportional
below 32 SMs (each halving: −42 % to −45 %). Traffic per token is steady at 13.4–14.3 MB, so the
tenant's bandwidth follows its throughput. On the full device the tenant keeps its SMs busy 59 %
of the time; on 8 SMs, 96 %: the same work fills the SMs that remain. This fits H3 but does not
show the block-level mechanism (E3.3).

Experiments
- [x] **E3.1 Tenant-alone bandwidth** at 170–8 SMs done (table above). Still to do: the figure
  (tenant GB/s and GB/s per SM against SM count, with the proportional line).
- [x] **E3.2 Normalisation of `sm_util`**: device-wide. Decode alone on 16 / 32 / 64 SMs reads
  9.0 / 17.6 / 33 %, that is 85–96 % of the allowed SMs. The table above divides by the allowed
  share. `sm_occ` scales the same way; what it counts exactly is not checked.
- [ ] **E3.3 Fixed-grid kernel.** A synthetic kernel with a fixed number of blocks, run on
  M = 8…170 SMs: once with fewer blocks than SMs, once with many more. Shows directly whether
  GB/s follows M or the grid. About 5 min GPU.
- [ ] **E3.4 Where the tenant's time goes.** From E2.5 or timing: share of tenant step time in
  kernels that are memory-bound against compute-bound. Explains why its throughput drops so
  slowly with SMs.

## C. How to improve (after A and B are measured)

- [ ] **C1 Byte budget for the tenant.** The VRAM scheduler gives the tenant a GB/s budget; the
  tenant converts it with its bytes per token and pauses between micro-batches. Compare with the
  existing TPOT-feedback cap and with an SM limit at equal tenant throughput (E1.1 is the first
  half). Nearest prior work: MemGuard, Ripple.
- [ ] **C2 Phase-aware scheduling.** Run tenant work when decode needs little bandwidth
  (prefill, idle gaps); earlier vLLM measurement: prefill about 13 %, decode about 81 % GPU
  memory busy. Needs a signal from the engine.
- [ ] **C3 Fewer bytes per tenant token.** The shape probe shows a 7× range in decode cost per
  tenant token. Find which shape parameters set bytes per token and pick the cheapest that still
  trains.
- [ ] **C4 Finer tenant steps.** Cut the tenant step into smaller units that can stop at operator
  boundaries (LLMStation-style tasklets), so a budget or pause takes effect within one decode step.
- Recorded as out of scope, with reasons: MIG (not on GeForce), memory-channel colouring
  (driver changes and per-model reverse engineering; it would also cap decode), kernel-level
  admission with call interception (large build).

## Report plan

After the hypotheses are tested, the colocation part of both reports is organised as:
1. What happens: decode slowdown against tenant throughput, shared and disjoint SMs (E1.1, E1.3).
2. Explanation: bandwidth accounting (E2.1, E2.2), other resources ruled out (E2.3), phase
   effect (E2.4), why SMs are a weak knob (E3.1, E3.3).
3. Improvement: C1–C4 against the SM limit, at equal tenant throughput.
Each figure uses the quantities the text argues with: tenant GB/s and decode slowdown.

Done on Nov 9: the standalone report `report/REPORT_bandwidth.md` and `report/REPORT_bandwidth_zh.md`
(for non-experts; Diagrams 08–10, figures from `plot_bw_report.py`: `bw_tenant`, `bw_knee`,
`bw_slowdown`, `bw_modes`). The main reports do not carry these results yet.

## Housekeeping (waiting for a go-ahead)

- [x] Redraw the co-run figure on the axes above: `plot_bw_report.py` → `bw_tenant.png`.
- [ ] Commit the aggressor work (`bw_aggr.py`, `aggr_corun.py`, `analyze_aggr.py`,
  `analyze_modes.py`, `plot_bw_report.py`, results), Diagrams 08–10 and the standalone report.
- [ ] Commit the co-run probe (`knee_corun.py`, `run_knee_corun.sh`, results), this file,
  `report/RELATED_bandwidth.md`, and the `HF_HOME` lines in `kvcached_validation/scripts/`.
