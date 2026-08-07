# visualizer/ — animated replay of a colocated run

Replays what the colocator and the GPU actually did, op by op, in the browser:
which app submitted a kernel and when the **interceptor** caught it, what the
**FCFS scheduler** issued in each round and onto which stream, what each
stream is holding that the GPU hasn't started, what is **actually running on
the GPU** (with a CONCURRENT badge when both clients overlap), what has
finished — and, for every timestamp, **which component measured it**.

Modeled on `~/Documents/Projects/gpu-interfere/visualizer` (same dark panel
GUI, client-side replay with play/speed/scrub), but exported as a **single
self-contained HTML file** instead of a server: no dependencies, no network —
anyone (e.g. on a Mac) can open it directly.

## Files

| File | Role |
|---|---|
| `template.html` | the page: styles + pipeline panels + replay engine (vanilla JS), with `__VIZ_DATA__` placeholder |
| `export.py` | one run → one standalone HTML (`runs/X/replay.html`) |
| `export_all.py` | every run under `runs/` → a portable `replays/` directory (`<run>.html` per run + `index.html`); auto-runs `analyze.py` when a run has observer CSVs but no `trace.json` yet, skips runs recorded without `--observer on`. Also exports **solo pages** (`<run>.solo.html`) from `<runs>/<name>/seq/` when the run has exactly one client + observer data |
| `server.py` / `serve.sh` | stdlib web server for `replays/` (+ `GET /api/runs`); idempotent launcher, logs to `/tmp/colocator_replay_server.log` |
| `mac/colocator-replay.sh` | Mac-side one-shot: start remote server via ssh, open tunnel, open browser |

## Usage

```bash
# all runs at once (the usual path) -> replays/<run>.html + replays/index.html
python colocator/visualizer/export_all.py            # scans runs/, writes replays/

# or a single run, with a custom label
python colocator/visualizer/export.py runs/prio --label "prio -5,0"
# -> runs/prio/replay.html   (~160 KiB, self-contained)
```

Runs need `run_demo.py --mode colocated --observer on` (the observer provides
the GPU-side timestamps the animation replays).

### Solo (observer-only) replays

A **single-client seq run** with observer data replays too — used for
workloads that can't be colocated yet (e.g. `llmdecode`, whose cuBLAS
launches bypass the interceptor):

```bash
python colocator/demo/run_demo.py --mode seq --clients llmdecode --observer on --out runs/llmdecode_solo
python colocator/visualizer/export_all.py   # -> replays/llmdecode_solo.solo.html
```

These pages carry `mode:"seq"` in their dataset and hide everything that has
no meaning without the colocator — the compare bar, the
Interceptor/Scheduler/Streams pipeline row, both provenance panels and the
concurrency panel — leaving the description box, controls, the GPU-timeline
Gantt (setup region dimmed, ⇥ jumps to the timed iterations) and the
per-iteration wall-time chart. Chip tooltips show `launched (host)` (CUPTI
RUNTIME record, when one exists) and the GPU start/end instead of the
four-stage colocator timestamps.

### Serving the replays (and viewing from a Mac)

On the GPU box — idempotent launcher (stdlib-only `server.py` serving
`replays/`; `GET /api/runs` lists exports; re-exports are picked up on
refresh, no restart):

```bash
bash colocator/visualizer/serve.sh        # -> http://localhost:8000
```

From a Mac: copy [`mac/colocator-replay.sh`](mac/colocator-replay.sh) to the
Mac (`~/bin/`), edit its `REMOTE=` line to your ssh target, `chmod +x`, then
run `colocator-replay.sh` — it starts the server remotely, opens the SSH
tunnel, and launches the browser at the replay index. Stop the server with
`pkill -f visualizer/server.py`.

Alternatives: `scp` any single replay file (they're self-contained), or
publish one as a Claude artifact web page (`export.py --fragment` writes the
wrapper-less variant used for that).

## Pair navigation & workload descriptions

At the top of every page: **two workload selectors** ("Compare: A vs B",
where B offers "(alone)" for single-client runs) plus a **throttle selector**
(none / cap N) and a **stream-limit selector** (off / N% — one entry per
exported `smlimit_<pct>` config; selecting a percentage navigates to that
run, where cuBLAS ops ran on a green-context stream capped to N% of the
SMs). On stream-limit pages the "Streams · issued, awaiting GPU" panel shows
**two labeled lanes per client** — `main (100% SMs)` and
`gemm (green ctx, N% SMs)` — with chips split by the actual GPU stream each
op ran on. Picking any combination navigates
to that run's replay file (`<pair>.<throttle_cfg>.html`; the navigation map
over all exported runs is embedded by `export_all.py`; missing combinations
show the exact `run_pairs.py` command to produce them). Below the selectors, **two description boxes** (left =
client 0, right = client 1) render each workload's `describe()` HTML — what it
does, kernel launch geometry/sizes, and the calibrated iteration count —
recorded into `results.json` at run time by `run_demo.py`.

## What the animation shows (and where each fact comes from)

Pipeline, left to right — an op (chip) moves through five stations:

```
Apps ──► Interceptor·queues ──► Scheduler·FCFS ──► Streams (issued) ──► Running on GPU ──► Done
         [INTERCEPTOR]          [SCHEDULER]        [SCHEDULER→OBSERVER]  [OBSERVER]
```

- **Apps** — one box per client (latency / throughput); pulses while that
  client has an op in flight; counts ops submitted.
- **Interceptor · queues** — the per-client software queue: an op sits here
  from `t_intercept` to `t_issue` (lockstep ⇒ ≤ 1 chip per client).
- **Scheduler · FCFS** — the current round number and a ticker of the last
  issue decisions: `#round → client N class → stream N`.
- **Streams** — issued to the stream's hardware queue but not yet executing
  (`t_issue` → `t_gpu_start`); this is where cross-client blocking is visible
  as pile-ups. Stream priority is labeled.
- **Running on GPU** — `t_gpu_start` → `t_gpu_end`, with an in-chip progress
  bar; the green **CONCURRENT** badge lights when both clients have kernels
  resident simultaneously.
- **Done** — per-client completed count + last finished op.
- **GPU timeline** — the familiar Gantt (kernels/copies × clients) as a
  minimap; white playhead; click to seek; the scrubber sits directly above at
  the same width; **green bands** mark concurrency periods.
- **Concurrency periods** — every interval where both clients' kernels were
  resident on the GPU simultaneously (intersection of the two merged
  kernel-busy interval sets, observer GPU timestamps — same math as
  analyze.py), listed as clickable `start – end ms` chips that jump the
  playhead there; the summary line gives the total concurrent time and what
  fraction of each client's kernel-busy time it covers. The chip containing
  the playhead highlights during playback.
- **Overhead panel** — two per-client charts of per-iteration wall time:
  the client **running alone** (grey, from the run's own `seq/` baseline, or
  `runs/full/seq` as fallback — `--baseline` overrides) vs **colocated**
  (client color), with p50s and the overhead ratio in the header. Log y-axis
  kicks in automatically when the spread demands it (e.g. 0.46 ms solo vs
  60 ms colocated tails).
- **Provenance panel** — the ground truth of the whole tool: `t_intercept`
  (+op identity) is known only to the **interceptor**, `t_issue` (+stream,
  round order) only to the **scheduler**'s issue log, `t_gpu_start/end`
  (+real duration) only to the **CUPTI observer**. Chip tooltips repeat the
  four timestamps with per-source badges and the derived queue/pend/run
  delays.

Chip color = kernel class (same palette as `analysis/plot_timeline.py`:
sgemm/gemm red, elementwise blue, reduce green, copies orange, attention
teal); chip outline color = client. Playback speed is "trace-time per wall-second"
(0.1 ms/s … real time); "skip idle gaps" jumps over stretches with no event
(setup, between phases).

Not shown: `cudaMalloc` / `cudaStreamSynchronize` records (no GPU activity to
draw) — they are in `issue_log.csv` if needed.

## Verification

`export.py` output is checked by executing the template's JS under node with
DOM stubs (`render()` across the timeline) and by validating the embedded
dataset (placeholder substitution, op counts, per-op timestamp ordering,
round monotonicity). Max observer clock skew in the demo data: 8 µs
(handled; chips just skip the sub-µs "pending" state at that scale).
