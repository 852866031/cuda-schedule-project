# demo/ — workloads, drivers, smoke tests

## Files

| File | Role |
|---|---|
| `workload_defs/` | **one workload per file** (9 registered: 2 application clients + 6 resource probes + `llmdecode`, a real Llama-3-8B decode client — colocatable via cuBLAS **API-level** interception) — see [workload_defs/README.md](workload_defs/README.md) |
| `workloads.py` | registry: auto-loads every `WORKLOAD` class from `workload_defs/`; standalone CLI: `--list`, `--client <name> --check` |
| `run_demo.py` | one run: `seq` / `streams` / `colocated` modes, writes `results.json` (+ `issue_log.csv` when colocated). Iterations auto-calibrate to **~500 ms per client** unless `--iters` forces them (single int or comma list). Records per-client `describe()` HTML and the barrier timestamp for the replay. |
| `run_pairs.py` | **run-all driver**: one solo `seq` pass over every workload (baseline + deterministic iteration counts), then one fresh colocated process per unordered pair (incl. self-pairs) → `runs/<throttle_cfg>/<a>__<b>/`, where the config dir is `throttle_none` (fcfs default) or `throttle_<N>` (`--policy throttle --throttle N`). Run once per throttle setting for side-by-side matrices (`--skip-solo` reuses the baseline). |
| `smoke_test.py` | Phase 1 interposer proof-of-life (ATen ops, no extension) |

## The workload cycle

Every workload implements the same contract (see `workload_defs/base.py`):
`setup → calibrate(~500 ms) → check → warmup ×3 → drain → barrier → timed
iterations → drain`. Iterations are **pipelined depth-2**: `run_iter()` admits
the next iteration as soon as the one from two admissions ago completes
(torch.cuda.Event, redirected by the colocator), so each client always has
two iterations in flight — no drain gap between kernels, solo or colocated.
In colocated mode each client runs in its own `torch.cuda.Stream` context
(its own allocator pool — a correctness requirement, see the header comment
in `run_demo.py`).

## Typical usage

```bash
# everything: solo baseline + all 28 pairs + replays
python colocator/demo/run_pairs.py
python colocator/visualizer/export_all.py

# one pair, custom colocator config
python colocator/demo/run_demo.py --mode colocated --clients l2,dram \
    --priorities=-5,0 --policy throttle --observer on --out runs/l2__dram

# one workload solo, verified
python colocator/demo/workloads.py --client fma --check
```

## Why retrieval uses pinned buffers + explicit stream sync (not `.cpu()`)

`tensor.cpu()` copies into freshly-allocated **pageable** memory, and
`cudaMemcpyAsync` into pageable memory silently blocks the calling thread
until the copy finishes. Under the colocator the calling thread is the
**scheduler**, so one client's slow D2H would stall *both* clients' queues
(measured as a 20 ms queue-delay tail before the fix). Application workloads
copy into preallocated pinned buffers with `non_blocking=True`; the following
`torch.cuda.current_stream().synchronize()` is intercepted and redirected to
the client's colocator stream.
