#!/usr/bin/env python
"""Tiered, preemptable LoRA fine-tune tenant (design §3.4).

Trains LoRA (r=16, q/k/v/o) on Qwen2.5-0.5B with deterministic synthetic batches; before each
optimizer step it reads the tenant budget published by the VRAM scheduler and picks the highest
tier whose calibrated peak fits:

  tier 0  full batch                               tier 1  micro-batches k=2/4/8 (grad accumulation)
  tier 2  + activation offload to pinned host RAM       tier 3  + frozen base weights of n layers streamed from host RAM
  tier 4  pause: snapshot, free the GPU, wait

The torch allocator cap follows the budget, so a mis-estimate OOMs this process, never the
engine; an OOM drops one tier. A "collision" rule forces k >= 4 when the engine's busy duty
cycle over the last second exceeds --collision-duty. Snapshots (LoRA weights, optimizer,
scheduler, step, RNG) go to pinned host RAM / /dev/shm every --snapshot-every steps, before every
tier change and on SIGTERM; the worker resumes from the snapshot on start. All host<->GPU
copies it makes are recorded through xfer_trace (src=ft).

    CUDA_VISIBLE_DEVICES=0 python ft_worker.py --out results/x_ft.csv --gate
    CUDA_VISIBLE_DEVICES=1 python ft_worker.py --calibrate --tiers-json tiers.json   # measure tier peaks
"""
import argparse, csv, json, math, mmap, os, signal, struct, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer_offload import OffloadManager
from xfer_trace import XferTrace, default_path
from green_ctx import GreenCtx
from pacing import Bpt, BlockPacer, BytePacer, SegTable, cap_tok_s
from loss_chunk import chunked_lm_loss
from head_loss import split_lm_loss, vocab_lm_loss

GiB = 1024 ** 3
TOTAL_GIB = 31.3536
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--batch", type=int, default=8); ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--lora-r", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--total-steps", type=int, default=100000, help="cosine schedule horizon")
ap.add_argument("--duration", type=float, default=0.0); ap.add_argument("--steps", type=int, default=0)
ap.add_argument("--out", default=None)
ap.add_argument("--budget-file", default="/dev/shm/coloc_ft_budget")
ap.add_argument("--state-file", default="/dev/shm/coloc_ft_state")
ap.add_argument("--snapshot", default="/dev/shm/coloc_ft_snapshot.pt")
ap.add_argument("--snapshot-every", type=int, default=10)
ap.add_argument("--fresh", action="store_true", help="ignore an existing snapshot")
ap.add_argument("--tiers-json", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiers.json"))
ap.add_argument("--fixed-budget-gib", type=float, default=None, help="ignore the budget file (static-split scheme / solo)")
ap.add_argument("--fixed-tier", type=int, default=None, help="pin a tier config index (see TIERS)")
ap.add_argument("--gate", action="store_true"); ap.add_argument("--collision-duty", type=float, default=0.6)
ap.add_argument("--calibrate", action="store_true"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--margin-gib", type=float, default=0.3, help="headroom kept between calibrated peak and budget")
ap.add_argument("--up-hold", type=float, default=3.0, help="seconds a larger budget must persist before climbing a tier")
ap.add_argument("--green-sms", type=int, default=0, help="run all tenant kernels on a CUDA green context of this many SMs (0 = full device)")
ap.add_argument("--green-remaining-of", type=int, default=0, help="run on the SMs left after an N-SM split (disjoint from an engine on the first N)")
ap.add_argument("--tok-s-cap", type=float, default=0.0, help="fixed tokens/s cap (probe runs); 0 = follow the budget file's tok_s_cap if present")
ap.add_argument("--green-prebuilt", action="store_true", help="P2: create every green context of --green-set once at start and only switch streams between steps (no create/destroy per switch)")
ap.add_argument("--green-set", default="128,96,64,32,16", help="SM counts to prebuild with --green-prebuilt")
ap.add_argument("--green-dynamic", action="store_true", help="take the SM count from the budget file's 'sms' field (VRAM-scheduler-driven); --green-sms is the fallback")
# ---- overnight plan 3: C1 (byte allowance, two pacing methods) and C2 (variants that move fewer bytes per token)
ap.add_argument("--gbps-cap", type=float, default=0.0, help="fixed GB/s allowance; 0 = follow the budget file's gbps if present")
ap.add_argument("--bpt-json", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_o3", "bytes_per_token.json"), help="MB per token by variant and shape (tenant_bytes.py)")
ap.add_argument("--bpt-variant", default="base", help="row of --bpt-json used to turn GB/s into tokens/s ('base' also for a variant that must do the usual tenant's tokens/s)")
ap.add_argument("--pace", default="step", choices=["step", "block", "bytes"], help="keep the cap by pausing after a step, in slices at block boundaries (equal work assumed per checkpoint), or by the byte ledger: chunks of equal bytes from the per-segment table; the budget file's 'pace' overrides")
ap.add_argument("--pace-window-ms", type=float, default=12.0, help="with --pace bytes: a chunk is GB/s x this window; 0 = one pause after the step (BWS-Budget). The budget file's 'window' overrides")
ap.add_argument("--seg-table", default=None, help="per-segment bytes of a step (segment_bytes.py); default: results_prof/seg_<variant of these flags>.csv")
ap.add_argument("--seg-table-cfg", type=int, default=2, help="the shape (TIERS index) the table was measured for; other shapes fall back to the pause after the step")
ap.add_argument("--slice-blocks", type=int, default=8, help="blocks per slice with --pace block; the budget file's 'slice' overrides")
ap.add_argument("--lora-top-layers", type=int, default=0, help="LoRA on the top N layers only (0 = all layers); changes the training")
ap.add_argument("--attn", default=None, help="attention implementation (sdpa, eager); default: the model's")
ap.add_argument("--compile", action="store_true", help="torch.compile the model (no block hooks: no idle gate, no block pacing, no mid-step preemption)")
ap.add_argument("--compile-blocks", action="store_true", help="torch.compile each of the model's blocks on its own: the block hooks stay, so the tenant can still pause after every block (the shapes with layers in host RAM are not supported: use --no-slow-shapes)")
ap.add_argument("--loss-chunk", type=int, default=0, help="output layer and loss over this many positions at a time (0 = the usual way)")
ap.add_argument("--head", default="model", choices=["model", "split", "vocab"], help="output layer and loss: the model's own; 'split' = the same operators with a checkpoint between them; 'vocab' = over blocks of the vocabulary, the full logits never exist (head_loss.py)")
ap.add_argument("--vocab-block", type=int, default=8192, help="vocabulary rows per block with --head vocab")
ap.add_argument("--head-fused", action="store_true", help="with --head vocab: the operators of a vocabulary block as compiled kernels (head_loss._fused_block_fns)")
ap.add_argument("--nvtx", action="store_true", help="profiling: synchronise and emit an NVTX mark at every checkpoint and at step begin/end (segment_bytes.py)")
ap.add_argument("--nvtx-steps", action="store_true", help="profiling: NVTX marks at step begin and end only, no extra synchronisation inside the step (pace_check.py)")
ap.add_argument("--forward-only", action="store_true", help="reference: forward pass only, nothing is trained")
ap.add_argument("--force-ckpt", action="store_true", help="gradient checkpointing in every shape")
ap.add_argument("--no-slow-shapes", action="store_true", help="shapes with offload count as 'does not fit': pause instead")
args = ap.parse_args()

# tier configurations, most to least GPU-hungry; (tier, k micro-batches, activation offload, n layers weight-offloaded)
# ckpt = gradient checkpointing: off for tiers 0-1 (fast, memory-hungry), on for tiers 2-3 (with activation
# offload of the checkpointed layer inputs and recompute)
TIERS = [dict(tier=0, k=1, act=False, off=0, ckpt=False), dict(tier=1, k=2, act=False, off=0, ckpt=False),
         dict(tier=1, k=4, act=False, off=0, ckpt=False), dict(tier=1, k=8, act=False, off=0, ckpt=False),
         dict(tier=2, k=8, act=True, off=0, ckpt=True),
         dict(tier=3, k=8, act=True, off=12, ckpt=True), dict(tier=3, k=8, act=True, off=20, ckpt=True),
         dict(tier=4, k=0, act=False, off=0, ckpt=False)]
PAUSE = len(TIERS) - 1

torch.manual_seed(args.seed)
dev = torch.device("cuda:0")
trace = XferTrace(default_path("ft"), "ft")
stop = {"flag": False}
signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True)); signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, **({"attn_implementation": args.attn} if args.attn else {}))
_lora = dict(r=args.lora_r, lora_alpha=2 * args.lora_r, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
if args.lora_top_layers: _lora["layers_to_transform"] = list(range(model.config.num_hidden_layers - args.lora_top_layers, model.config.num_hidden_layers))
model = get_peft_model(model, LoraConfig(**_lora))
if not args.lora_top_layers: model.enable_input_require_grads()   # with LoRA on the top layers only, the backward pass stops at the lowest adapted layer
if args.compile_blocks:                    # R2 of the plan of Nov 21: one compiled function per block; checkpoints between blocks survive
    _ly = model.base_model.model.model.layers
    # every block and every micro-batch size is its own compilation of the same code object; with the default limit (8) only
    # the first 8 blocks are compiled and the rest run uncompiled without any error
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 16 * len(_ly))
    torch._dynamo.config.accumulated_recompile_limit = max(torch._dynamo.config.accumulated_recompile_limit, 128 * len(_ly))
    for _j in range(len(_ly)): _ly[_j] = torch.compile(_ly[_j])
net = torch.compile(model) if args.compile else model
body = torch.compile(model.base_model.model.model) if (args.compile and (args.loss_chunk or args.head != "model")) else None
bpt = Bpt(args.bpt_json, args.bpt_variant)
model.train()
ckpt_state = {"on": None}
def set_ckpt(on):
    if ckpt_state["on"] == on: return
    if on: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else: model.gradient_checkpointing_disable()
    ckpt_state["on"] = on
layers = model.base_model.model.model.layers
opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.total_steps)
vocab = model.config.vocab_size
gen = torch.Generator().manual_seed(args.seed)
param_ptrs = set()

# ---------------------------------------------------------------- gate (idle-window) + busy duty
page = None
if args.gate:
    fd = os.open("/dev/shm/coloc_hp_busy", os.O_CREAT | os.O_RDWR, 0o666); os.ftruncate(fd, 7 * 8)
    page = mmap.mmap(fd, 7 * 8); os.close(fd)
MAGIC, STALE_NS = 0x434F4C4F43, 1_500_000_000
gate_ns = {"v": 0}; fence = {"ev": None}
class Preempt(Exception):
    """Raised inside a step (block hook or gate wait) when the budget no longer fits the current config."""
preempt_state = {"cur": None, "last_poll": 0.0, "count": 0}
def poll_budget_midstep():
    """Called from the per-block hooks: abort the step as soon as the published budget requires a lower tier."""
    now = time.time()
    if now - preempt_state["last_poll"] < 0.05 or preempt_state["cur"] is None or args.fixed_tier is not None: return
    preempt_state["last_poll"] = now
    budget, _ = read_budget()
    cfg_i = preempt_state["cur"]
    if peaks.get(cfg_i, 0) + args.margin_gib * GiB > budget:
        raise Preempt(f"budget {budget/GiB:.2f} GiB < need of cfg {cfg_i}")
def rd(i): return struct.unpack_from("<q", page, i * 8)[0]
mark = {"step": 0, "label": ""}; NV = torch.cuda.nvtx if args.nvtx else None; NVS = torch.cuda.nvtx if (args.nvtx or args.nvtx_steps) else None
def nv_mark(name):
    """profiling only: the mark sits where the GPU work issued so far has finished"""
    sync_tenant(); torch.cuda.nvtx.mark(name)
def ckpt(label=""):
    """a named checkpoint: the label names the work that follows it (block hooks, head_loss.py)"""
    mark["label"] = label; gate()
def gate(*_):
    if NV is not None: nv_mark(f"s{mark['step']}u{pacer.units + 1}:{mark['label']}")
    mark["label"] = ""
    pacer.tick(); bpacer.tick()
    poll_budget_midstep()
    if page is None or rd(0) != MAGIC: return
    t0 = time.monotonic_ns()
    if fence["ev"] is not None: fence["ev"].synchronize()
    while rd(2) and time.monotonic_ns() - rd(1) < STALE_NS:
        time.sleep(0.0002)
        poll_budget_midstep()          # a gated tenant must still release memory when asked
    ev = torch.cuda.Event(); ev.record(); fence["ev"] = ev
    gate_ns["v"] += time.monotonic_ns() - t0
if not args.compile:
    for j, m in enumerate(layers):
        m.register_forward_pre_hook(lambda *a, j=j: ckpt(f"blk{j}:fwd")); m.register_full_backward_pre_hook(lambda *a, j=j: ckpt(f"blk{j}:bwd"))
duty_state = {"t": time.monotonic_ns(), "busy": 0}
def busy_duty():
    """fraction of the last interval the engine reported busy (from the page's busy_ns_total)."""
    if page is None or rd(0) != MAGIC: return 0.0
    t = time.monotonic_ns(); b = rd(5)
    dt, db = t - duty_state["t"], b - duty_state["busy"]
    duty_state.update(t=t, busy=b)
    return min(1.0, db / dt) if dt > 0 else 0.0

# ---------------------------------------------------------------- activation offload (saved_tensors_hooks)
copy_stream = torch.cuda.Stream()
act_stats = {"out_b": 0, "in_b": 0, "out_ms": 0.0, "in_ms": 0.0}
MIN_OFFLOAD = 1 << 20
pin_pool = {}; pin_returning = []
def pin_get(t):
    key = (tuple(t.shape), t.dtype); lst = pin_pool.get(key)
    if lst: return lst.pop()
    return torch.empty_like(t, device="cpu", pin_memory=True)
def pin_recycle():
    for key, buf in pin_returning: pin_pool.setdefault(key, []).append(buf)
    pin_returning.clear()
def pack(t):
    if (not t.is_cuda) or t.numel() * t.element_size() < MIN_OFFLOAD or t.data_ptr() in param_ptrs or not t.is_contiguous():
        return ("gpu", t)
    cpu = pin_get(t)
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    copy_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(copy_stream):
        e0.record(); cpu.copy_(t, non_blocking=True); e1.record()
    t.record_stream(copy_stream)
    act_stats["out_b"] += cpu.numel() * cpu.element_size()
    return ("cpu", cpu, e0, e1, t.device)
def unpack(x):
    if x[0] == "gpu": return x[1]
    _, cpu, e0, e1, device = x
    e1.synchronize()
    try: act_stats["out_ms"] += e0.elapsed_time(e1)
    except Exception: pass
    s0 = torch.cuda.Event(enable_timing=True); s1 = torch.cuda.Event(enable_timing=True)
    s0.record(); g = cpu.to(device, non_blocking=True); s1.record()
    act_stats["in_b"] += cpu.numel() * cpu.element_size(); pin_returning.append(((tuple(cpu.shape), cpu.dtype), cpu))
    return g
class act_offload:
    def __init__(self, on): self.on = on; self.ctx = None
    def __enter__(self):
        if self.on:
            self.ctx = torch.autograd.graph.saved_tensors_hooks(pack, unpack); self.ctx.__enter__()
    def __exit__(self, *a):
        if self.ctx: self.ctx.__exit__(*a)

# ---------------------------------------------------------------- weight offload + residency
offload = {"mgr": None, "n": 0}
def set_weight_offload(n):
    if offload["n"] == n: return
    if offload["mgr"] is not None:
        offload["mgr"].remove()
        for s in offload["mgr"].streamers: s.bind()   # bring everything back resident
        offload["mgr"] = None
    offload["n"] = n
    if n > 0:
        set_ckpt(True)
        offload["mgr"] = OffloadManager(layers, n, device="cuda", prefetch=True, depth=1, pattern="tail")
    param_ptrs.clear(); param_ptrs.update(p.data_ptr() for p in model.parameters())
    torch.cuda.empty_cache()

resident = {"gpu": False}
def to_gpu():
    if resident["gpu"]: return
    t0 = time.time(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record(); model.to(dev)
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v): st[k] = v.to(dev)
    e1.record(); torch.cuda.synchronize()
    nbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    trace.record("resume", "h2d", nbytes, t0, time.time(), e0.elapsed_time(e1))
    param_ptrs.clear(); param_ptrs.update(p.data_ptr() for p in model.parameters()); resident["gpu"] = True
def to_cpu():
    if not resident["gpu"]: return
    # Pausing must not allocate on the GPU (the budget may already be ~0): streamed layers go
    # straight to their pinned host copies instead of being bound back first.
    if offload["mgr"] is not None:
        offload["mgr"].remove()
        for st in offload["mgr"].streamers:
            params = st._params()
            for name, host in st.cpu.items(): params[name].data = host
            st.staged.clear(); st.resident = False
        offload["mgr"] = None; offload["n"] = 0
    t0 = time.time(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record(); nbytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.is_cuda); model.cpu()
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v): st[k] = v.cpu()
    e1.record(); torch.cuda.synchronize(); torch.cuda.empty_cache()
    trace.record("pause", "d2h", nbytes, t0, time.time(), e0.elapsed_time(e1)); resident["gpu"] = False

# ---------------------------------------------------------------- snapshots
def snapshot(step):
    t0 = time.time()
    state = dict(lora={k: v.detach().to("cpu", copy=True) for k, v in get_peft_model_state_dict(model).items()},
                 opt=opt.state_dict(), sched=sched.state_dict(), step=step, rng=torch.get_rng_state(), gen=gen.get_state())
    nbytes = sum(v.numel() * v.element_size() for v in state["lora"].values())
    torch.save(state, args.snapshot + ".tmp"); os.replace(args.snapshot + ".tmp", args.snapshot)
    trace.record("snapshot", "d2h", nbytes, t0, time.time(), None, f"step={step}")
    return time.time() - t0
def restore():
    if args.fresh or not os.path.exists(args.snapshot): return 0
    st = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    set_peft_model_state_dict(model, st["lora"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    torch.set_rng_state(st["rng"]); gen.set_state(st["gen"])
    print(f"ft_worker: resumed from snapshot at step {st['step']}", flush=True); return st["step"]

# ---------------------------------------------------------------- green context (SM partition)
green = {"ctx": None, "sms": 0, "req": 0, "t": 0.0, "pool": {}}
def prebuild_green():
    """P2: one context per SM count, built before any training allocation; switching later only changes the stream."""
    for n in sorted({int(x) for x in args.green_set.split(",") if x}, reverse=True):
        try:
            g = GreenCtx(n, device=0); green["pool"][n] = g
            print(f"ft_worker: prebuilt green context {n} -> {g.sm_count} SMs", flush=True)
        except Exception as e:
            print(f"ft_worker: green context {n} failed: {e}", flush=True)
def set_green(sms):
    """Switch the tenant's compute stream to a green context of `sms` SMs (0 = full device).
    Compares against the *requested* count (the driver rounds), switches at most every 5 s, and
    returns the caching allocator's blocks of the old stream (they cannot serve the new one)."""
    if sms == green["req"] or time.time() - green["t"] < (1.0 if args.green_prebuilt else 5.0): return
    torch.cuda.synchronize()
    old = green["ctx"]
    if args.green_prebuilt:
        if sms > 0:
            key = min(green["pool"], key=lambda k: abs(k - sms)) if green["pool"] else None
            if key is None: return
            g = green["pool"][key]; green["ctx"] = g; green["sms"] = g.sm_count; torch.cuda.set_stream(g.torch_stream())
        else:
            green["ctx"] = None; green["sms"] = 0; torch.cuda.set_stream(torch.cuda.default_stream())
        green["req"] = sms; green["t"] = time.time(); fence["ev"] = None; torch.cuda.empty_cache()
        trace.record("green_switch", "none", 0, time.time(), time.time(), None, f"sms={green['sms']} reserved={torch.cuda.memory_reserved(dev)/GiB:.2f}")
        return
    if sms > 0:
        g = GreenCtx(sms, device=0, remaining=(args.green_remaining_of > 0)); green["ctx"] = g; green["sms"] = g.sm_count
        torch.cuda.set_stream(g.torch_stream())
    else:
        green["ctx"] = None; green["sms"] = 0; torch.cuda.set_stream(torch.cuda.default_stream())
    green["req"] = sms; green["t"] = time.time()
    if old is not None: old.close()
    fence["ev"] = None; torch.cuda.empty_cache()
    trace.record("green_switch", "none", 0, time.time(), time.time(), None, f"sms={green['sms']}")

# ---------------------------------------------------------------- budget / tiers
peaks = {}
if os.path.exists(args.tiers_json):
    peaks = {int(k): float(v) for k, v in json.load(open(args.tiers_json)).items()}
budget_extra = {"sms": None, "cap": None, "gbps": None, "pace": None, "slice": None, "window": None}
compiled_ok = set()   # with --compile: the shapes compiled at start
def sync_tenant():
    """wait for the tenant's queued kernels (the hook may run in autograd's device thread, so name the stream)"""
    (green["ctx"].torch_stream() if green["ctx"] is not None else torch.cuda.default_stream()).synchronize()
pacer = BlockPacer(sync_tenant, poll=poll_budget_midstep)
bpacer = BytePacer(sync_tenant, poll=poll_budget_midstep, stop=lambda: stop["flag"])
def seg_table_path(cfg_i=None):
    """the table of shape cfg_i: seg_<variant>.csv for the usual shape (--seg-table-cfg), seg_<variant>_t<i>.csv for another"""
    if args.seg_table and cfg_i in (None, args.seg_table_cfg): return args.seg_table
    v = ("compile_" if args.compile else "cb_" if args.compile_blocks else "") + (f"vocab{args.vocab_block}{'f' if args.head_fused else ''}" if args.head == "vocab" else "split" if args.head == "split" else "")
    name = (v.rstrip('_') or 'base') + ("" if cfg_i in (None, args.seg_table_cfg) else f"_t{cfg_i}")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_prof", f"seg_{name}.csv")
segtab = {}          # shape index -> SegTable, or None where no table was measured (that shape pauses after the step)
def seg_table_for(cfg_i):
    if cfg_i not in segtab:
        try:
            segtab[cfg_i] = SegTable(seg_table_path(cfg_i))
            print(f"ft_worker: per-segment table for shape {cfg_i}: {segtab[cfg_i].path}: {segtab[cfg_i].n} checkpoints, {segtab[cfg_i].total:.2f} GB per step, largest piece {segtab[cfg_i].largest:.2f} GB", flush=True)
        except Exception as e:
            segtab[cfg_i] = None
            if args.pace == "bytes": print(f"ft_worker: no per-segment table for shape {cfg_i} ({e}); it pauses after the step", flush=True)
    return segtab[cfg_i]
seg_table_for(args.seg_table_cfg)
def read_budget():
    if args.fixed_budget_gib is not None:
        if args.tok_s_cap: budget_extra["cap"] = args.tok_s_cap
        return args.fixed_budget_gib * GiB, 0
    try:
        j = json.load(open(args.budget_file)); budget_extra["sms"] = j.get("sms"); budget_extra["cap"] = args.tok_s_cap or j.get("tok_s_cap")
        budget_extra["gbps"] = j.get("gbps"); budget_extra["pace"] = j.get("pace"); budget_extra["slice"] = j.get("slice"); budget_extra["window"] = j.get("window")
        return float(j["bytes"]), int(j.get("seq", 0))
    except Exception:
        return TOTAL_GIB * GiB, 0
def choose(budget, duty):
    if args.fixed_tier is not None: return args.fixed_tier
    for i, cfg in enumerate(TIERS[:-1]):
        if cfg["k"] and duty > args.collision_duty and cfg["k"] < 4: continue
        if args.no_slow_shapes and (cfg["act"] or cfg["off"]): continue
        if (args.compile or args.compile_blocks or args.head_fused) and compiled_ok and i not in compiled_ok: continue   # a shape first met in mid-run would compile for ~45 s inside one step
        if peaks.get(i, 1e9) + args.margin_gib * GiB <= budget: return i
    return PAUSE
def apply_cap(budget):
    torch.cuda.set_per_process_memory_fraction(max(0.01, min(1.0, budget / (TOTAL_GIB * GiB))), dev)
def publish_state(seq, i, extra=""):
    try:
        tmp = args.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(dict(seq_ack=seq, tier=TIERS[i]["tier"], cfg=i, k=TIERS[i]["k"], off=TIERS[i]["off"],
                           footprint_gib=round(torch.cuda.memory_reserved(dev) / GiB, 3), ts=time.time(), extra=extra), f)
        os.replace(tmp, args.state_file)
    except Exception: pass

# ---------------------------------------------------------------- one optimizer step under a config
def train_step(cfg, ids):
    k = cfg["k"]; tokens = ids.numel(); loss_sum = 0.0
    act_stats.update(out_b=0, in_b=0, out_ms=0.0, in_ms=0.0)
    ckpt("step")
    for i in range(k):
        sub = ids[i::k]
        if args.forward_only:
            with torch.no_grad(): loss_sum += net(input_ids=sub, labels=sub).loss.item()
            continue
        with act_offload(cfg["act"]):
            if args.loss_chunk:
                sur, lval = chunked_lm_loss(model, sub, args.loss_chunk, body); loss = sur / k
            elif args.head == "split":
                loss = split_lm_loss(model, sub, ckpt, body) / k; lval = None
            elif args.head == "vocab":
                sur, lval = vocab_lm_loss(model, sub, args.vocab_block, ckpt, body, args.head_fused); loss = sur / k
            else:
                out = net(input_ids=sub, labels=sub); loss = out.loss / k; lval = None
        if offload["mgr"] is not None:
            offload["mgr"].pre_forward() if False else None
            offload["mgr"].backward(loss)
        else:
            loss.backward()
        loss_sum += float(lval) if lval is not None else loss.item() * k
        if k > 1: ckpt("mb_end")
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); pin_recycle()
    return loss_sum / k, tokens

def main():
    step = restore()
    out = open(args.out, "w", newline="") if args.out else None
    w = csv.writer(out) if out else None
    if w: w.writerow(["ts", "step", "cfg", "tier", "k", "off", "act", "dt_ms", "tok_s", "loss", "peak_gib", "reserved_gib",
                      "gated_ms", "budget_gib", "duty", "act_out_mb", "act_in_mb", "snapshot_ms", "event", "sms", "cap_tok_s", "sleep_ms", "pace", "slices", "slice_sleep_ms", "gbps", "units", "window_ms", "late", "late_ms", "forgiven_ms"])
    to_gpu(); cur = None; seq_seen = -1; t_start = time.time(); last_snap = step; up_since = {"cfg": None, "t": 0.0}
    last_budget = {"v": TOTAL_GIB * GiB}
    if args.green_prebuilt:
        # materialise AdamW state on the default stream (zero grads, no weight decay for this one step) so no later
        # stream's allocator pool has to grow around long-lived state, then build the contexts
        wd = [g.get("weight_decay", 0.0) for g in opt.param_groups]
        for g in opt.param_groups: g["weight_decay"] = 0.0
        for p in model.parameters():
            if p.requires_grad: p.grad = torch.zeros_like(p)
        opt.step(); opt.zero_grad(set_to_none=True)
        for g, v in zip(opt.param_groups, wd): g["weight_decay"] = v
        torch.cuda.synchronize(); prebuild_green()
    if args.green_remaining_of: set_green(args.green_remaining_of)
    elif args.green_sms and not args.green_dynamic: set_green(args.green_sms)
    def warm_loss(sub, k):
        """the loss of one micro-batch by the path the run will use (so that this path is what gets compiled)"""
        if args.head == "vocab": return vocab_lm_loss(model, sub, args.vocab_block, lambda *_: None, body, args.head_fused)[0] / k
        if args.head == "split": return split_lm_loss(model, sub, lambda *_: None, body) / k
        return net(input_ids=sub, labels=sub).loss / k
    if (args.compile or args.compile_blocks or args.head_fused) and not args.calibrate:
        # compile every shape that fits the budget now, before the first real step: a recompilation in mid-run would hold
        # the tenant inside one "step" for tens of seconds, with no hooks to release memory through
        b0, _ = read_budget()
        for i_, c_ in enumerate(TIERS[:-1]):
            if c_["act"] or c_["off"] or peaks.get(i_, 1e18) + args.margin_gib * GiB > b0: continue
            try:
                apply_cap(b0); set_ckpt(c_["ckpt"] or args.force_ckpt); tw = time.time()
                ids = torch.randint(0, vocab, (args.batch, args.seq), generator=torch.Generator().manual_seed(12345)).to(dev)   # its own generator: the training data of step n is then the same with and without this warm-up
                for j_ in range(2):
                    sub = ids[j_::c_["k"]]; warm_loss(sub, c_["k"]).backward()
                opt.zero_grad(set_to_none=True); torch.cuda.synchronize(); compiled_ok.add(i_); print(f"ft_worker: compiled shape cfg {i_} (k={c_['k']}) in {time.time() - tw:.0f} s", flush=True)
            except torch.OutOfMemoryError:
                opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); print(f"ft_worker: shape cfg {i_} did not fit while compiling", flush=True)
    if args.calibrate:
        results = {}
        for i, cfg in enumerate(TIERS[:-1]):
            if args.no_slow_shapes and (cfg["act"] or cfg["off"]): results[i] = int(peaks.get(i, 1e12)); continue     # shapes this tenant never uses: keep the old figure
            apply_cap(TOTAL_GIB * GiB); set_ckpt(cfg["ckpt"]); set_weight_offload(cfg["off"]); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
            dts = []
            for _ in range(3):
                ids = torch.randint(0, vocab, (args.batch, args.seq), generator=gen).to(dev)
                torch.cuda.synchronize(); t0 = time.time(); train_step(cfg, ids); torch.cuda.synchronize(); dts.append(time.time() - t0)
            alloc = torch.cuda.max_memory_allocated(dev); peak = torch.cuda.max_memory_reserved(dev)
            results[i] = int(alloc * 1.1 + 0.25 * GiB)
            print(f"cfg {i} {cfg}: peak allocated {alloc/GiB:.2f} GiB, reserved {peak/GiB:.2f} GiB -> need {results[i]/GiB:.2f}; step {min(dts)*1e3:.0f} ms, {args.batch*args.seq/min(dts):.0f} tok/s", flush=True)
        json.dump({str(k): v for k, v in results.items()}, open(args.tiers_json, "w"), indent=1)
        print("wrote", args.tiers_json); return
    while not stop["flag"]:
        if args.steps and step >= args.steps: break
        if args.duration and time.time() - t_start > args.duration: break
        budget, seq = read_budget(); duty = busy_duty() if args.gate else 0.0
        want_sms = (int(budget_extra["sms"]) if (args.green_dynamic and budget_extra["sms"]) else args.green_sms)
        before = green["req"]; set_green(want_sms)
        event_sm = f"sms->{green['sms']}" if green["req"] != before else ""
        i = choose(budget, duty); event = event_sm
        # hysteresis: drop a tier at once, climb only after the budget has allowed it for --up-hold seconds
        if cur is not None and i < cur:
            if up_since["cfg"] != i: up_since.update(cfg=i, t=time.time())
            if time.time() - up_since["t"] < args.up_hold: i = cur
        else:
            up_since.update(cfg=None)
        if i == PAUSE and cur == PAUSE:
            time.sleep(0.2); continue
        if i != cur:
            event = (event + " " if event else "") + f"tier {cur}->{i}"
            if cur is not None: snapshot(step)
            if i == PAUSE:
                opt.zero_grad(set_to_none=True); to_cpu(); torch.cuda.empty_cache(); apply_cap(max(budget, 0.2 * GiB)); publish_state(seq, i, "paused")
                cur = i; seq_seen = seq
                if w: w.writerow([f"{time.time():.3f}", step, i, 4, 0, 0, 0, "", "", "", "", f"{torch.cuda.memory_reserved(dev)/GiB:.2f}", "", f"{budget/GiB:.2f}", f"{duty:.2f}", "", "", "", event]); out.flush()
                time.sleep(0.2); continue
            try:
                apply_cap(max(budget, 1.5 * GiB))   # transient room for re-binding; the real cap follows
                to_gpu(); set_ckpt(TIERS[i]["ckpt"] or args.force_ckpt); set_weight_offload(TIERS[i]["off"]); torch.cuda.empty_cache(); apply_cap(budget); cur = i
            except torch.OutOfMemoryError:
                print(f"ft_worker: OOM while moving to cfg {i} (budget {budget/GiB:.2f}); pausing", flush=True)
                opt.zero_grad(set_to_none=True); to_cpu(); torch.cuda.empty_cache(); peaks[i] = max(peaks.get(i, 0), budget + 0.1 * GiB)
                cur = PAUSE; publish_state(seq, PAUSE, "paused-oom"); time.sleep(0.2); continue
            publish_state(seq, i, event)
        elif seq != seq_seen:
            if budget < last_budget["v"]: torch.cuda.empty_cache()   # a smaller budget: hand cached-but-free blocks back at once
            apply_cap(budget); publish_state(seq, i); seq_seen = seq
        last_budget["v"] = budget
        cfg = TIERS[i]
        ids = torch.randint(0, vocab, (args.batch, args.seq), generator=gen).to(dev)
        g0 = gate_ns["v"]; torch.cuda.reset_peak_memory_stats(dev); t0 = time.time(); preempt_state["cur"] = i
        # the cap of this step: a tokens/s cap, and/or a GB/s allowance turned into tokens/s with this shape's bytes per token
        cap = budget_extra["cap"]; gbps = args.gbps_cap or budget_extra["gbps"]
        if gbps:
            cg = cap_tok_s(float(gbps), bpt.mb(i)); cap = min(cap, cg) if cap else cg
        pace = budget_extra["pace"] or args.pace; nslice = int(budget_extra["slice"] or args.slice_blocks)
        pacer.begin_step(i, ids.numel(), cap if pace == "block" else None, nslice)
        window_ms = float(budget_extra["window"] if budget_extra["window"] is not None else args.pace_window_ms)
        use_tab = seg_table_for(i) if (pace == "bytes" and gbps) else None     # one table per shape
        bpacer.begin_step(use_tab, float(gbps) if gbps else None, window_ms / 1e3)
        if NVS is not None: mark["step"] += 1; nv_mark(f"s{mark['step']}begin")
        try:
            loss, tokens = train_step(cfg, ids)
            torch.cuda.synchronize()
            if NVS is not None: nv_mark(f"s{mark['step']}end")
        except Preempt as e:
            preempt_state["cur"] = None; preempt_state["count"] += 1; pacer.active = False; bpacer.active = False
            opt.zero_grad(set_to_none=True); torch.cuda.synchronize(); torch.cuda.empty_cache()
            event = f"preempted mid-step ({e}) after {(time.time()-t0)*1e3:.0f} ms"
            if w: w.writerow([f"{time.time():.3f}", step, i, cfg["tier"], cfg["k"], cfg["off"], int(cfg["act"]), "", "", "", "", f"{torch.cuda.memory_reserved(dev)/GiB:.2f}", f"{(gate_ns['v']-g0)/1e6:.0f}", f"{budget/GiB:.2f}", f"{duty:.2f}", "", "", "", event]); out.flush()
            up_since.update(cfg=None); continue
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            peaks[i] = max(peaks.get(i, 0), budget + 0.1 * GiB)   # this config does not fit this budget
            event = f"oom at cfg {i}, budget {budget/GiB:.2f}"; print("ft_worker:", event, flush=True)
            if w: w.writerow([f"{time.time():.3f}", step, i, cfg["tier"], cfg["k"], cfg["off"], int(cfg["act"]), "", "", "", "", f"{torch.cuda.memory_reserved(dev)/GiB:.2f}", "", f"{budget/GiB:.2f}", f"{duty:.2f}", "", "", "", event]); out.flush()
            cur = None; continue
        preempt_state["cur"] = None; pst = pacer.end_step()
        bst = bpacer.end_step()                 # byte ledger: settles the step (sleeps until its bytes are paid for)
        if bst["mismatch"]:
            print(f"ft_worker: the per-segment table of shape {i} has {segtab[i].n} checkpoints, this step had {bst['units']}; table dropped", flush=True); segtab[i] = None
        dt = time.time() - t0; step += 1
        # P1.1: throughput cap from the VRAM scheduler -- idle after the step so tokens/s (incl. the idle) stays under the cap
        sleep_ms = bst["final_sleep_s"] * 1e3
        if bst["active"]: pst = dict(pst, slices=bst["slices"], sleep_s=bst["sleep_s"])
        elif cap and tokens / dt > cap:
            pause_s = tokens / cap - dt; t_sl = time.time()
            while time.time() - t_sl < pause_s and not stop["flag"]:
                time.sleep(min(0.05, pause_s - (time.time() - t_sl)))
                poll_budget_midstep()   # keep releasing memory promptly while idling
            sleep_ms = (time.time() - t_sl) * 1e3; dt = time.time() - t0
        if act_stats["out_b"]: trace.record("act_out", "d2h", act_stats["out_b"], t0, time.time(), act_stats["out_ms"], f"step={step}")
        if act_stats["in_b"]: trace.record("act_in", "h2d", act_stats["in_b"], t0, time.time(), None, f"step={step}")
        if offload["mgr"] is not None and offload["mgr"].fetches:
            nb = offload["mgr"].fetches * (offload["mgr"].offloaded_bytes // max(1, len(offload["mgr"].streamers)))
            trace.record("weight_in", "h2d", nb, t0, time.time(), None, f"step={step}"); offload["mgr"].fetches = 0
        snap_ms = ""
        if step - last_snap >= args.snapshot_every:
            snap_ms = f"{snapshot(step)*1e3:.0f}"; last_snap = step
        if w:
            w.writerow([f"{time.time():.3f}", step, i, cfg["tier"], cfg["k"], cfg["off"], int(cfg["act"]), f"{dt*1e3:.0f}", f"{tokens/dt:.0f}",
                        f"{loss:.4f}", f"{torch.cuda.max_memory_allocated(dev)/GiB:.2f}", f"{torch.cuda.memory_reserved(dev)/GiB:.2f}",
                        f"{(gate_ns['v']-g0)/1e6:.0f}", f"{budget/GiB:.2f}", f"{duty:.2f}", f"{act_stats['out_b']/2**20:.0f}", f"{act_stats['in_b']/2**20:.0f}", snap_ms, event, green["sms"],
                        "" if not cap else f"{cap:.0f}", f"{sleep_ms:.0f}", pace, pst["slices"], f"{pst['sleep_s']*1e3:.0f}", "" if not gbps else f"{float(gbps):.1f}", pst["units"],
                        f"{window_ms:g}" if bst["active"] else "", bst["late"], f"{bst['late_s']*1e3:.1f}", f"{bst['forgiven_s']*1e3:.1f}"])
            out.flush()
        if step % 20 == 0:
            print(f"  step {step} cfg {i} (tier {cfg['tier']} k={cfg['k']} off={cfg['off']}): {dt*1e3:.0f} ms, {tokens/dt:.0f} tok/s, "
                  f"peak {torch.cuda.max_memory_allocated(dev)/GiB:.2f} GiB, budget {budget/GiB:.2f}, gated {(gate_ns['v']-g0)/1e6:.0f} ms", flush=True)
        publish_state(seq, i)
    snapshot(step)
    if out: out.close()
    print(f"ft_worker: stopped at step {step}", flush=True)

main()
