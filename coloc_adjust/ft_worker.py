#!/usr/bin/env python
"""Tiered, preemptable LoRA fine-tune tenant (design §3.4).

Trains LoRA (r=16, q/k/v/o) on Qwen2.5-0.5B with deterministic synthetic batches; before each
optimizer step it reads the tenant budget published by the controller and picks the highest
tier whose calibrated peak fits:

  tier 0  full batch                               tier 1  micro-batches k=2/4/8 (grad accumulation)
  tier 2  + activation offload to pinned DRAM       tier 3  + frozen base weights of n layers streamed from DRAM
  tier 4  pause: snapshot, free the GPU, wait

The torch allocator cap follows the budget, so a mis-estimate OOMs this process, never the
engine; an OOM drops one tier. A "collision" rule forces k >= 4 when the engine's busy duty
cycle over the last second exceeds --collision-duty. Snapshots (LoRA weights, optimizer,
scheduler, step, RNG) go to pinned DRAM / /dev/shm every --snapshot-every steps, before every
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
ap.add_argument("--green-dynamic", action="store_true", help="take the SM count from the budget file's 'sms' field (controller-driven); --green-sms is the fallback")
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
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r,
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
model.enable_input_require_grads()
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
def gate(*_):
    poll_budget_midstep()
    if page is None or rd(0) != MAGIC: return
    t0 = time.monotonic_ns()
    if fence["ev"] is not None: fence["ev"].synchronize()
    while rd(2) and time.monotonic_ns() - rd(1) < STALE_NS:
        time.sleep(0.0002)
        poll_budget_midstep()          # a gated tenant must still release memory when asked
    ev = torch.cuda.Event(); ev.record(); fence["ev"] = ev
    gate_ns["v"] += time.monotonic_ns() - t0
for m in layers:
    m.register_forward_pre_hook(lambda *a: gate()); m.register_full_backward_pre_hook(lambda *a: gate())
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
green = {"ctx": None, "sms": 0, "req": 0, "t": 0.0}
def set_green(sms):
    """Switch the tenant's compute stream to a green context of `sms` SMs (0 = full device).
    Compares against the *requested* count (the driver rounds), switches at most every 5 s, and
    returns the caching allocator's blocks of the old stream (they cannot serve the new one)."""
    if sms == green["req"] or time.time() - green["t"] < 5.0: return
    torch.cuda.synchronize()
    old = green["ctx"]
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
budget_extra = {"sms": None}
def read_budget():
    if args.fixed_budget_gib is not None: return args.fixed_budget_gib * GiB, 0
    try:
        j = json.load(open(args.budget_file)); budget_extra["sms"] = j.get("sms"); return float(j["bytes"]), int(j.get("seq", 0))
    except Exception:
        return TOTAL_GIB * GiB, 0
def choose(budget, duty):
    if args.fixed_tier is not None: return args.fixed_tier
    for i, cfg in enumerate(TIERS[:-1]):
        if cfg["k"] and duty > args.collision_duty and cfg["k"] < 4: continue
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
    gate()
    for i in range(k):
        sub = ids[i::k]
        with act_offload(cfg["act"]):
            out = model(input_ids=sub, labels=sub)
            loss = out.loss / k
        if offload["mgr"] is not None:
            offload["mgr"].pre_forward() if False else None
            offload["mgr"].backward(loss)
        else:
            loss.backward()
        loss_sum += loss.item() * k
        if k > 1: gate()
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); pin_recycle()
    return loss_sum / k, tokens

def main():
    step = restore()
    out = open(args.out, "w", newline="") if args.out else None
    w = csv.writer(out) if out else None
    if w: w.writerow(["ts", "step", "cfg", "tier", "k", "off", "act", "dt_ms", "tok_s", "loss", "peak_gib", "reserved_gib",
                      "gated_ms", "budget_gib", "duty", "act_out_mb", "act_in_mb", "snapshot_ms", "event", "sms"])
    to_gpu(); cur = None; seq_seen = -1; t_start = time.time(); last_snap = step; up_since = {"cfg": None, "t": 0.0}
    if args.green_remaining_of: set_green(args.green_remaining_of)
    elif args.green_sms and not args.green_dynamic: set_green(args.green_sms)
    if args.calibrate:
        results = {}
        for i, cfg in enumerate(TIERS[:-1]):
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
                to_gpu(); set_ckpt(TIERS[i]["ckpt"]); set_weight_offload(TIERS[i]["off"]); torch.cuda.empty_cache(); apply_cap(budget); cur = i
            except torch.OutOfMemoryError:
                print(f"ft_worker: OOM while moving to cfg {i} (budget {budget/GiB:.2f}); pausing", flush=True)
                opt.zero_grad(set_to_none=True); to_cpu(); torch.cuda.empty_cache(); peaks[i] = max(peaks.get(i, 0), budget + 0.1 * GiB)
                cur = PAUSE; publish_state(seq, PAUSE, "paused-oom"); time.sleep(0.2); continue
            publish_state(seq, i, event)
        elif seq != seq_seen:
            apply_cap(budget); publish_state(seq, i); seq_seen = seq
        cfg = TIERS[i]
        ids = torch.randint(0, vocab, (args.batch, args.seq), generator=gen).to(dev)
        g0 = gate_ns["v"]; torch.cuda.reset_peak_memory_stats(dev); t0 = time.time(); preempt_state["cur"] = i
        try:
            loss, tokens = train_step(cfg, ids)
            torch.cuda.synchronize()
        except Preempt as e:
            preempt_state["cur"] = None; preempt_state["count"] += 1
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
        preempt_state["cur"] = None
        dt = time.time() - t0; step += 1
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
                        f"{(gate_ns['v']-g0)/1e6:.0f}", f"{budget/GiB:.2f}", f"{duty:.2f}", f"{act_stats['out_b']/2**20:.0f}", f"{act_stats['in_b']/2**20:.0f}", snap_ms, event, green["sms"]])
            out.flush()
        if step % 20 == 0:
            print(f"  step {step} cfg {i} (tier {cfg['tier']} k={cfg['k']} off={cfg['off']}): {dt*1e3:.0f} ms, {tokens/dt:.0f} tok/s, "
                  f"peak {torch.cuda.max_memory_allocated(dev)/GiB:.2f} GiB, budget {budget/GiB:.2f}, gated {(gate_ns['v']-g0)/1e6:.0f} ms", flush=True)
        publish_state(seq, i)
    snapshot(step)
    if out: out.close()
    print(f"ft_worker: stopped at step {step}", flush=True)

main()
