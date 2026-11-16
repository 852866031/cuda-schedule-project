"""Pacing helpers for the fine-tune tenant (C1 of PLAN_overnight3.md). Pure Python, no GPU needed.

  Bpt          bytes moved per token, per tenant variant and shape (written by tenant_bytes.py)
  cap_tok_s    GB/s allowance -> tokens/s cap
  BlockPacer   the second method of C1: keep the allowance inside a step, in slices of N blocks
"""
import json, math, os, time

DEFAULT_MB_PER_TOKEN = 13.8     # usual shape (4 micro-batches) on 32 SMs, Nov 9 (results/knee_bw2_summary.md)

class Bpt:
    """MB of GPU-memory traffic per trained token. File: {"variant": {"cfg index": MB, ...}, ...}.
    A variant measured for one shape only is scaled to the other shapes with the base variant's ratios."""
    def __init__(self, path=None, variant="base", default=DEFAULT_MB_PER_TOKEN):
        self.variant, self.default, self.t = variant, default, {}
        if path and os.path.exists(path):
            try: self.t = {v: {int(c): float(x) for c, x in d.items()} for v, d in json.load(open(path)).items() if isinstance(d, dict)}
            except Exception as e: print(f"pacing: could not read {path}: {e}", flush=True)
    def mb(self, cfg, variant=None):
        v = self.t.get(variant or self.variant, {}); base = self.t.get("base", {})
        if cfg in v: return v[cfg]
        if v and base:                                    # scale: same ratio to base as at a shape measured for both
            common = [c for c in v if c in base]
            if common and cfg in base: return base[cfg] * v[common[0]] / base[common[0]]
        if v: return next(iter(v.values()))               # shape not measured: this variant's first measured shape
        return base.get(cfg, next(iter(base.values())) if base else self.default)

def cap_tok_s(gbps, mb_per_token):
    """tokens/s that move `gbps` GB/s at `mb_per_token` MB per token"""
    return gbps * 1e3 / mb_per_token

def slow_update(gbps, rel_slowdown, target, gain, step, lo, hi):
    """one period of the VRAM scheduler's slow TPOT loop: multiplicative, at most `step` up or down, clamped to [lo, hi]"""
    factor = min(1.0 + step, max(1.0 - step, math.exp(-gain * (rel_slowdown - target))))
    return min(hi, max(lo, gbps * factor))

class BlockPacer:
    """Second method of C1. tick() is called from every block hook (forward and backward pre-hook of each
    layer). Every `slice_blocks` ticks it waits for the queued kernels (sync) and then sleeps until the
    work done so far is allowed to be finished at `cap` tokens/s. Work per tick is tokens / ticks of the
    previous step, so the first step of a shape is not paced inside (the pause after the step still holds
    the average). Being late earns no credit, so a stall is not followed by a burst."""
    def __init__(self, sync, clock=time.time, sleep=time.sleep, poll=None):
        self.sync, self.clock, self.sleep, self.poll = sync, clock, sleep, poll
        self.units_prev = {}; self.active = False; self.units = 0; self.slices = 0; self.sleep_s = 0.0; self.key = None
    def begin_step(self, key, tokens, cap, slice_blocks):
        """key identifies the shape (ticks per step differ between shapes)"""
        self.key = key; self.units = 0; self.slices = 0; self.sleep_s = 0.0
        n = self.units_prev.get(key)
        self.active = bool(cap and slice_blocks and n)
        if self.active:
            self.cap = float(cap); self.slice = int(slice_blocks); self.tok_per_unit = tokens / n; self.t0 = self.clock()
    def tick(self):
        self.units += 1
        if not self.active or self.units % self.slice: return
        self.sync()
        now = self.clock(); due = self.t0 + self.units * self.tok_per_unit / self.cap
        if due > now:
            self.slices += 1
            while True:
                rest = due - self.clock()
                if rest <= 0: break
                self.sleep(min(0.05, rest))
                if self.poll: self.poll()
            self.sleep_s += self.clock() - now
        else:
            self.t0 += now - due          # late: the schedule restarts here, lost time is not made up
    def end_step(self):
        if self.key is not None and self.units: self.units_prev[self.key] = self.units
        self.active = False
        return dict(units=self.units, slices=self.slices, sleep_s=self.sleep_s)
