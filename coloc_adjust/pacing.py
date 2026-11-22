"""Pacing helpers for the fine-tune tenant (C1 of PLAN_overnight3.md; BWS in reports/3_c1_c2.md). Pure Python, no GPU needed.

  Bpt          bytes moved per token, per tenant variant and shape (written by tenant_bytes.py)
  cap_tok_s    GB/s allowance -> tokens/s cap
  BlockPacer   the second method of C1 as run on Nov 16: slices of N blocks, equal work assumed per checkpoint
  SegTable     bytes per segment of one step, measured by segment_bytes.py (the profile)
  BytePacer    the byte ledger (Nov 21): a step is cut into chunks of equal bytes from the SegTable,
               chunk = budget x window, and the tenant pauses at chunk borders; window 0 = one pause after the step
"""
import csv, json, math, os, time

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


class SegTable:
    """Bytes per segment of one step of one tenant variant and shape (results_prof/seg_<variant>.csv).
    cum[k] is the GB moved by the time checkpoint k of the step is reached (k = 1 .. n); total the GB of a step."""
    def __init__(self, path=None, gb=None):
        if gb is None:
            rows = list(csv.DictReader(open(path))); gb = [float(r["total_gb"]) for r in rows]
        self.path = path; self.cum = [0.0]
        for x in gb: self.cum.append(self.cum[-1] + x)
        self.n = len(gb) - 1; self.total = self.cum[-1]; self.largest = max(gb)
    def chunks(self, chunk_gb):
        """the checkpoints at which a pause falls for this chunk size (for inspection and tests)"""
        out, nxt = [], chunk_gb
        for k in range(1, self.n + 1):
            if self.cum[k] >= nxt: out.append(k); nxt = (math.floor(self.cum[k] / chunk_gb) + 1) * chunk_gb
        return out

class BytePacer:
    """The byte ledger. tick() is called at every checkpoint of the tenant. The table says how many GB the step has
    moved when checkpoint k is reached; those bytes are 'paid for' at time t0 + GB / budget. The step is cut into
    chunks of `chunk` GB (budget x window): at the first checkpoint past each chunk border the tenant waits for the
    GPU (sync) and sleeps until the bytes done so far are paid for. end_step() settles the whole step, so the step's
    period is its bytes / budget whatever the window; with window 0 that is the only pause (BWS-Budget).
    Being late (a segment slower than the budget allows) is carried as a debt and made up by shorter pauses, up to
    `max_debt` seconds; anything beyond is forgiven, so a stall is not followed by a long burst."""
    def __init__(self, sync, clock=time.time, sleep=time.sleep, poll=None, min_sleep=2e-4, stop=lambda: False):
        self.sync, self.clock, self.sleep, self.poll, self.min_sleep, self.stop = sync, clock, sleep, poll, min_sleep, stop
        self.active = False; self.units = 0; self.debt = 0.0; self.table = None; self._zero()
    def _zero(self): self.slices = 0; self.sleep_s = 0.0; self.final_sleep_s = 0.0; self.late = 0; self.late_s = 0.0; self.forgiven_s = 0.0; self.mismatch = False
    def begin_step(self, table, gbps, window_s):
        """table: a SegTable for the shape of this step, or None (then nothing is paced here); gbps: the budget"""
        self.units = 0; self._zero(); self.table = table
        self.active = bool(table is not None and gbps and gbps > 0)
        if not self.active: self.debt = 0.0; return
        self.gbps = float(gbps); self.window = float(window_s or 0.0)
        self.chunk = self.gbps * self.window if self.window > 0 else float("inf")
        self.max_debt = self.window if self.window > 0 else 0.05
        self.next_border = self.chunk; self.t0 = self.clock() - min(self.debt, self.max_debt)
    def _settle(self, done_gb, final=False):
        self.sync()
        now = self.clock(); due = self.t0 + done_gb / self.gbps
        if due - now > self.min_sleep:
            while not self.stop():
                rest = due - self.clock()
                if rest <= 0: break
                self.sleep(min(0.05, rest))
                if self.poll: self.poll()
            slept = self.clock() - now
            if final: self.final_sleep_s += slept
            else: self.slices += 1; self.sleep_s += slept
        elif now > due:
            self.late += 1; self.late_s = max(self.late_s, now - due)
            if now - due > self.max_debt:                      # forgive what exceeds the allowed debt
                self.forgiven_s += now - due - self.max_debt; self.t0 += now - due - self.max_debt
    def tick(self):
        self.units += 1
        if not self.active: return
        if self.units > self.table.n: self.mismatch = True; return
        done = self.table.cum[self.units]
        if done < self.next_border: return
        self._settle(done); self.next_border = (math.floor(done / self.chunk) + 1) * self.chunk
    def end_step(self):
        r = dict(active=self.active, units=self.units)
        if self.active:
            if self.units != self.table.n: self.mismatch = True
            self._settle(self.table.total, final=True)
            self.debt = max(0.0, min(self.max_debt, self.clock() - (self.t0 + self.table.total / self.gbps)))
        r.update(slices=self.slices, sleep_s=self.sleep_s, final_sleep_s=self.final_sleep_s, late=self.late, late_s=self.late_s, forgiven_s=self.forgiven_s, mismatch=self.mismatch)
        self.active = False
        return r
