"""CPU tests of the byte ledger (pacing.BytePacer) with a fake clock: no GPU, no sleeping.   python test_pacing_bytes.py
A fake tenant runs segment k in gb[k] / rate[k] seconds; sync is instantaneous (the fake work is done when it returns)."""
import math
from pacing import SegTable, BytePacer

class Clock:
    def __init__(self): self.t = 0.0; self.sleeps = []
    def now(self): return self.t
    def sleep(self, d): self.sleeps.append((self.t, d)); self.t += d
def run(gb, rates, gbps, window, steps=3, clock=None):
    """returns per step: period, the list of (time, GB done) at every checkpoint, and the pacer's report"""
    c = clock or Clock(); tab = SegTable(gb=gb); p = BytePacer(lambda: None, clock=c.now, sleep=c.sleep); out = []
    for _ in range(steps):
        t_start = c.t; p.begin_step(tab, gbps, window); trace = []
        for k in range(len(gb)):
            c.t += gb[k] / rates[k]                       # the tenant computes segment k
            if k < len(gb) - 1: p.tick(); trace.append((c.t, tab.cum[k + 1]))
        r = p.end_step(); out.append((c.t - t_start, trace, r))
    return out, c
def worst_window(trace, t0, window):
    """largest number of GB moved inside any stretch of `window` seconds (checked at checkpoint times)"""
    pts = [(t0, 0.0)] + trace; worst = 0.0
    for i, (ta, ga) in enumerate(pts):
        gb_in = max([g for t, g in pts[i:] if t - ta <= window + 1e-12] or [ga]) - ga; worst = max(worst, gb_in)
    return worst
ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name + (" — " + info if info else ""))

# 1. even tenant, faster than the budget: the period is bytes / budget, pauses fall at chunk borders
gb = [0.0] + [0.12] * 100; rates = [100.0] * len(gb); total = sum(gb)
res, c = run(gb, rates, gbps=50.0, window=0.012)
check("period = bytes / budget (sliced)", all(abs(p - total / 50.0) < 1e-6 for p, _, _ in res), f"{res[1][0]:.6f} s against {total / 50.0:.6f}")
check("pauses inside the step", res[1][2]["slices"] >= 15, f"{res[1][2]['slices']} pauses")
w = worst_window(res[1][1], sum(r[0] for r in res[:1]), 0.012)
check("no window of 12 ms holds more than a chunk plus one segment", w <= 50.0 * 0.012 + 0.12 + 1e-9, f"{w:.3f} GB, chunk {50 * 0.012:.2f}")
# 2. window 0: one pause after the step, same period
res0, _ = run(gb, rates, gbps=50.0, window=0.0)
check("window 0: only the pause after the step", res0[1][2]["slices"] == 0 and res0[1][2]["final_sleep_s"] > 0, f"final pause {res0[1][2]['final_sleep_s'] * 1e3:.0f} ms")
check("window 0: period = bytes / budget", all(abs(p - total / 50.0) < 1e-6 for p, _, _ in res0))
# 3. a lump that cannot be cut: the average still holds, the pause after it is long
gbl = [0.0] + [0.11] * 24 + [7.97] + [0.15] * 24; ratesl = [90.0] * 25 + [413.0] + [99.0] * 24; tl = sum(gbl)
resl, _ = run(gbl, ratesl, gbps=50.0, window=0.012)
check("lump: period = bytes / budget", all(abs(p - tl / 50.0) < 1e-6 for p, _, _ in resl), f"{resl[1][0] * 1e3:.1f} ms")
check("lump: nothing is late", resl[1][2]["late"] == 0)
check("lump: chunk borders listed", len(SegTable(gb=gbl).chunks(0.6)) < tl / 0.6, f"{len(SegTable(gb=gbl).chunks(0.6))} pauses for {tl / 0.6:.0f} chunks")
# 4. a segment slower than the budget allows: late, the debt is made up, at most one window is carried
gbs = [0.0] + [0.12] * 50 + [0.6] + [0.12] * 50; rs = [100.0] * 51 + [20.0] + [100.0] * 50; ts = sum(gbs)
ress, _ = run(gbs, rs, gbps=50.0, window=0.012, steps=4)
check("slow segment: reported late", ress[1][2]["late"] >= 1, f"late {ress[1][2]['late']} times, worst {ress[1][2]['late_s'] * 1e3:.1f} ms, forgiven {ress[1][2]['forgiven_s'] * 1e3:.1f} ms")
check("slow segment: the period stays within one window of bytes / budget", all(ts / 50.0 - 1e-9 <= p <= ts / 50.0 + 0.012 + 0.6 / 20.0 for p, _, _ in ress), f"{ress[2][0] * 1e3:.1f} ms against {ts / 50.0 * 1e3:.1f}")
# 5. tenant slower than the budget everywhere: never sleeps, the period is its own
resv, _ = run(gb, [30.0] * len(gb), gbps=50.0, window=0.012)
check("tenant slower than the budget: no pause", resv[1][2]["slices"] == 0 and resv[1][2]["final_sleep_s"] == 0 and abs(resv[1][0] - total / 30.0) < 1e-9)
# 6. a larger budget gives larger chunks, so fewer pauses for the same step
a = run(gb, rates, 25.0, 0.012)[0][1][2]["slices"]; b = run(gb, rates, 75.0, 0.012)[0][1][2]["slices"]
check("fewer pauses per step at a larger budget", b < a, f"{a} pauses at 25 GB/s, {b} at 75 GB/s")
# 7. table and step do not match: flagged
c2 = Clock(); p = BytePacer(lambda: None, clock=c2.now, sleep=c2.sleep); p.begin_step(SegTable(gb=[0.0, 1.0, 1.0]), 50.0, 0.012)
for _ in range(5): p.tick()
check("mismatch between table and step is flagged", p.end_step()["mismatch"])
# 8. inactive without a table
p.begin_step(None, 50.0, 0.012); p.tick(); r = p.end_step(); check("no table: inactive, units still counted", not r["active"] and r["units"] == 1)
print("ALL PASS" if ok else "SOME FAILED"); raise SystemExit(0 if ok else 1)
