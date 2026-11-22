#!/usr/bin/env python3
"""Fills the tables and numbers of reports/night_2026-11-22*.md from results_n21/night_runs.csv (analyze_night.py), so that no
number in the text is typed by hand.
  python make_night_tables.py             writes reports/night_2026-11-22_zh.md and reports/night_2026-11-22.md
The text lives in report/night_tmpl/: a_<lang>.md (sections that do not depend on the vLLM runs) and b_<lang>.tmpl (sections
that do); both are rendered. In a template, <<T:name>> is replaced by a table and <<V:group:key>> by one value (the mean over the repeats of a
setting). Every percentage is against the engine alone on the same night (the pooled baseline runs)."""
import csv, os, re, statistics as st
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, "results_n21"); TM = os.path.join(H, "report", "night_tmpl"); OUT = os.path.join(H, "..", "reports")
G = {}
for r in csv.DictReader(open(os.path.join(D, "night_runs.csv"))):
    G.setdefault(r["group"], []).append({k: float(v) for k, v in r.items() if k not in ("run", "group") and v not in ("", None)})
PCT = ("mean", "tpot50", "tpot90", "s50", "s90", "s99", "g90", "g95", "g99")
def fmt(k, x): return f"{x:+.1f} %" if k in PCT else f"{x:.0f} %" if k == "comp" else f"{x:,.0f}"
def value(g, k, zh=True):
    if g == "gain_tok_all":      # how much more the setting named in k trained over the whole 11 minutes than the whole-model compilation of Nov 16
        return f"{(st.mean(r['tok_all'] for r in G[k or 'cvfm_w12']) / st.mean(r['tok_all'] for r in G['c2a']) - 1) * 100:.0f} %"
    v = [r[k] for r in G.get(g, []) if k in r]
    return fmt(k, st.mean(v)) if v else ("（未测）" if zh else "(not measured)")
def cell(g, k, zh):
    v = [r[k] for r in G.get(g, []) if k in r]
    if not v: return "–"
    m = fmt(k, st.mean(v))
    if len(v) == 1 or (max(v) - min(v) < (0.15 if k in PCT else 30 if k != "comp" else 2)): return m          # repeats that agree: the mean alone
    return f"{m}（{fmt(k, min(v))} 到 {fmt(k, max(v))}）" if zh else f"{m} ({fmt(k, min(v))} to {fmt(k, max(v))})"
COLS = {"mean": ("decode 平均 token 间隔", "decode mean token gap"), "tpot50": ("TPOT 中位数", "TPOT median"), "tpot90": ("TPOT p90", "TPOT p90"), "s50": ("单步中位数", "single step, median"),
        "s90": ("单步 p90", "single step p90"), "s99": ("单步 p99", "single step p99"), "g90": ("所有间隔 p90", "all gaps p90"), "g95": ("所有间隔 p95（不过滤）", "all gaps p95 (nothing left out)"),
        "tok": ("tenant 进度（tok/s）", "tenant progress (tok/s)"), "tok_all": ("整个 11 分钟的 tenant 进度（tok/s）", "tenant progress over the 11 minutes (tok/s)"), "comp": ("tenant 在计算的时间占比", "share of time the tenant computes")}
FULL = ("mean", "tpot90", "s50", "s90", "s99", "g95", "tok", "tok_all")
TABLES = {
    "controls": (FULL, [("c1_fix", "普通 tenant，步后停", "usual tenant, pause after the step"), ("c2a", "整体编译的 tenant，步后停", "tenant compiled as a whole, pause after the step")]),
    "tenants": (FULL + ("comp",), [("c1_fix", "普通 tenant", "usual tenant"), ("v_step", "只改输出层的 tenant", "only the output layer changed"),
                                   ("c2a", "整体编译的 tenant（11 月 16 日最好的）", "compiled as a whole (the best of Nov 16)"), ("cvfm_step", "新 tenant", "new tenant")]),
    "slices": (FULL, [("v_step", "只改输出层的 tenant，步后停", "only the output layer changed, pause after the step"), ("v_w12", "只改输出层的 tenant，切片", "only the output layer changed, slices"),
                      ("cvfm_step", "新 tenant，步后停", "new tenant, pause after the step"), ("cvfm_w12", "新 tenant，切片", "new tenant, slices")]),
    "tradeoff": (FULL + ("comp",), [("cvfm_w12_half", "每秒 9.6 GB", "9.6 GB per second"), ("cvfm_w12", "每秒 19.2 GB", "19.2 GB per second"), ("cvfm_w12_x2", "每秒 38.4 GB", "38.4 GB per second")]),
    "sms": (("mean", "tpot90", "s90", "s99", "tok", "comp"), [("cvfm_w12_sm16", "16 个 SM", "16 SMs"), ("cvfm_w12", "32 个 SM", "32 SMs"), ("cvfm_w12_sm64", "64 个 SM", "64 SMs")]),
    "slo": (FULL, [("c1_slow5", "普通 tenant，目标 5 %（11 月 16 日的做法）", "usual tenant, target 5 % (as on Nov 16)"), ("cvfm_slow5_step", "新 tenant，目标 5 %，步后停", "new tenant, target 5 %, pause after the step"),
                   ("cvfm_slow5_w12", "新 tenant，目标 5 %，切片", "new tenant, target 5 %, slices"),
                   ("c1_slow15", "普通 tenant，目标 15 %（11 月 16 日的做法）", "usual tenant, target 15 % (as on Nov 16)"), ("cvfm_slow15_step", "新 tenant，目标 15 %，步后停", "new tenant, target 15 %, pause after the step"),
                   ("cvfm_slow15_w12", "新 tenant，目标 15 %，切片", "new tenant, target 15 %, slices")]),
    "others": (FULL, [("cvf_stepcut", "新 tenant，步后停，最后 13 秒被中断的一次", "new tenant, pause after the step, the run cut 13 s before its end"),
                      ("cvf_step", "新 tenant，步后停，沿用普通 tenant 的显存需求表", "new tenant, pause after the step, with the usual tenant's table of memory needs"),
                      ("cvfm0_step", "新 tenant，步后停，还在用“1 批、8 条”分法时", "new tenant, pause after the step, while it still computed a step as 1 batch of 8"),
                      ("cvfmhi_slow5_w12", "新 tenant，自动调节目标 5 %，切片，额度范围未改时", "new tenant, automatic budget, target 5 %, slices, before the range was changed")]),
}
def table(name, zh):
    cols, rows = TABLES[name]
    out = ["| " + ("设置" if zh else "setting") + " | " + ("次数" if zh else "runs") + " | " + " | ".join(COLS[c][0 if zh else 1] for c in cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for g, z, e in rows:
        if g in G: out.append(f"| {z if zh else e} | {len(G[g])} | " + " | ".join(cell(g, c, zh) for c in cols) + " |")
    return "\n".join(out)
def render(text, zh):
    text = re.sub(r"<<T:(\w+)>>", lambda m: table(m.group(1), zh), text)
    return re.sub(r"<<V:(\w+):?(\w*)>>", lambda m: value(m.group(1), m.group(2), zh), text)
if __name__ == "__main__":
    for lang, name in (("zh", "night_2026-11-22_zh.md"), ("en", "night_2026-11-22.md")):
        a, b = os.path.join(TM, f"a_{lang}.md"), os.path.join(TM, f"b_{lang}.tmpl")
        if not (os.path.exists(a) and os.path.exists(b)): print("no template for", lang); continue
        open(os.path.join(OUT, name), "w", encoding="utf-8").write(render(open(a, encoding="utf-8").read() + open(b, encoding="utf-8").read(), lang == "zh"))
        print("wrote", name)
