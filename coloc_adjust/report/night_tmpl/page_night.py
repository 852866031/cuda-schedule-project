#!/usr/bin/env python3
"""Puts the night report (reports/night_2026-11-22_zh.md) on the lavish page (.lavish/project-recap.html) as the section before the
BWS / BWR chapter, replacing the earlier profile section, and fixes the pointers to it. Can be run again after the report changes.
  python page_night.py"""
import os, re, shutil, sys
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, H); from md_to_page import convert
P = os.path.join(ROOT, ".lavish", "project-recap.html"); s = open(P, encoding="utf-8").read()
bak = os.path.join(ROOT, ".lavish", "project-recap.before-night.html")
if not os.path.exists(bak): shutil.copy(P, bak)
for f in ("prof_method.png", "prof_trace.png", "prof_variants.png", "night_tenants.png", "prof_trace_vocab2048.png", "night_methods.png", "night_tradeoff.png", "night_slo.png"):
    shutil.copy(os.path.join(ROOT, "coloc_adjust", "report", "figures", f), os.path.join(ROOT, ".lavish", "assets", f))
frag = convert(open(os.path.join(ROOT, "reports", "night_2026-11-22_zh.md"), encoding="utf-8").read())
frag = frag.replace("</h2>", '</h2>\n<p id="night-lead"><span class="pill ok">已测量</span>这一节是 11 月 21–22 日这一轮工作的完整讲述，从“把 tenant 每一段搬多少数据量出来”开始，到在 vLLM 上和 decode 一起跑的结果。它取代了之前这里的 profile 一节。文字版在 <code>reports/night_2026-11-22_zh.md</code>。</p>', 1)
a = s.find('<h2 id="night">'); a = a if a >= 0 else s.index('<h2 id="plan-prof">'); b = s.index('<h2 id="plan">')
s = s[:a] + frag + s[b:]
def sub(old, new, need=False):
    global s
    if old in s: s = s.replace(old, new)
    elif need: raise SystemExit("not found: " + old[:60])
# the block of the BWR chapter that repeated the tenant variants is now part of the night section
i = s.find('<h3 id="plan-c2all">')
if i >= 0: s = s[:i] + s[s.index('<h3 id="plan-c2b">'):]
s = s.replace('href="#plan-prof"', 'href="#night"').replace('href="#plan-c2v"', 'href="#night"').replace('href="#plan-c2all"', 'href="#night"')
sub('<a href="#night">profile</a><a href="#plan">', '<a href="#night">11 月 21–22 日</a><a href="#plan">')
sub('这三部分共同的基础是<a href="#night">上一节的 profile</a>，这里不重复。</p>',
    '<strong>11 月 21–22 日更新</strong>：这一章是 11 月 16 日的结果。之后的一轮（量 tenant 每一段搬的数据、改 tenant 的计算、改停顿的规则，并在 vLLM 上重新比较）在<a href="#night">上一节</a>，那里的结果比这一章新。</p>')
sub('<p><strong>还不能下结论：输出层按词表分块。</strong>在 tenant 这一边，它去掉了一步 39–48 % 的字节，最大的一块不超过 0.18 GB；主体再编译能到 50–58 %，速度和普通 tenant 一样（见<a href="#night">profile</a>和<a href="#night">BWR 的对比</a>）。它还没有在 decode 旁边跑过，所以不在第三层的表里。</p>\n',
    '<p><strong>11 月 21–22 日之后的方法</strong>（输出层按词表分块、逐层编译、融合 kernel、按数据量安排停顿）不在这张 11 月 16 日的表里，它们在 vLLM 上的结果见<a href="#night">上一节</a>。</p>\n')
sub('目前只量了 tenant 这一边</td><td><a href="#night">profile</a>、<a href="#night">下文</a></td></tr>', '和 decode 一起跑的结果见上一节</td><td><a href="#night">上一节</a></td></tr>')
# the plan of Nov 21 in the "next steps" section: say what became of it
i = s.find('<h3 id="open-plan">')
if i >= 0:
    j = s.index('<h3 id="open-why">')
    s = s[:i] + '''<h3 id="open-plan">下一步（11 月 22 日更新）</h3>
<p id="plan22-done">11 月 21 日列的清单里，下面这些已经做完，结果在<a href="#night">11 月 21–22 日一节</a>：量每一段搬的数据；按数据量安排停顿并核对；在 vLLM 上比较“步后停”和“切片”；让额度自动跟着 decode 的变慢程度调，加和不加切片；输出层的融合 kernel；逐层编译并保留停顿位置；每一步改动都在 decode 这一边验证。</p>
<ol id="plan22-next">
<li><strong>量 tenant 和 decode 一起跑时每段搬多少数据。</strong>现在用的是 tenant 单独跑时量的值。</li>
<li><strong>修自动调节偏保守的问题</strong>，让“目标 15 %”真的用到 15 %。</li>
<li><strong>24 层里剩下的数据量</strong>：新 tenant 的数据量 85 % 在这 24 层里；普通 tenant 的 24 层中 58 % 在 MLP 部分。这是下一个可以用融合 kernel 去削的地方。</li>
<li><strong>在真实数据上把一次训练跑到底</strong>，确认对 tenant 的三处改动不影响最终模型。</li>
<li>更高到达率、更大显存压力下的表现。</li>
</ol>
''' + s[j:]
# --- the summary layers of the page (page top, story line, the Nov 16 overview) also have to say what is best now ---
rep = open(os.path.join(ROOT, "reports", "night_2026-11-22_zh.md"), encoding="utf-8").read()
# the numbers written out below are the report's; stop if the report no longer has them
for v in ("+6.7 %", "+4.4 %", "+3.8 %", "+3.9 %", "+23.7 %", "+9.7 %", "+5.5 %", "+4.5 %", "3,538", "3,570", "3,734", "3,728", "57.2 GB", "21.9 GB",
          "62 %", "7.97 GB", "+5.7 %", "+3.3 %", "+2.6 %", "+6.2 %", "1,812", "7,011"):
    if v not in rep: raise SystemExit("the report no longer contains " + v + "; update the summary blocks in page_night.py")
def block(name, html, at):
    """Puts html into the page at index at(s), between markers, so that a second run replaces it."""
    global s
    a, b = "<!-- n22:%s:start -->" % name, "<!-- n22:%s:end -->" % name
    s = re.sub(re.escape(a) + ".*?" + re.escape(b), "", s, flags=re.S)
    i = at(s); s = s[:i] + a + html + b + s[i:]
def after(start, end=None):
    def at(t):
        i = t.index(start)
        return i + len(start) if end is None else t.index(end, i) + len(end)
    return at
def hid(prefix):
    m = re.search(r'<h[34] id="(night-\d+)"[^>]*>' + re.escape(prefix), s)
    if not m: raise SystemExit("no night heading starts with " + prefix)
    return m.group(1)
h34, h53, h6 = hid("3.4"), hid("5.3"), hid("6.")
block("latest", '''
<div class="note" id="latest">
<p id="latest-h"><strong>最新结果（11 月 21–22 日）：现在最好的做法是“新 tenant”。</strong>下面四行都是在 vLLM 服务上和 decode 一起跑时量的，统计每秒 0.5 和 1 个请求两档负载。三种 tenant 的额度各自设成让它做差不多的训练量（每秒约 3,500–3,700 个 token）。百分比是和同一晚没有 tenant 时比：“平均”一列比的是 decode 的平均 token 间隔，“p99”一列比的是 decode 单步时间的 p99。</p>
<div class="scroll"><table class="tw" id="latest-t">
<tr><th style="min-width:0">tenant，和它怎么停</th><th style="min-width:7rem">decode 平均慢</th><th style="min-width:7.5rem">decode 单步时间的 p99</th><th style="min-width:6.5rem">tenant 进度，tok/s</th></tr>
<tr><td>普通 tenant，每步之后停（BWS-Budget）</td><td>+6.7 %</td><td>+23.7 %</td><td>3,538</td></tr>
<tr><td>整体编译的 tenant，每步之后停（BWS-Budget + BWR-Compile，11 月 16 日最好的）</td><td>+4.4 %</td><td>+9.7 %</td><td>3,570</td></tr>
<tr class="ok"><td><strong>新 tenant</strong>，每步之后停</td><td><strong>+3.8 %</strong></td><td><strong>+5.5 %</strong></td><td>3,734</td></tr>
<tr class="ok"><td><strong>新 tenant</strong>，在一步中间切片停</td><td><strong>+3.9 %</strong></td><td><strong>+4.5 %</strong></td><td>3,728</td></tr>
</table></div>
<p id="latest-what"><strong>新 tenant 是什么</strong>：训练内容不变，计算方式改了三处。输出层和 loss 沿词表一块一块地算；模型的 24 层逐层编译；每个词表块里的小计算合并成一个融合的 kernel。同一种分批方式（4 批、每批 2 条）下，它一步搬 21.9 GB 数据，普通 tenant 搬 57.2 GB（这两个数是 tenant 单独跑时量的）。</p>
<p id="latest-where"><strong>在这一页的哪里</strong>：<a href="#night">11 月 21–22 日一节</a>。其中：<a href="#H34">3.4 新 tenant 的具体配置</a> · <a href="#H53">5.3 和 decode 一起跑的对比</a> · <a href="#H6">6 结论</a>。再往后的 <a href="#plan">BWS、BWR 一章</a>是 11 月 16 日的结果，比这里旧。</p>
</div>
'''.replace("H34", h34).replace("H53", h53).replace("H6", h6), after('<p class="lede" id="answer">', '</p>'))
sub('前八步是已测量的，最后一步是计划。', '最后两步是 11 月 16 日和 11 月 21–22 日两轮的方法和实测。')
sub('<h4>BWS 和 BWR：做出来并测了 <span class="pill ok">', '<h4>BWS 和 BWR：做出来并测了（11 月 16 日） <span class="pill ok">')
block("story", '''
<p class="next">留下三个问题：切片没有用；decode 最慢的 1 %% 的步慢两成多，原因不明；只知道 tenant 一步总共搬多少数据，不知道哪一段搬得多。下一个目标：从最后一个入手，把每一段量出来。</p></li>

<li id="st-9" data-n="9" class="s4"><div class="card"><details><summary><h4>量清每一段，改 tenant，按数据量停（11 月 21–22 日） <span class="pill ok">已测量</span> <a href="#night">11 月 21–22 日一节</a></h4><p class="concl">tenant 做同样多的训练时，decode 平均慢 +3.8 %%（上一步的方法是 +6.7 %%），最慢 1 %% 的步从 +23.7 %% 降到 +5.5 %%。这是现在最好的做法。</p></summary>
<dl>
<dt>已知</dt><dd>上一步留下的三个问题（见上面的蓝字）。</dd>
<dt>做了</dt><dd><ul>
<li><strong>量</strong>：tenant 单独跑，量出一步里每一段搬多少数据。</li>
<li><strong>改 tenant 的计算</strong>，三处：输出层和 loss 沿词表分块算；24 层逐层编译；每个词表块里的小计算合并成融合的 kernel。三处都做的叫“新 tenant”（<a href="#%s">具体配置</a>）。</li>
<li><strong>改停顿的规则</strong>：按每一段实际搬的数据量安排停顿，可以每步之后停，也可以在一步中间切片停。</li>
<li><strong>比较</strong>：在 vLLM 服务上和 decode 一起跑 11 分钟的参考负载，和同一晚没有 tenant 时比。</li>
</ul></dd>
<dt>发现</dt><dd><ul>
<li>普通 tenant 一步搬 57.2 GB，其中 55 %% 在输出层和 loss：四块各 7.97 GB，中间不能停。decode 最慢的那 1 %% 就是它们造成的。</li>
<li>新 tenant 一步搬 21.9 GB，少 62 %%；单独跑的速度不变；每一层之后都能停。</li>
<li>和 decode 一起跑、tenant 进度相同（每秒约 3,500–3,700 个 token）：decode 平均 +6.7 %% → +3.8 %%，单步 p99 +23.7 %% → +5.5 %%。11 月 16 日最好的方法（整体编译）是 +4.4 %% 和 +9.7 %%（<a href="#%s">对比</a>）。</li>
<li>切片现在能用了，但作用有限：decode 单步的 p90 从 +5.7 %% 降到 +3.3 %%，平均不变。</li>
<li>decode 的代价跟着 tenant 搬的数据量走：新 tenant 的额度减半、翻倍，decode 平均是 +2.6 %% 和 +6.2 %%（tenant 每秒 1,812 和 7,011 个 token）；tenant 用 16、32、64 个 SM，代价几乎一样。</li>
</ul></dd>
<dt>结论</dt><dd>用新 tenant，停顿按实际搬的数据量安排（<a href="#%s">结论一节</a>）。</dd>
<dt>还不知道</dt><dd><ul>
<li>tenant 和 decode 一起跑时每段搬多少数据：没有量，用的是单独跑时的值。</li>
<li>额度自动调节偏保守，没有修。</li>
<li>只统计了每秒 0.5 和 1 个请求两档负载。</li>
</ul></dd>
</dl></details></div>''' % (h34, h53, h6), lambda t: t.index('</li>\n</ol>\n<!-- story:end -->'))
sub('<h3 id="plan-layers">总览：做了什么，哪个方法最好</h3>', '<h3 id="plan-layers">总览（11 月 16 日这一批）：做了什么，哪个方法最好</h3>')
block("planlay", '''
<div class="note warn" id="plan-layers-upd"><p><strong>这张总览只包括 11 月 16 日的方法。</strong>现在最好的做法是 11 月 21–22 日的“新 tenant”：tenant 进度相同时，decode 平均慢 +3.8 %%、单步时间的 p99 高 +5.5 %%；同一晚重测，下面表里最好的 BWS-Budget + BWR-Compile 是 +4.4 %% 和 +9.7 %%，单用 BWS-Budget 是 +6.7 %% 和 +23.7 %%。见<a href="#latest">页首的表</a>和<a href="#%s">上一节的结论</a>。</p></div>
''' % h6, after('<h3 id="plan-layers">', '</h3>'))
sub('<p><strong>最好：BWS-Budget 加 BWR-Compile。</strong></p>', '<p><strong>这一批里最好：BWS-Budget 加 BWR-Compile。</strong></p>')
sub('<h3 id="plan-sum">总结论：现在的建议，和还不知道的</h3>', '<h3 id="plan-sum">11 月 16 日这一批的总结论：当时的建议，和还不知道的</h3>')
block("plansum", '''
<div class="note warn" id="plan-sum-upd"><p><strong>下面是 11 月 16 日的建议。</strong>11 月 21–22 日之后的建议是：用“新 tenant”，停顿按实际搬的数据量安排。见<a href="#%s">上一节的结论</a>。</p></div>
''' % h6, after('<h3 id="plan-sum">', '</h3>'))
sub('<strong>现在的建议配置</strong>', '<strong>当时的建议配置</strong>')
open(P, "w", encoding="utf-8").write(s)
ids = re.findall(r'id="([^"]+)"', s); dup = sorted({i for i in ids if ids.count(i) > 1}); dead = sorted({h for h in re.findall(r'href="#([^"]+)"', s) if h not in ids})
print("page written;", len(s), "chars; duplicate ids:", dup, "; links without a target:", dead)
