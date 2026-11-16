#!/usr/bin/env python3
"""Draw the SVG diagrams of the next-steps section (12, 15, 16, 17) and render them to ../report/figures.
  python3 diagrams/make_svg.py        (needs google-chrome and Pillow; network for the web font)
Numbers in the diagrams come from results/knee_bw2_summary.md, results/overlap_summary.md and tiers.json."""
import os, subprocess, tempfile
from PIL import Image, ImageChops
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "..", "report", "figures")
INK, INK2, MUTED, LINE2 = "#18212f", "#465166", "#77818f", "#b9c1cd"
BLUE, ORANGE, AQUA, RED, GOOD, WARN = "#2a78d6", "#eb6834", "#1baf7a", "#cf3a3a", "#178a58", "#9a6a00"
CSS = f'''svg{{background:#fff}} text{{font-family:"IBM Plex Sans","Segoe UI",Helvetica,Arial,sans-serif}}
.d-box{{fill:#ffffff;stroke:{LINE2};stroke-width:1.4}}.d-soft{{fill:#eaf1fb;stroke:{BLUE};stroke-width:1.2}}.d-good{{fill:#e7f5ee;stroke:{GOOD};stroke-width:1.2}}
.d-warn{{fill:#fbf3dc;stroke:{WARN};stroke-width:1.2}}.d-t{{fill:{INK};font-size:13px}}.d-b{{fill:{INK};font-size:14px;font-weight:600}}
.d-m{{fill:{MUTED};font-size:12px}}.d-g{{fill:{GOOD};font-size:12.5px;font-weight:600}}.d-r{{fill:{RED};font-size:12.5px;font-weight:600}}.d-ln{{stroke:{INK2};stroke-width:1.5;fill:none}}'''
ARR = f'<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L8 4L0 8z" fill="{INK2}"/></marker></defs>'
def svg(w, h, body): return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}"><style>{CSS}</style><rect width="{w}" height="{h}" fill="#fff"/>{ARR}{body}</svg>\n'
O = f'fill="{ORANGE}"'; GL = f'stroke="{LINE2}" stroke-width="1" fill="none"'
D = {}

# ---- 12: C1, a byte allowance (first method), to scale over 3.2 s
X0, X1, WIN = 170, 620, 3200.0
X = lambda t: X0 + (X1 - X0) * t / WIN
y0 = 56; b = f'<rect x="{X0}" y="18" width="14" height="12" {O}/><text class="d-m" x="{X0+20}" y="29">the tenant\'s GPU-memory traffic</text>'
b += (f'<text class="d-b" x="10" y="{y0+22}">Today: fewer SMs</text><text class="d-m" x="10" y="{y0+41}">8 SMs · always running</text><path d="M{X0} {y0+44}H{X1}" {GL}/>'
      f'<rect x="{X0}" y="{y0+44-12.6:.1f}" width="{X1-X0}" height="12.6" {O}/><text class="d-t" x="{X0+6}" y="{y0+22}">moving all the time, about 37 GB/s</text>'
      f'<text class="d-m" x="640" y="{y0+14}">decode slowdown</text><text class="d-b" x="640" y="{y0+34}" style="font-size:15px">+5.4–7.5 %</text><text class="d-g" x="640" y="{y0+52}">measured</text>')
y0 = 156; bw = X(470) - X(0)
b += (f'<text class="d-b" x="10" y="{y0+22}">C1: an allowance</text><text class="d-m" x="10" y="{y0+41}">32 SMs · run, then pause</text><path d="M{X0} {y0+44}H{X1}" {GL}/>'
      + "".join(f'<rect x="{X(t):.1f}" y="{y0+4}" width="{bw:.1f}" height="40" {O}/>' for t in (100, 1675))
      + f'<path d="M{X0} {y0+44-12.6:.1f}H{X1}" stroke="{INK}" stroke-width="1.4" stroke-dasharray="6 4" fill="none"/>'
      f'<text class="d-b" x="{X(2180):.0f}" y="{y0+22}" style="font-size:12.5px">allowance (average)</text>'
      f'<text class="d-m" x="{X(335):.0f}" y="{y0-4}" text-anchor="middle">run: 0.47 s</text><text class="d-m" x="{X(1120):.0f}" y="{y0+22}" text-anchor="middle">pause: 1.1 s</text>'
      f'<text class="d-m" x="640" y="{y0+14}">decode slowdown</text><text class="d-t" x="640" y="{y0+34}">to be measured</text><text class="d-m" x="640" y="{y0+52}">experiment E1.1</text>')
ya = 232
b += f'<path class="d-ln" d="M{X0} {ya}H{X1}" stroke-width="1"/>' + "".join(f'<path class="d-ln" d="M{X(t):.1f} {ya}v6"/><text class="d-m" x="{X(t):.1f}" y="{ya+20}" text-anchor="middle">{int(t/1000)} s</text>' for t in (0, 1000, 2000, 3000))
D["12_c1_budget"] = svg(780, 262, b)

# ---- 15: from the measurements to the ideas
b = (f'<rect class="d-soft" x="10" y="12" width="300" height="86" rx="8"/><text class="d-b" x="24" y="36">Section 2.7: what to control</text><text class="d-t" x="24" y="60">decode\'s cost follows the tenant\'s bytes</text><text class="d-t" x="24" y="82">an SM limit is indirect and weak</text>'
     f'<rect class="d-good" x="10" y="110" width="300" height="108" rx="8"/><text class="d-b" x="24" y="134">Section 2.2: how to control it</text><text class="d-t" x="24" y="158">decode is at baseline while it pauses</text><text class="d-t" x="24" y="180">cost grows with the time it runs</text><text class="d-t" x="24" y="202">cheapest per token on 16–64 SMs</text>'
     f'<path class="d-ln" d="M310 55C335 55 335 92 358 96" marker-end="url(#arr)"/><path class="d-ln" d="M310 164C335 164 335 140 358 136" marker-end="url(#arr)"/>'
     f'<rect class="d-warn" x="360" y="36" width="230" height="160" rx="8"/><text class="d-b" x="374" y="62">C1 byte budget</text><text class="d-t" x="374" y="88">quantity: GB per second</text><text class="d-t" x="374" y="112">means: run, then pause</text>'
     f'<text class="d-t" x="374" y="136">SMs: enough, 16–32</text><text class="d-t" x="374" y="160">allowance: steady</text><text class="d-m" x="374" y="182">two methods · not built</text>'
     f'<path class="d-ln" d="M590 116H636" marker-end="url(#arr)"/><rect class="d-box" x="638" y="76" width="132" height="80" rx="8"/><text class="d-b" x="704" y="102" text-anchor="middle">First: E1.1</text><text class="d-m" x="704" y="122" text-anchor="middle">tests this pairing</text><text class="d-m" x="704" y="140" text-anchor="middle">about 15 min GPU</text>'
     f'<text class="d-m" x="10" y="238">another idea, the finer method of C1, and one idea that the measurements rule out</text>'
     f'<rect class="d-warn" x="10" y="246" width="246" height="46" rx="8"/><text class="d-b" x="22" y="266" style="font-size:13px">C2 a tenant that uses less</text><text class="d-m" x="22" y="284">same work, fewer bytes moved</text>'
     f'<rect class="d-warn" x="267" y="246" width="246" height="46" rx="8"/><text class="d-b" x="279" y="266" style="font-size:13px">C1, finer method</text><text class="d-m" x="279" y="284">one burst spans about 30 decode steps</text>'
     f'<rect class="d-box" x="524" y="246" width="246" height="46" rx="8"/><text class="d-b" x="536" y="266" style="font-size:13px">Timing: not pursued</text><text class="d-m" x="536" y="284">no gaps between decode steps</text>')
D["15_measurements_to_ideas"] = svg(780, 300, b)

# ---- 16: C1, the finer method, at the scale of single decode steps (1.6 s)
A0, A1, W3, STEP = 150, 760, 1600.0, 15.0
XX = lambda t: A0 + (A1 - A0) * t / W3; n = int(W3 / STEP)
NORM, FULL, PART = f'fill="{BLUE}" opacity=".28"', f'fill="{RED}"', f'fill="{RED}" opacity=".32"'
cells = lambda y, f: "".join(f'<rect x="{XX(k*STEP)+0.6:.1f}" y="{y}" width="{XX(STEP)-XX(0)-1.2:.1f}" height="16" {f(k*STEP)}/>' for k in range(n))
b = (f'<rect x="150" y="16" width="14" height="12" {O}/><text class="d-m" x="170" y="27">tenant working</text><rect x="290" y="16" width="14" height="12" {NORM}/><text class="d-m" x="310" y="27">decode step, no overlap</text>'
     f'<rect x="480" y="16" width="14" height="12" {FULL}/><text class="d-m" x="500" y="27">full overlap</text><rect x="600" y="16" width="14" height="12" {PART}/><text class="d-m" x="620" y="27">partial overlap</text>')
ra = 62
b += (f'<text class="d-b" x="10" y="{ra}">First method: one burst of 0.47 s, then a pause of 1.1 s</text><text class="d-m" x="10" y="{ra+30}">tenant</text><rect x="{XX(100):.1f}" y="{ra+12}" width="{XX(470)-XX(0):.1f}" height="26" {O}/>'
      f'<text class="d-m" x="10" y="{ra+64}">decode, step by step</text>{cells(ra+50, lambda t: FULL if 100-STEP < t < 570 else NORM)}'
      f'<text class="d-r" x="{XX(335):.0f}" y="{ra+84}" text-anchor="middle">about thirty steps overlap the tenant</text><text class="d-m" x="{XX(1090):.0f}" y="{ra+84}" text-anchor="middle">these steps do not</text>')
rb = 186
b += (f'<text class="d-b" x="10" y="{rb}">Second method: the same work in slices of a few milliseconds</text><text class="d-m" x="10" y="{rb+30}">tenant</text>'
      + "".join(f'<rect x="{XX(k*STEP+5):.1f}" y="{rb+12}" width="{XX(4.5)-XX(0):.1f}" height="26" {O}/>' for k in range(n))
      + f'<text class="d-m" x="10" y="{rb+64}">decode, step by step</text>{cells(rb+50, lambda t: PART)}<text class="d-m" x="{XX(800):.0f}" y="{rb+84}" text-anchor="middle">every step overlaps one short slice</text>')
ya = 290
b += f'<path class="d-ln" d="M{A0} {ya}H{A1}" stroke-width="1"/>' + "".join(f'<path class="d-ln" d="M{XX(t):.1f} {ya}v6"/><text class="d-m" x="{XX(t):.1f}" y="{ya+20}" text-anchor="middle">{t/1000:g} s</text>' for t in (0, 500, 1000, 1500))
D["16_c1_finer"] = svg(780, 320, b)

# ---- 17: the five tenant shapes (memory from tiers.json; speed and slowdown while running from overlap_summary.md)
shapes = [("4 micro-batches", 5.0, 11.4, 22.3), ("8 micro-batches", 3.2, 6.2, 15.6), ("+ activations moved to host RAM", 2.5, 2.8, 16.8), ("+ 12 layers of weights streamed", 2.2, 2.2, 21.4), ("+ 20 layers of weights streamed", 1.9, 2.0, 19.6)]
MX, SX, CX = 250, 405, 585
b = (f'<text class="d-b" x="10" y="44">Tenant shape</text><text class="d-b" x="{MX}" y="44">GPU memory needed</text><text class="d-m" x="{MX}" y="62">GiB</text>'
     f'<text class="d-b" x="{SX}" y="44">Tenant speed</text><text class="d-m" x="{SX}" y="62">while running, tokens/s</text><text class="d-b" x="{CX}" y="44">Decode slowdown</text><text class="d-m" x="{CX}" y="62">while the tenant runs</text><path d="M10 72H770" {GL}/>')
for i, (nm, mem, spd, ov) in enumerate(shapes):
    y = 84 + i * 36
    b += (f'<text class="d-t" x="10" y="{y+15}">{nm}</text><rect x="{MX}" y="{y}" width="{mem*20:.0f}" height="20" rx="3" fill="{AQUA}"/><text class="d-t" x="{MX+mem*20+7:.0f}" y="{y+15}">{mem:.1f}</text>'
          f'<rect x="{SX}" y="{y}" width="{spd*10.5:.0f}" height="20" rx="3" {O}/><text class="d-t" x="{SX+spd*10.5+7:.0f}" y="{y+15}">{spd:.1f}k</text>'
          f'<rect x="{CX}" y="{y}" width="{ov*5.15:.0f}" height="20" rx="3" fill="{RED}"/><text class="d-t" x="{CX+ov*5.15+7:.0f}" y="{y+15}">+{ov:.0f} %</text>')
b += (f'<path d="M{MX} 268V278H{MX+125}V268" stroke="{INK2}" stroke-width="1.2" fill="none"/><text class="d-m" x="{MX+62}" y="296" text-anchor="middle">today: only this column</text>'
      f'<path d="M{SX} 268V278H{CX+150}V268" stroke="{WARN}" stroke-width="1.6" fill="none"/><text x="{(SX+CX+150)/2:.0f}" y="296" text-anchor="middle" fill="{WARN}" font-size="12.5" font-weight="600">these two set what a token costs decode</text>')
D["17_shapes"] = svg(780, 312, b)

for name, s in D.items():
    open(os.path.join(HERE, name + ".svg"), "w").write(s)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".html", delete=False)
    tmp.write(f'<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap"><style>body{{margin:20px;background:#fff}}</style></head><body>{s}</body></html>'); tmp.close()
    out = os.path.join(FIG, name + ".png")
    subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2", "--window-size=900,900", "--virtual-time-budget=8000", f"--screenshot={out}", "file://" + tmp.name], check=True, capture_output=True)
    os.unlink(tmp.name); im = Image.open(out).convert("RGB"); bb = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    im.crop((max(0, bb[0] - 30), max(0, bb[1] - 30), min(im.size[0], bb[2] + 30), min(im.size[1], bb[3] + 30))).save(out, optimize=True); print(out, Image.open(out).size)
