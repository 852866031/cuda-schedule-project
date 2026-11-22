#!/usr/bin/env python3
"""Storyline figure: what the VRAM scheduler does on each check, with the need formula written out.
Writes figures/story_03_scheduler.drawio and renders it to figures/story_03_scheduler.png (diagrams.net viewer in headless Chrome,
same method as coloc_adjust/diagrams/export_png.py). The numbers are the defaults of coloc_adjust/controller.py.
  python3 storyline/make_scheduler_fig.py"""
import html, json, os, subprocess, tempfile
from PIL import Image, ImageChops

H = os.path.dirname(os.path.abspath(__file__)); F = os.path.join(H, "figures")
BLUE, BLUE_D, BLUE_L = "#1a73e8", "#174ea6", "#d2e3fc"
ORG, ORG_D, ORG_L = "#e37400", "#8a4600", "#fde9c8"
GREY, INK = "#9aa0a6", "#202124"
cells = []
def esc(s): return html.escape(s, quote=True)
def box(i, x, y, w, h, head, lines, fill="#ffffff", stroke=GREY, shape="rounded=1;arcSize=6;", sw=1):
    body = f'<b style="font-size:15px">{head}</b>' + "".join(f'<div style="margin-top:5px">{l}</div>' for l in lines)
    cells.append(f'<mxCell id="{i}" value="{esc(body)}" style="{shape}whiteSpace=wrap;html=1;fontSize=13;align=left;verticalAlign=middle;spacingLeft=14;spacingRight=10;'
                 f'fillColor={fill};strokeColor={stroke};strokeWidth={sw};fontColor={INK};" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
def diamond(i, x, y, w, h, text):
    cells.append(f'<mxCell id="{i}" value="{esc(text)}" style="rhombus;whiteSpace=wrap;html=1;fontSize=13;fillColor=#f1f3f4;strokeColor={GREY};fontColor={INK};spacing=14;" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
def text(i, x, y, w, h, s, size=13, bold=False, color=INK, align="left"):
    cells.append(f'<mxCell id="{i}" value="{esc(s)}" style="text;html=1;whiteSpace=wrap;fontSize={size};fontStyle={1 if bold else 0};align={align};verticalAlign=middle;fontColor={color};" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
def edge(i, s, t, label="", color="#3c4043", sw=1, ex=None, en=None, pts=(), fc=None):
    st = f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;fontSize=12;labelBackgroundColor=#ffffff;strokeColor={color};strokeWidth={sw};fontColor={fc or color};"
    if ex: st += f"exitX={ex[0]};exitY={ex[1]};exitDx=0;exitDy=0;"
    if en: st += f"entryX={en[0]};entryY={en[1]};entryDx=0;entryDy=0;"
    arr = ('<Array as="points">' + "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts) + "</Array>") if pts else ""
    cells.append(f'<mxCell id="{i}" value="{esc(label)}" style="{st}" edge="1" source="{s}" target="{t}" parent="1"><mxGeometry relative="1" as="geometry">{arr}</mxGeometry></mxCell>')

text("t", 40, 8, 900, 34, "VRAM scheduler: one check every 100 ms", size=21, bold=True)
box("in", 40, 60, 250, 120, "Router's table", ["running and waiting requests", "arrival rate λ"])
box("need", 330, 60, 660, 120, "KV need&nbsp; = &nbsp;D &nbsp;+&nbsp; A &nbsp;+&nbsp; H", [
    "<b>D</b> &nbsp;KV of running + waiting requests, at their final length",
    "<b>A</b> &nbsp;arrivals expected in the next 1 s:&nbsp; λ × 1 s × KV per request",
    "<b>H</b> &nbsp;headroom, 0.5 GiB"], stroke=BLUE_D, sw=2)
diamond("q1", 40, 225, 250, 100, "need &gt; limit ?")
box("grow", 330, 225, 340, 100, "Grow: after 1 check", [
    "tenant's budget down first", "limit up, as far as memory is free"], fill=BLUE_L, stroke=BLUE, sw=2)
box("bud", 330, 370, 340, 70, "Tenant's budget", ["= card − engine − margins − <b>KV limit</b>"], stroke=ORG, sw=2)
box("shrink", 330, 485, 340, 100, "Shrink: after 50 checks in a row (5 s)", [
    "limit down, never below what requests hold", "tenant's budget up once memory is back"], fill=ORG_L, stroke=ORG, sw=2)
diamond("q2", 40, 485, 250, 100, "need &lt; limit − 1 GiB ?")
box("keep", 100, 625, 130, 40, "no change", [], fill="#f1f3f4")
box("kvc", 770, 355, 220, 100, "kvcached", ["maps and frees KV pages;", "running requests are never cut"], stroke=BLUE, sw=2)

edge("e1", "in", "need")
edge("e2", "need", "q1", ex=(0.05, 1), en=(0.5, 0), pts=((363, 203), (165, 203)))
edge("e3", "q1", "grow", "yes", color=BLUE_D)
edge("e4", "q1", "q2", "no")
edge("e5", "q2", "shrink", "yes", color=ORG_D)
edge("e6", "q2", "keep", "no")
edge("e7", "grow", "kvc", "set limit", color=BLUE, sw=2, ex=(1, 0.5), en=(0.5, 0), pts=((880, 275),), fc=BLUE_D)
edge("e8", "shrink", "kvc", "set limit", color=BLUE, sw=2, ex=(1, 0.5), en=(0.5, 1), pts=((880, 535),), fc=BLUE_D)
edge("e9", "grow", "bud", "budget down", color=ORG, sw=2, ex=(0.5, 1), en=(0.5, 0), fc=ORG_D)
edge("e10", "shrink", "bud", "budget up", color=ORG, sw=2, ex=(0.5, 0), en=(0.5, 1), fc=ORG_D)

xml = '<mxGraphModel adaptiveColors="auto"><root><mxCell id="0"/><mxCell id="1" parent="0"/>\n' + "\n".join(cells) + "\n</root></mxGraphModel>\n"
src = os.path.join(F, "story_03_scheduler.drawio"); open(src, "w", encoding="utf-8").write(xml)

W, Hh = 1400, 1400
cfg = {"nav": False, "resize": False, "toolbar": "", "edit": "", "border": 20, "xml": xml}
page = ('<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0;background:#fff}</style></head><body>'
        f'<div class="mxgraph" style="border:0;width:{W}px;height:{Hh}px;overflow:visible" data-mxgraph="{html.escape(json.dumps(cfg), quote=True)}"></div>'
        '<script src="https://viewer.diagrams.net/js/viewer-static.min.js"></script></body></html>')
tmp = tempfile.NamedTemporaryFile("w", suffix=".html", delete=False); tmp.write(page); tmp.close()
out = os.path.join(F, "story_03_scheduler.png")
subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                f"--window-size={W + 40},{Hh + 40}", "--virtual-time-budget=8000", f"--screenshot={out}", "file://" + tmp.name], check=True, capture_output=True)
os.unlink(tmp.name)
im = Image.open(out).convert("RGB"); bb = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
im.crop((max(0, bb[0] - 30), max(0, bb[1] - 30), bb[2] + 30, bb[3] + 30)).save(out, optimize=True); print(out, Image.open(out).size)
