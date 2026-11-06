#!/usr/bin/env python3
"""Render each diagrams/*.drawio to PNG with the diagrams.net viewer in headless Chrome.
  python3 diagrams/export_png.py [out_dir]      (default: ../report/figures)
Needs google-chrome, Pillow and network access (viewer-static.min.js)."""
import glob, html, json, os, subprocess, sys, tempfile
from PIL import Image, ImageChops

here = os.path.dirname(os.path.abspath(__file__))
out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "report", "figures")
W, H = 1400, 1100  # generous canvas; the result is auto-cropped
for f in sorted(glob.glob(os.path.join(here, "*.drawio"))):
    cfg = {"nav": False, "resize": False, "toolbar": "", "edit": "", "border": 20, "xml": open(f).read()}
    page = ('<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0;background:#fff}</style></head><body>'
            f'<div class="mxgraph" style="border:0;width:{W}px;height:{H}px;overflow:visible" data-mxgraph="{html.escape(json.dumps(cfg), quote=True)}"></div>'
            '<script src="https://viewer.diagrams.net/js/viewer-static.min.js"></script></body></html>')
    tmp = tempfile.NamedTemporaryFile("w", suffix=".html", delete=False); tmp.write(page); tmp.close()
    out = os.path.join(out_dir, os.path.basename(f)[:-7] + ".png")
    subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                    "--force-device-scale-factor=2", f"--window-size={W + 40},{H + 40}", "--virtual-time-budget=8000",
                    f"--screenshot={out}", "file://" + tmp.name], check=True, capture_output=True)
    os.unlink(tmp.name)
    im = Image.open(out).convert("RGB")
    bbox = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    im.crop((max(0, bbox[0] - 30), max(0, bbox[1] - 30), bbox[2] + 30, bbox[3] + 30)).save(out, optimize=True)
    print(out, Image.open(out).size)
