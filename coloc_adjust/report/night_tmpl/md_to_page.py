#!/usr/bin/env python3
"""Turns reports/night_2026-11-22_zh.md into an HTML fragment for the lavish page (.lavish/project-recap.html): only the Markdown
this report uses (headings, paragraphs, lists with continuation lines, tables, images with an italic caption, bold, code, rules).
  python md_to_page.py <report.md> > fragment.html
Headings: the title becomes the section's h2 (id night); ## -> h3, ### -> h4. Images are taken from assets/<file name>."""
import html, os, re, sys
def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    return re.sub(r"(?<![*\w])\*([^*]+)\*(?![*\w])", r"<em>\1</em>", t)
def convert(md):
    L = md.split("\n"); out = []; i = 0; n = len(L); sec = 0
    def para(lines): return "<p>" + inline(" ".join(x.strip() for x in lines)) + "</p>"
    while i < n:
        s = L[i]
        if not s.strip(): i += 1; continue
        if s.startswith("# "):
            out.append(f'<h2 id="night"><span class="sec">11 月 21–22 日这一轮</span>{inline(s[2:])}</h2>'); i += 1; continue
        m = re.match(r"(#{2,3}) (.*)", s)
        if m:
            sec += 1; tag = "h3" if len(m.group(1)) == 2 else "h4"; st = ' style="margin:1.6em 0 .4em;font-size:1rem"' if tag == "h4" else ""
            out.append(f'<{tag} id="night-{sec}"{st}>{inline(m.group(2))}</{tag}>'); i += 1; continue
        if s.strip() == "---": i += 1; continue
        if s.lstrip().startswith("!["):
            m = re.match(r"\s*!\[([^\]]*)\]\(([^)]+)\)", s); i += 1
            while i < n and not L[i].strip(): i += 1
            cap = []
            if i < n and L[i].lstrip().startswith("*") and not L[i].lstrip().startswith("**"):
                while i < n and L[i].strip(): cap.append(L[i].strip()); i += 1
            c = inline(" ".join(cap).strip("*")) if cap else ""
            out.append(f'<figure class="wide"><img src="assets/{os.path.basename(m.group(2))}" alt="{html.escape(m.group(1))}" loading="lazy">' + (f"<figcaption>{c}</figcaption>" if c else "") + "</figure>"); continue
        if s.lstrip().startswith("|"):
            rows = []
            while i < n and L[i].lstrip().startswith("|"): rows.append([c.strip() for c in L[i].strip().strip("|").split("|")]); i += 1
            body = [r for k, r in enumerate(rows) if not (k == 1 and all(re.fullmatch(r":?-+:?", c) for c in r))]
            h = "<tr>" + "".join(f'<th style="min-width:0">{inline(c)}</th>' for c in body[0]) + "</tr>"
            out.append('<div class="scroll"><table class="tw">' + h + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body[1:]) + "</table></div>"); continue
        m = re.match(r"(\s*)(-|\d+\.) (.*)", s)
        if m:
            tag = "ul" if m.group(2) == "-" else "ol"; items = []
            while i < n:
                m2 = re.match(r"(\s*)(-|\d+\.) (.*)", L[i])
                if m2 and len(m2.group(1)) <= len(m.group(1)) + 1: items.append([m2.group(3)]); i += 1
                elif L[i].strip() and (L[i].startswith("  ") or L[i].startswith("\t")) and not L[i].lstrip().startswith(("|", "![")): items[-1].append(L[i].strip()); i += 1
                elif not L[i].strip() and i + 1 < n and re.match(r"(\s*)(-|\d+\.) ", L[i + 1]) and (L[i + 1].lstrip()[0] == "-") == (tag == "ul"): i += 1
                else: break
            out.append(f'<{tag}' + (' class="key"' if tag == "ul" else "") + ">" + "".join("<li>" + inline(" ".join(x)) + "</li>" for x in items) + f"</{tag}>"); continue
        buf = []
        while i < n and L[i].strip() and not L[i].startswith("#") and not L[i].lstrip().startswith(("|", "![")) and not re.match(r"\s*(-|\d+\.) ", L[i]) and L[i].strip() != "---": buf.append(L[i]); i += 1
        out.append(para(buf))
    return "\n".join(out) + "\n"
if __name__ == "__main__":
    print(convert(open(sys.argv[1], encoding="utf-8").read()))
