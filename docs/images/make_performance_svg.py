"""Render the README performance summary as a 16:9 SVG of grid tables.

Usage: python make_performance_svg.py out.svg          # light + dark via prefers-color-scheme (README)
       python make_performance_svg.py out.svg --dark   # dark only (for rendering the shareable PNG)
"""
import sys
from html import escape

# (label, simplellm, vllm, verdict, difference); verdict: win | tie | loss | match
LEFT = [
    ("Decode", "1,000 output tokens, tok/s", [
        ("gpt-oss-20b, batch 1", "381", "331", "win", "+15.2%"),
        ("gpt-oss-20b, batch 64", "9,844", "8,881", "win", "+10.8%"),
        ("gpt-oss-120b, batch 1", "268", "233", "win", "+14.9%"),
        ("gpt-oss-120b, batch 64", "4,780", "4,270", "win", "+12.0%"),
    ]),
    ("Max context", "122k-token prompts, tok/s", [
        ("gpt-oss-20b, batch 1", "266", "229", "win", "+16.2%"),
        ("gpt-oss-20b, batch 14", "709", "671", "win", "+5.6%"),
        ("gpt-oss-120b, batch 1", "188", "164", "win", "+14.3%"),
        ("gpt-oss-120b, batch 2", "268", "237", "win", "+13.4%"),
        ("time to first token, 20b / 120b", "9.6 / 14.6 s", "6.1 / 9.0 s", "loss", "1.6x slower"),
    ]),
]
RIGHT = [
    ("Online serving", "200 chat + coding requests, tok/s", [
        ("gpt-oss-20b, all at once", "3,665", "3,190", "win", "+14.9%"),
        ("gpt-oss-120b, all at once", "1,797", "1,899", "loss", "−5.4%"),
        ("gpt-oss-120b, 1 to 2.5 req/s", "582–1,327", "577–1,291", "tie", "tie to +2.1%"),
    ]),
    ("1,000 requests", "gpt-oss-20b, tok/s", [
        ("same concurrency (64)", "7,309", "5,923", "win", "+23.4%"),
        ("each at its max (256 vs 1,024)", "12,682", "22,482", "loss", "−43.6%"),
    ]),
    ("Accuracy", "GPQA / AIME 2025", [
        ("gpt-oss-20b (OpenAI: 56.8 / 37.1)", "56.3 / 36.3", "56.8 / 37.9", "match", "✓ matches"),
        ("gpt-oss-120b (OpenAI: 67.1 / 50.4)", "66.8 / 50.8", "65.5 / 51.7", "match", "✓ matches"),
    ]),
]

LIGHT = ("--bg:#fcfcfb; --table:#ffffff; --head:#f5f4f1; --line:#e4e3df; --ink:#0b0b0b; --ink2:#52514e; --ink3:#8a8984; "
         "--num:#6b6a65; --best-bg:rgba(42,120,214,0.13); --win:#0a7a0a; --loss:#b42828; --tie:#6b6a65;")
DARK = ("--bg:#121211; --table:#191918; --head:#1f1f1d; --line:#34342f; --ink:#ffffff; --ink2:#c3c2b7; --ink3:#8f8e86; "
        "--num:#a9a89f; --best-bg:rgba(85,152,231,0.22); --win:#6fd46f; --loss:#f08b8b; --tie:#a9a89f;")

W, H = 1640, 923
M, GAP = 48, 40
TW = (W - 2 * M - GAP) // 2                      # table width
TOP = 150
TITLE_H, HEAD_H, ROW_H, AFTER = 42, 40, 54, 28
COLS = [0, 300, 444, 588, TW]                    # column edges: label | SimpleLLM | vLLM | difference
PADX = 16
COLOR = {"win": "win", "loss": "loss", "tie": "tie", "match": "win"}
ARROW = {"win": "▲ ", "loss": "▼ ", "tie": "", "match": ""}

def text(x, y, s, cls, anchor="start", style=""):
    st = f' style="{style}"' if style else ""
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}"{st}>{escape(s)}</text>'

dark_only = "--dark" in sys.argv[2:]
theme = f"svg {{ {DARK} }}" if dark_only else (
    f"svg {{ {LIGHT} }}\n  @media (prefers-color-scheme: dark) {{ svg {{ {DARK} }} }}")
out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
       f'aria-label="SimpleLLM vs vLLM 0.31.0 on one H100: faster at decode in every batch and context tested, '
       f'slower at time to first token, 120b saturated serving and 1,000-way concurrency; accuracy matches OpenAI.">',
f"""<style>
  {theme}
  text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: var(--ink); }}
  .title {{ font-size: 40px; font-weight: 700; letter-spacing: -0.01em; }}
  .sub {{ font-size: 18px; fill: var(--ink3); }}
  .group {{ font-size: 21px; font-weight: 700; }}
  .unit {{ font-size: 15px; font-weight: 400; fill: var(--ink3); }}
  .colh {{ font-size: 13px; font-weight: 650; letter-spacing: .06em; fill: var(--ink3); }}
  .label {{ font-size: 17px; fill: var(--ink2); }}
  .num {{ font-size: 19px; font-variant-numeric: tabular-nums; fill: var(--num); }}
  .best {{ fill: var(--ink); font-weight: 700; }}
  .delta {{ font-size: 18px; font-weight: 700; font-variant-numeric: tabular-nums; }}
  .foot {{ font-size: 15px; fill: var(--ink3); }}
</style>""",
f'<rect width="{W}" height="{H}" rx="12" fill="var(--bg)"/>',
text(M, 74, "SimpleLLM vs vLLM 0.31.0", "title"),
text(M, 110, "gpt-oss-20b and gpt-oss-120b on one H100 80GB · official MXFP4 weights · greedy decoding · median of 3 runs", "sub")]

clip_id = 0

def tables(x0, groups):
    global clip_id
    y = TOP
    for title, unit, items in groups:
        out.append(f'<text x="{x0}" y="{y+28}" class="group">{escape(title)}<tspan class="unit" dx="12">{escape(unit)}</tspan></text>')
        y += TITLE_H
        th = HEAD_H + ROW_H * len(items)
        clip_id += 1
        out.append(f'<clipPath id="c{clip_id}"><rect x="{x0}" y="{y}" width="{TW}" height="{th}" rx="10"/></clipPath>')
        out.append(f'<g clip-path="url(#c{clip_id})">')
        out.append(f'<rect x="{x0}" y="{y}" width="{TW}" height="{th}" fill="var(--table)"/>')
        out.append(f'<rect x="{x0}" y="{y}" width="{TW}" height="{HEAD_H}" fill="var(--head)"/>')
        ry = y + HEAD_H
        for label, us, them, verdict, delta in items:
            col = {"win": 1, "loss": 2}.get(verdict)  # tint the cell holding the better number
            if col: out.append(f'<rect x="{x0+COLS[col]}" y="{ry}" width="{COLS[col+1]-COLS[col]}" height="{ROW_H}" fill="var(--best-bg)"/>')
            ry += ROW_H
        out.append('</g>')
        for i in range(len(items)):
            ly = y + HEAD_H + ROW_H * i
            out.append(f'<line x1="{x0}" x2="{x0+TW}" y1="{ly}" y2="{ly}" stroke="var(--line)"/>')
        for cx in COLS[1:-1]:
            out.append(f'<line x1="{x0+cx}" x2="{x0+cx}" y1="{y}" y2="{y+th}" stroke="var(--line)"/>')
        out.append(f'<rect x="{x0+0.5}" y="{y+0.5}" width="{TW-1}" height="{th-1}" rx="10" fill="none" stroke="var(--line)"/>')
        hb = y + 26
        out.append(text(x0 + COLS[2] - PADX, hb, "SIMPLELLM", "colh", "end"))
        out.append(text(x0 + COLS[3] - PADX, hb, "VLLM", "colh", "end"))
        out.append(text(x0 + COLS[4] - PADX, hb, "DIFFERENCE", "colh", "end"))
        ry = y + HEAD_H
        for label, us, them, verdict, delta in items:
            b = ry + 35
            out.append(text(x0 + PADX, b, label, "label"))
            out.append(text(x0 + COLS[2] - PADX, b, us, "num best" if verdict == "win" else "num", "end"))
            out.append(text(x0 + COLS[3] - PADX, b, them, "num best" if verdict == "loss" else "num", "end"))
            out.append(text(x0 + COLS[4] - PADX, b, ARROW[verdict] + delta, "delta", "end", f"fill:var(--{COLOR[verdict]})"))
            ry += ROW_H
        y += th + AFTER

tables(M, LEFT)
tables(M + TW + GAP, RIGHT)
fy = H - 30
out.append(text(M, fy, "▲ SimpleLLM faster    ▼ vLLM faster    tie = within 2% run-to-run noise    ✓ inside OpenAI's published 95% interval    blue = better number", "foot", style="white-space:pre"))
out.append(text(W - M, fy, "github.com/naklecha/simple-llm", "foot", "end"))
out.append("</svg>")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
print(W, H)
