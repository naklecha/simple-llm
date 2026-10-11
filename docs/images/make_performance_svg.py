"""Render the README performance summary as an SVG table (light + dark via prefers-color-scheme)."""
import sys
from html import escape

# (label, simplellm, vllm, verdict, pill text); verdict: win | tie | loss | match
GROUPS = [
    ("Decode, 1,000 output tokens", "output tok/s", [
        ("gpt-oss-20b, batch 1", "381", "331", "win", "+15.2%"),
        ("gpt-oss-20b, batch 64", "9,844", "8,881", "win", "+10.8%"),
        ("gpt-oss-120b, batch 1", "268", "233", "win", "+14.9%"),
        ("gpt-oss-120b, batch 64", "4,780", "4,270", "win", "+12.0%"),
    ]),
    ("Max context, 122k-token prompts, 8,192 output tokens", "output tok/s", [
        ("gpt-oss-20b, batch 1", "266", "229", "win", "+16.2%"),
        ("gpt-oss-20b, batch 14", "709", "671", "win", "+5.6%"),
        ("gpt-oss-120b, batch 1", "188", "164", "win", "+14.3%"),
        ("gpt-oss-120b, batch 2", "268", "237", "win", "+13.4%"),
        ("time to first token, 20b / 120b", "9.6 / 14.6 s", "6.1 / 9.0 s", "loss", "1.6x slower"),
    ]),
    ("Online serving, 200 chat + coding requests", "output tok/s", [
        ("gpt-oss-20b, all sent at once", "3,665", "3,190", "win", "+14.9%"),
        ("gpt-oss-120b, all sent at once", "1,797", "1,899", "loss", "−5.4%"),
        ("gpt-oss-120b, 1 to 2.5 req/s arrivals", "582 to 1,327", "577 to 1,291", "tie", "tie to +2.1%"),
    ]),
    ("1,000 requests at once, gpt-oss-20b", "output tok/s", [
        ("same concurrency (64)", "7,309", "5,923", "win", "+23.4%"),
        ("each at its max (256 vs 1,024)", "12,682", "22,482", "loss", "−43.6%"),
    ]),
    ("Accuracy, low reasoning effort", "OpenAI published in brackets", [
        ("gpt-oss-20b, GPQA / AIME 2025", "56.3 / 36.3%", "56.8 / 37.9%", "match", "matches (56.8 / 37.1)"),
        ("gpt-oss-120b, GPQA / AIME 2025", "66.8 / 50.8%", "65.5 / 51.7%", "match", "matches (67.1 / 50.4)"),
    ]),
]

W, PAD = 900, 28
COL = [PAD, 410, 560, 610]          # label, simplellm (right edge), vllm (right edge), pill (left)
ROW_H, GROUP_H, HEAD_H = 34, 46, 92

rows = []
y = HEAD_H
for title, unit, items in GROUPS:
    y += 10
    rows.append(("group", y, title, unit))
    y += GROUP_H - 10
    for it in items:
        rows.append(("row", y, it))
        y += ROW_H
H = y + 50

def text(x, y, s, cls, anchor="start"):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(s)}</text>'

out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
       f'aria-label="SimpleLLM vs tuned vLLM 0.31.0 on one H100: faster at decode in every batch and context tested, '
       f'slower at time to first token, 120b saturated serving and 1,000-way concurrency; accuracy matches OpenAI.">']
out.append("""<style>
  svg { --bg:#fcfcfb; --border:#e4e3df; --ink:#0b0b0b; --ink2:#52514e; --ink3:#8a8984; --rule:#efeeea; --band:#f5f4f1;
        --win-bg:#e3f4e3; --win-ink:#0a6b0a; --loss-bg:#fbe5e5; --loss-ink:#a82828; --tie-bg:#efeeea; --tie-ink:#52514e;
        --us:#2a78d6; --best:rgba(42,120,214,0.09); }
  @media (prefers-color-scheme: dark) {
    svg { --bg:#1a1a19; --border:#33332f; --ink:#ffffff; --ink2:#c3c2b7; --ink3:#8f8e86; --rule:#2a2a27; --band:#222220;
          --win-bg:#173a17; --win-ink:#7fd67f; --loss-bg:#432020; --loss-ink:#f19a9a; --tie-bg:#2e2e2b; --tie-ink:#c3c2b7;
          --us:#5598e7; --best:rgba(85,152,231,0.16); }
  }
  text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: var(--ink); }
  .title { font-size: 20px; font-weight: 650; }
  .sub { font-size: 13px; fill: var(--ink2); }
  .colh { font-size: 12px; font-weight: 600; fill: var(--ink2); letter-spacing: .04em; text-transform: uppercase; }
  .us { fill: var(--us); }
  .group { font-size: 14px; font-weight: 650; }
  .unit { font-size: 12px; fill: var(--ink3); }
  .label { font-size: 14px; fill: var(--ink2); }
  .num { font-size: 14px; font-variant-numeric: tabular-nums; }
  .numb { font-size: 14px; font-weight: 650; font-variant-numeric: tabular-nums; }
  .foot { font-size: 12px; fill: var(--ink3); }
  .pill { font-size: 13px; font-weight: 650; }
</style>""")
out.append(f'<rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="12" fill="var(--bg)" stroke="var(--border)"/>')
out.append(text(PAD, 40, "SimpleLLM vs tuned vLLM 0.31.0", "title"))
out.append(text(PAD, 62, "One NVIDIA H100 80GB, official MXFP4 checkpoints, greedy decoding, median of 3 runs in separate processes", "sub"))
out.append(text(COL[1], 86, "SimpleLLM", "colh us", "end"))
out.append(text(COL[2], 86, "vLLM", "colh", "end"))
out.append(text(COL[3], 86, "Difference", "colh"))

PILL = {"win": ("win-bg", "win-ink", "▲ "), "loss": ("loss-bg", "loss-ink", "▼ "),
        "tie": ("tie-bg", "tie-ink", "= "), "match": ("win-bg", "win-ink", "✓ ")}
for r in rows:
    if r[0] == "group":
        _, gy, title, unit = r
        out.append(f'<line x1="{PAD}" x2="{W-PAD}" y1="{gy-4}" y2="{gy-4}" stroke="var(--border)"/>')
        out.append(text(PAD, gy + 22, title, "group"))
        out.append(text(W - PAD, gy + 22, unit, "unit", "end"))
    else:
        _, ry, (label, us, them, verdict, pill) = r
        base = ry + 22
        out.append(text(PAD + 12, base, label, "label"))
        best = {"win": (290, 420), "loss": (440, 570)}.get(verdict)  # full-cell tint behind the better number; runs read as a column
        if best: out.append(f'<rect x="{best[0]}" y="{ry}" width="{best[1] - best[0]}" height="{ROW_H}" fill="var(--best)"/>')
        out.append(text(COL[1], base, us, "numb" if verdict == "win" else "num", "end"))
        out.append(text(COL[2], base, them, "numb" if verdict == "loss" else "num", "end"))
        bg, ink, icon = PILL[verdict]
        s = icon + pill
        pw = 16 + 7.6 * len(s)
        out.append(f'<rect x="{COL[3]}" y="{ry+6}" width="{pw:.0f}" height="23" rx="11.5" fill="var(--{bg})"/>')
        out.append(f'<text x="{COL[3]+8}" y="{base}" class="pill" style="fill:var(--{ink})">{escape(s)}</text>')
for x, s_ in ((PAD, "\u25B2 SimpleLLM faster"), (PAD + 150, "\u25BC vLLM faster"), (PAD + 270, "= within run-to-run noise (2%)"),
              (PAD + 470, "\u2713 inside the published 95% interval   \u2022 blue = better number")):
    out.append(text(x, H - 22, s_, "foot"))
out.append("</svg>")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
print(W, H)
