"""Self-contained illustrated HTML report.

One file, no external assets: charts are embedded as base64 PNGs and the
styling is inlined, so the report can be mailed or archived as-is.
"""

from __future__ import annotations

import base64
from datetime import datetime
import html
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__

if TYPE_CHECKING:
    from .pipeline import PipelineResult

_CATEGORY_BADGES = {"esoteric": "#a98fd6", "theological": "#d4af6a", "scientific": "#5ec8b8"}

_CSS = """
:root{--bg:#0e1017;--panel:#171a24;--panel2:#1e2230;--ink:#e8e2d4;
--muted:#8b8fa3;--gold:#d4af6a;--teal:#5ec8b8;--violet:#a98fd6;
--rose:#d67f8f;--line:#2a2e3e}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);
font-family:Georgia,'Times New Roman',serif;line-height:1.65}
.wrap{max-width:1080px;margin:0 auto;padding:48px 28px 80px}
header.hero{text-align:center;padding:64px 20px 44px;
border-bottom:1px solid var(--line);margin-bottom:44px}
.hero .mark{font-size:13px;letter-spacing:.42em;color:var(--gold);
text-transform:uppercase;margin-bottom:18px}
.hero h1{font-size:44px;font-weight:normal;letter-spacing:.02em}
.hero .sub{color:var(--muted);margin-top:14px;font-size:15px}
.hero .rule{width:72px;height:2px;background:var(--gold);
margin:26px auto 0}
h2{font-size:24px;font-weight:normal;color:var(--gold);
margin:52px 0 18px;padding-bottom:10px;border-bottom:1px solid var(--line)}
h3{font-size:17px;margin:22px 0 8px;color:var(--ink)}
p{margin:0 0 12px}
.grid{display:grid;gap:16px}
.cards{grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);
border-radius:10px;padding:18px 20px}
.card .num{font-size:26px;color:var(--gold);font-variant-numeric:tabular-nums}
.card .lbl{font-size:11px;color:var(--muted);text-transform:uppercase;
letter-spacing:.14em;margin-top:4px}
figure{background:var(--panel);border:1px solid var(--line);
border-radius:10px;padding:14px;margin:18px 0}
figure img{width:100%;border-radius:6px;display:block}
.charts2{grid-template-columns:1fr 1fr}
@media(max-width:820px){.charts2{grid-template-columns:1fr}}
blockquote{background:var(--panel2);border-left:3px solid var(--gold);
padding:14px 18px;margin:12px 0;border-radius:0 8px 8px 0;
font-style:italic;color:#d6d0c2}
.badge{display:inline-block;font-size:11px;letter-spacing:.08em;
text-transform:uppercase;padding:3px 10px;border-radius:20px;
color:#0e1017;font-family:Helvetica,Arial,sans-serif;font-weight:bold}
.meta{color:var(--muted);font-size:13px}
.terms{color:var(--teal);font-size:13px}
.insight{background:var(--panel);border:1px solid var(--line);
border-radius:10px;padding:18px 22px;margin:12px 0}
.insight h3{margin-top:0;color:var(--gold)}
.chapter{background:var(--panel);border:1px solid var(--line);
border-radius:10px;padding:20px 24px;margin:14px 0}
.kw{color:var(--teal);font-size:13px}
table{width:100%;border-collapse:collapse;margin:14px 0;font-size:14px}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-size:11px;text-transform:uppercase;
letter-spacing:.12em;font-family:Helvetica,Arial,sans-serif}
footer{margin-top:70px;text-align:center;color:var(--muted);
font-size:13px;border-top:1px solid var(--line);padding-top:26px}
.warn{background:#2a1e1e;border:1px solid #55393c;border-radius:8px;
padding:12px 16px;color:#d8a7a7;font-size:14px;margin:10px 0}
"""


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _fig(chart_paths: dict[str, Path], key: str, caption: str) -> str:
    path = chart_paths.get(key)
    if not path or not Path(path).exists():
        return ""
    return f'<figure><img alt="{html.escape(caption)}" ' f'src="data:image/png;base64,{_b64(Path(path))}"/></figure>'


def _esc(text: str) -> str:
    return html.escape(text or "")


def build_html_report(result: "PipelineResult", out_path: str | Path) -> Path:
    a = result.analytics
    refs = result.references
    parts: list[str] = []
    add = parts.append

    add(f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(result.book_name)} — Arcanum Report</title>
<style>{_CSS}</style></head><body><div class="wrap">
<header class="hero">
<div class="mark">Arcanum · Antique Book Intelligence</div>
<h1>{_esc(result.book_name)}</h1>
<div class="sub">Restoration, analysis &amp; reference survey ·
{datetime.now():%B %d, %Y}</div>
<div class="rule"></div></header>""")

    # ---- Overview cards ------------------------------------------------
    if a:
        cards = [
            (f"{a.page_count:,}", "Pages"),
            (f"{a.word_count:,}", "Words"),
            (f"{len(result.chapters)}", "Chapters"),
            (f"{a.mean_ocr_confidence:.0f}%", "OCR confidence"),
            (f"{sum(a.category_counts.values()):,}", "Reference passages"),
            (f"{a.flesch_score:.0f}", "Reading ease"),
        ]
        add('<section><div class="grid cards">')
        for num, label in cards:
            add(f'<div class="card"><div class="num">{num}</div>' f'<div class="lbl">{label}</div></div>')
        add("</div></section>")

    if result.warnings:
        add("<section>")
        for warning in result.warnings:
            add(f'<div class="warn">{_esc(warning.splitlines()[0])}</div>')
        add("</section>")

    # ---- Analytics charts ----------------------------------------------
    add("<h2>Analytics</h2>")
    add(_fig(result.chart_paths, "ocr_confidence", "OCR confidence"))
    add('<div class="grid charts2">')
    add(_fig(result.chart_paths, "category_donut", "Category distribution"))
    add(_fig(result.chart_paths, "chapter_words", "Words per chapter"))
    add("</div>")
    add(_fig(result.chart_paths, "density_heatmap", "Reference density"))
    add('<div class="grid charts2">')
    add(_fig(result.chart_paths, "top_words", "Top words"))
    add(_fig(result.chart_paths, "restoration", "Restoration quality"))
    add("</div>")

    # ---- Insights --------------------------------------------------------
    if result.insights and result.insights.insights:
        add("<h2>Insights</h2>")
        for insight in result.insights.insights:
            add(f'<div class="insight"><h3>{_esc(insight.title)}</h3>' f"<p>{_esc(insight.detail)}</p></div>")
        if result.insights.notable_quotes:
            add("<h3>Notable passages</h3>")
            for quote in result.insights.notable_quotes[:5]:
                add(f"<blockquote>{_esc(quote)}</blockquote>")

    # ---- Synopsis ----------------------------------------------------------
    if result.synopses:
        add("<h2>Synopsis by Chapter</h2>")
        for s in result.synopses:
            add(f"""<div class="chapter">
<h3>{s.chapter_number:02d}. {_esc(s.chapter_title)}</h3>
<div class="meta">Pages {s.start_page}–{s.end_page} ·
{s.word_count:,} words</div>
<p style="margin-top:10px">{_esc(s.summary) or "<i>No readable text.</i>"}</p>
<div class="kw">Keywords: {_esc(", ".join(s.keywords))}</div></div>""")

    # ---- References ---------------------------------------------------------
    if refs:
        add("<h2>Reference Survey</h2>")
        for category in ("esoteric", "theological", "scientific"):
            hits = refs.hits.get(category, [])
            color = _CATEGORY_BADGES[category]
            add(
                f'<h3><span class="badge" style="background:{color}">'
                f"{category}</span> &nbsp;{len(hits)} passages</h3>"
            )
            for hit in hits[:10]:
                add(f"""<blockquote>{_esc(hit.excerpt)}
<div class="meta" style="margin-top:8px;font-style:normal">
Chapter {hit.chapter_number} — {_esc(hit.chapter_title)},
pages {hit.start_page}–{hit.end_page} ·
<span class="terms">{_esc(", ".join(hit.terms))}</span></div>
</blockquote>""")
            if len(hits) > 10:
                add(f'<p class="meta">…and {len(hits) - 10} more in ' f"references/{category}.md</p>")

        if refs.syncretic:
            add(
                '<h3><span class="badge" style="background:#d67f8f">'
                "syncretic</span> &nbsp;passages blending traditions</h3>"
            )
            for hit in refs.syncretic[:6]:
                add(f"""<blockquote>{_esc(hit.excerpt)}
<div class="meta" style="margin-top:8px;font-style:normal">
{_esc(hit.category)} · Chapter {hit.chapter_number} ·
<span class="terms">{_esc(", ".join(hit.terms))}</span></div>
</blockquote>""")

    # ---- Translations ----------------------------------------------------
    if result.translations:
        add("<h2>Translations</h2><table><tr><th>Language</th>" "<th>Status</th><th>Output</th></tr>")
        for t in result.translations:
            status = "✓ complete" if t.ok else f"failed — {_esc(t.error)}"
            output = f"translations/{t.language}/full_text_{t.language}.txt" if t.ok else "—"
            add(f"<tr><td>{_esc(t.language_name)}</td><td>{status}</td>" f"<td>{output}</td></tr>")
        add("</table>")

    add(f"""<footer>Generated by Arcanum v{__version__} ·
{datetime.now():%Y-%m-%d %H:%M} ·
processing time {result.elapsed_seconds:.0f}s</footer>
</div></body></html>""")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts), encoding="utf-8")
    return out_path
