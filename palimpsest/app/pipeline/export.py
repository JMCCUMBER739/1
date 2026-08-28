"""Export cleaned PDFs, reports, manifests, and structured JSON/CSV."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from app.pipeline.models import JobResult, ReferenceKind


def images_to_pdf(image_paths: list[Path], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    try:
        for img_path in image_paths:
            img = pymupdf.open(img_path)
            rect = img[0].rect
            pdf_bytes = img.convert_to_pdf()
            img.close()
            page_pdf = pymupdf.open("pdf", pdf_bytes)
            page = doc.new_page(width=rect.width, height=rect.height)
            page.show_pdf_page(page.rect, page_pdf, 0)
            page_pdf.close()
        doc.save(out_path)
    finally:
        doc.close()
    return out_path


def text_pages_to_pdf(pages: list[str], out_path: Path, title: str = "Translated Text") -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=11,
        leading=15,
        spaceAfter=8,
    )
    heading = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading1"],
        fontName="Times-Bold",
        fontSize=16,
        spaceAfter=12,
    )
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
        title=title,
    )
    story: list[Any] = [Paragraph(title, heading), Spacer(1, 0.2 * inch)]
    for i, text in enumerate(pages):
        story.append(Paragraph(f"<b>Page {i + 1}</b>", styles["Heading3"]))
        safe = (text or "(empty)").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe = safe.replace("\n", "<br/>")
        story.append(Paragraph(safe, body))
        story.append(Spacer(1, 0.15 * inch))
    doc.build(story)
    return out_path


def _serialize_result(result: JobResult) -> dict[str, Any]:
    def pathify(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, ReferenceKind):
            return obj.value
        if hasattr(obj, "__dataclass_fields__"):
            return {k: pathify(v) for k, v in asdict(obj).items()}
        if isinstance(obj, list):
            return [pathify(x) for x in obj]
        if isinstance(obj, dict):
            return {k: pathify(v) for k, v in obj.items()}
        return obj

    return pathify(result)


def write_manifest(result: JobResult) -> Path:
    path = result.output_dir / "manifest.json"
    data = _serialize_result(result)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def write_references_csv(result: JobResult) -> Path:
    path = result.output_dir / "references" / "all_references.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["kind", "category", "term", "page", "chapter", "excerpt", "score"],
        )
        writer.writeheader()
        for r in result.references:
            writer.writerow(
                {
                    "kind": r.kind.value,
                    "category": r.category,
                    "term": r.term,
                    "page": r.page_index + 1,
                    "chapter": r.chapter_title or "",
                    "excerpt": r.excerpt,
                    "score": r.context_score,
                }
            )
    # Split by kind
    for kind in ReferenceKind:
        kind_path = result.output_dir / "references" / f"{kind.value}_references.csv"
        subset = [r for r in result.references if r.kind == kind]
        with open(kind_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["category", "term", "page", "chapter", "excerpt"],
            )
            writer.writeheader()
            for r in subset:
                writer.writerow(
                    {
                        "category": r.category,
                        "term": r.term,
                        "page": r.page_index + 1,
                        "chapter": r.chapter_title or "",
                        "excerpt": r.excerpt,
                    }
                )
    return path


def write_chapter_synopses(result: JobResult) -> Path:
    path = result.output_dir / "synopses" / "chapters.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Chapter Synopses — {result.source_pdf.name}", ""]
    for ch in result.chapters:
        lines.append(f"## {ch.title}")
        lines.append(f"*Pages {ch.start_page + 1}–{ch.end_page + 1} · {ch.word_count} words*")
        lines.append("")
        lines.append(ch.synopsis)
        lines.append("")
        lines.append("### Insights")
        for insight in ch.insights:
            lines.append(f"- {insight}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

    json_path = result.output_dir / "synopses" / "chapters.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "index": c.index,
                    "title": c.title,
                    "start_page": c.start_page + 1,
                    "end_page": c.end_page + 1,
                    "word_count": c.word_count,
                    "synopsis": c.synopsis,
                    "insights": c.insights,
                }
                for c in result.chapters
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def write_analytics_json(result: JobResult) -> Path:
    path = result.output_dir / "analytics" / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result.analytics), indent=2), encoding="utf-8")
    return path


def write_full_text(result: JobResult) -> Path:
    text_dir = result.output_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    cleaned = text_dir / "cleaned_full.txt"
    translated = text_dir / "translated_full.txt"
    cleaned.write_text(
        "\n\n".join(f"----- PAGE {t.page_index + 1} -----\n{t.cleaned_text}" for t in result.texts),
        encoding="utf-8",
    )
    translated.write_text(
        "\n\n".join(
            f"----- PAGE {t.page_index + 1} -----\n{t.translated_text or t.cleaned_text}"
            for t in result.texts
        ),
        encoding="utf-8",
    )
    return cleaned


def write_html_report(result: JobResult) -> Path:
    path = result.output_dir / "report.html"
    a = result.analytics
    ref_rows = "".join(
        f"<tr><td>{r.kind.value}</td><td>{r.category}</td><td><b>{r.term}</b></td>"
        f"<td>{r.page_index + 1}</td><td>{r.excerpt}</td></tr>"
        for r in result.references[:200]
    )
    chapter_blocks = "".join(
        f"<section><h3>{c.title}</h3><p class='meta'>Pages {c.start_page + 1}–{c.end_page + 1}</p>"
        f"<p>{c.synopsis}</p><ul>{''.join(f'<li>{i}</li>' for i in c.insights)}</ul></section>"
        for c in result.chapters
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Palimpsest Report — {result.source_pdf.name}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;700&family=Source+Sans+3:wght@400;600&display=swap');
  :root {{ --ink:#1A1612; --paper:#F7F1E8; --accent:#C45C26; --slate:#2C3E50; --sage:#4A6B5C; }}
  body {{ margin:0; font-family:'Source Sans 3',sans-serif; background:linear-gradient(160deg,#EDE4D7,#F7F1E8 40%,#E8EFEA); color:var(--ink); }}
  header {{ padding:48px 8vw 24px; background:radial-gradient(ellipse at 20% 0%, rgba(196,92,38,.18), transparent 50%), linear-gradient(120deg,#1A1612,#2C3E50); color:#F7F1E8; }}
  h1,h2,h3 {{ font-family:'Cormorant Garamond',serif; letter-spacing:.02em; }}
  h1 {{ font-size:3rem; margin:0 0 .25rem; }}
  .brand {{ text-transform:uppercase; letter-spacing:.28em; font-size:.75rem; color:#E8C4A8; }}
  main {{ padding:32px 8vw 64px; max-width:1100px; margin:0 auto; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:16px; margin:24px 0; }}
  .metric {{ background:rgba(255,255,255,.55); border-left:3px solid var(--accent); padding:16px 18px; }}
  .metric b {{ display:block; font-size:1.6rem; font-family:'Cormorant Garamond',serif; }}
  table {{ width:100%; border-collapse:collapse; font-size:.92rem; background:rgba(255,255,255,.5); }}
  th,td {{ border-bottom:1px solid rgba(26,22,18,.12); padding:10px 12px; text-align:left; vertical-align:top; }}
  th {{ background:rgba(44,62,80,.08); }}
  section {{ margin:28px 0; }}
  .meta {{ color:var(--slate); font-size:.9rem; }}
</style>
</head>
<body>
<header>
  <div class="brand">Palimpsest</div>
  <h1>Restoration Report</h1>
  <p>{result.source_pdf.name} · {result.created_at.strftime('%Y-%m-%d %H:%M')}</p>
</header>
<main>
  <div class="grid">
    <div class="metric"><span>Pages</span><b>{a.page_count}</b></div>
    <div class="metric"><span>Words</span><b>{a.word_count:,}</b></div>
    <div class="metric"><span>Chapters</span><b>{a.chapter_count}</b></div>
    <div class="metric"><span>Readability</span><b>{a.readability_score}</b></div>
    <div class="metric"><span>Lexical Diversity</span><b>{a.lexical_diversity}</b></div>
    <div class="metric"><span>References</span><b>{sum(a.reference_counts.values())}</b></div>
  </div>
  <h2>Chapter Synopses</h2>
  {chapter_blocks}
  <h2>Reference Index</h2>
  <table>
    <thead><tr><th>Kind</th><th>Category</th><th>Term</th><th>Page</th><th>Excerpt</th></tr></thead>
    <tbody>{ref_rows or '<tr><td colspan="5">No references matched.</td></tr>'}</tbody>
  </table>
</main>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")
    return path


def export_all(result: JobResult) -> JobResult:
    cleaned_imgs = [p.cleaned_path for p in result.pages if p.cleaned_path]
    if cleaned_imgs:
        result.cleaned_pdf = images_to_pdf(cleaned_imgs, result.output_dir / "pdf" / "cleaned.pdf")

    translated_pages = [t.translated_text or t.cleaned_text for t in result.texts]
    result.translated_pdf = text_pages_to_pdf(
        translated_pages,
        result.output_dir / "pdf" / "translated.pdf",
        title=f"Translated — {result.source_pdf.stem}",
    )

    write_full_text(result)
    write_chapter_synopses(result)
    write_references_csv(result)
    write_analytics_json(result)
    result.report_html = write_html_report(result)
    write_manifest(result)
    return result
