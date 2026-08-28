"""End-to-end processing pipeline.

Orchestrates: render → restore → OCR → chapters → synopsis → references
→ insights → analytics → translations → charts → HTML report, and writes
a fully organized output folder:

    <output>/
      cleaned/<book>_cleaned.pdf         searchable, restored PDF
      images/original/page_XXXX.png      page scans as rendered
      images/cleaned/page_XXXX.png       restored page images
      images/embedded/…                  embedded images, original format
      text/full_text.txt                 corrected OCR text
      text/chapters/chapter_XX.txt
      translations/<lang>/full_text_<lang>.txt
      synopsis/synopsis.md
      insights/insights.md
      references/{esoteric,theological,scientific,syncretic}.md + .json
      analytics/analytics.json + charts/*.png
      report.html                        self-contained illustrated report
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import shutil
import traceback
from typing import Callable

from . import __version__
from .analytics import BookAnalytics, compute_analytics
from .chapters import Chapter, detect_chapters
from .charts import render_all_charts
from .imaging import CleanResult, clean_page
from .insights import InsightReport, build_insights
from .ocr import PageOCR, ocr_page
from .pdfio import build_image_pdf, build_searchable_pdf, extract_embedded_images, open_pdf, render_page, save_image_png
from .references import ReferenceReport, scan_references
from .synopsis import ChapterSynopsis, build_synopses
from .textutils import normalize_ocr_text
from .translate import SUPPORTED_LANGUAGES, TranslationResult, translate_text

ProgressFn = Callable[[int, str], None]  # (percent 0-100, message)


@dataclass
class PipelineOptions:
    input_pdf: str
    output_dir: str
    dpi: int = 250
    ocr_language: str = "eng"
    translate_to: list[str] = field(default_factory=list)
    source_language: str = "auto"  # source language for translation
    deskew: bool = True
    max_pages: int | None = None  # None = all pages
    synopsis_sentences: int = 5
    export_original_images: bool = True
    export_cleaned_images: bool = True
    extract_embedded: bool = True


@dataclass
class PipelineResult:
    options: PipelineOptions
    book_name: str = ""
    output_dir: Path | None = None
    page_texts: list[str] = field(default_factory=list)
    full_text: str = ""
    chapters: list[Chapter] = field(default_factory=list)
    synopses: list[ChapterSynopsis] = field(default_factory=list)
    references: ReferenceReport | None = None
    insights: InsightReport | None = None
    analytics: BookAnalytics | None = None
    translations: list[TranslationResult] = field(default_factory=list)
    chart_paths: dict[str, Path] = field(default_factory=dict)
    cleaned_pdf: Path | None = None
    report_html: Path | None = None
    original_image_paths: list[Path] = field(default_factory=list)
    cleaned_image_paths: list[Path] = field(default_factory=list)
    embedded_image_paths: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class PipelineCancelled(Exception):
    pass


def run_pipeline(
    options: PipelineOptions, progress: ProgressFn | None = None, cancel_check: Callable[[], bool] | None = None
) -> PipelineResult:
    """Execute the full pipeline. Raises PipelineCancelled if cancelled."""
    started = datetime.now()

    def report(pct: int, msg: str) -> None:
        if cancel_check and cancel_check():
            raise PipelineCancelled()
        if progress:
            progress(pct, msg)

    result = PipelineResult(options=options)
    input_path = Path(options.input_pdf)
    result.book_name = input_path.stem
    out = Path(options.output_dir) / f"{input_path.stem}_arcanum"
    out.mkdir(parents=True, exist_ok=True)
    result.output_dir = out

    report(1, "Opening document…")
    doc = open_pdf(input_path)
    total_pages = doc.page_count
    if options.max_pages:
        total_pages = min(total_pages, options.max_pages)

    # ------------------------------------------------------------------
    # Stage 1 — render, restore, OCR (60% of the progress budget)
    # ------------------------------------------------------------------
    clean_results: list[CleanResult] = []
    ocr_pages: list[PageOCR] = []
    page_pdf_parts: list[bytes] = []
    cleaned_images = []

    for i in range(total_pages):
        pct = 2 + int(58 * i / max(1, total_pages))
        report(pct, f"Restoring & reading page {i + 1}/{total_pages}…")

        original = render_page(doc, i, dpi=options.dpi)
        if options.export_original_images:
            path = out / "images" / "original" / f"page_{i + 1:04d}.png"
            save_image_png(original, path)
            result.original_image_paths.append(path)

        cleaned = clean_page(original, deskew=options.deskew)
        clean_results.append(cleaned)
        cleaned_images.append(cleaned.image)
        if options.export_cleaned_images:
            path = out / "images" / "cleaned" / f"page_{i + 1:04d}.png"
            save_image_png(cleaned.image, path)
            result.cleaned_image_paths.append(path)

        page_ocr = ocr_page(cleaned.image, lang=options.ocr_language, make_pdf=True)
        ocr_pages.append(page_ocr)
        if page_ocr.searchable_pdf:
            page_pdf_parts.append(page_ocr.searchable_pdf)

    doc.close()
    result.page_texts = [normalize_ocr_text(p.text) for p in ocr_pages]
    result.full_text = "\n\n".join(result.page_texts)

    # ------------------------------------------------------------------
    # Stage 2 — cleaned searchable PDF + embedded images
    # ------------------------------------------------------------------
    report(62, "Assembling cleaned, searchable PDF…")
    cleaned_pdf = out / "cleaned" / f"{input_path.stem}_cleaned.pdf"
    try:
        if page_pdf_parts:
            build_searchable_pdf(page_pdf_parts, cleaned_pdf)
        else:
            build_image_pdf(cleaned_images, cleaned_pdf, dpi=options.dpi)
        result.cleaned_pdf = cleaned_pdf
    except Exception as exc:
        result.warnings.append(f"Cleaned PDF assembly failed: {exc}")

    if options.extract_embedded:
        report(66, "Extracting embedded images (original format)…")
        try:
            result.embedded_image_paths = extract_embedded_images(input_path, out / "images" / "embedded")
        except Exception as exc:
            result.warnings.append(f"Embedded image extraction failed: {exc}")

    # Keep a copy of the source next to the outputs for provenance.
    try:
        shutil.copy2(input_path, out / f"{input_path.name}")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Stage 3 — text intelligence
    # ------------------------------------------------------------------
    report(70, "Detecting chapters…")
    result.chapters = detect_chapters(result.page_texts)

    report(73, "Writing corrected text…")
    text_dir = out / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir / "full_text.txt").write_text(result.full_text, encoding="utf-8")
    chapter_dir = text_dir / "chapters"
    chapter_dir.mkdir(exist_ok=True)
    for chapter in result.chapters:
        name = f"chapter_{chapter.number:02d}.txt"
        (chapter_dir / name).write_text(
            f"{chapter.title}\n(pages {chapter.start_page}-" f"{chapter.end_page})\n\n{chapter.text}", encoding="utf-8"
        )

    report(76, "Composing chapter synopses…")
    result.synopses = build_synopses(result.chapters, options.synopsis_sentences)
    _write_synopsis_md(result, out / "synopsis" / "synopsis.md")

    report(80, "Mining esoteric, theological & scientific references…")
    result.references = scan_references(result.chapters)
    _write_reference_files(result.references, out / "references")

    report(84, "Computing analytics…")
    result.analytics = compute_analytics(
        result.page_texts, ocr_pages, clean_results, result.chapters, result.references
    )
    analytics_dir = out / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    (analytics_dir / "analytics.json").write_text(
        json.dumps(result.analytics.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report(86, "Distilling insights…")
    result.insights = build_insights(result.full_text, result.chapters, result.analytics, result.references)
    _write_insights_md(result, out / "insights" / "insights.md")

    # ------------------------------------------------------------------
    # Stage 4 — translations
    # ------------------------------------------------------------------
    targets = [t for t in options.translate_to if t]
    for index, target in enumerate(targets):
        label = SUPPORTED_LANGUAGES.get(target, target)
        report(88 + int(6 * index / max(1, len(targets))), f"Translating into {label}…")
        translation = translate_text(result.full_text, target, source=options.source_language)
        if translation.ok:
            lang_dir = out / "translations" / target
            lang_dir.mkdir(parents=True, exist_ok=True)
            (lang_dir / f"full_text_{target}.txt").write_text(translation.text, encoding="utf-8")
            summaries = {}
            for synopsis in result.synopses:
                translated = translate_text(synopsis.summary, target, source=options.source_language)
                if translated.ok:
                    key = f"{synopsis.chapter_number:02d} {synopsis.chapter_title}"
                    summaries[key] = translated.text
            translation.chapter_summaries = summaries
            if summaries:
                lines = [f"# Chapter Synopses — {label}", ""]
                for key, text in summaries.items():
                    lines += [f"## {key}", "", text, ""]
                (lang_dir / f"synopses_{target}.md").write_text("\n".join(lines), encoding="utf-8")
        else:
            result.warnings.append(f"Translation to {label} failed: {translation.error}")
        result.translations.append(translation)

    # ------------------------------------------------------------------
    # Stage 5 — charts + report
    # ------------------------------------------------------------------
    report(95, "Rendering analytics charts…")
    try:
        result.chart_paths = render_all_charts(result.analytics, analytics_dir / "charts")
    except Exception as exc:
        result.warnings.append(f"Chart rendering failed: {exc}\n" f"{traceback.format_exc(limit=2)}")

    report(98, "Building illustrated report…")
    try:
        from .report import build_html_report

        result.report_html = build_html_report(result, out / "report.html")
    except Exception as exc:
        result.warnings.append(f"Report generation failed: {exc}\n" f"{traceback.format_exc(limit=2)}")

    result.elapsed_seconds = (datetime.now() - started).total_seconds()
    report(100, "Complete.")
    return result


# ----------------------------------------------------------------------
# Markdown writers
# ----------------------------------------------------------------------


def _write_synopsis_md(result: PipelineResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {result.book_name} — Synopsis by Chapter",
        "",
        f"_Generated by Arcanum v{__version__} on " f"{datetime.now():%Y-%m-%d %H:%M}_",
        "",
    ]
    for s in result.synopses:
        lines += [
            f"## {s.chapter_number:02d}. {s.chapter_title} "
            f"(pages {s.start_page}–{s.end_page}, "
            f"{s.word_count:,} words)",
            "",
            s.summary or "_No readable text on these pages._",
            "",
        ]
        if s.keywords:
            lines += [f"**Keywords:** {', '.join(s.keywords)}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_reference_files(references: ReferenceReport, ref_dir: Path) -> None:
    ref_dir.mkdir(parents=True, exist_ok=True)
    for category, hits in references.hits.items():
        lines = [f"# {category.title()} References", "", f"{len(hits)} passages found.", ""]
        for hit in hits:
            lines += [
                f"### Chapter {hit.chapter_number} — {hit.chapter_title} " f"(pages {hit.start_page}–{hit.end_page})",
                "",
                f"> {hit.excerpt}",
                "",
                f"**Terms:** {', '.join(hit.terms)}  |  " f"**Relevance:** {hit.score:.1f}",
                "",
            ]
        (ref_dir / f"{category}.md").write_text("\n".join(lines), encoding="utf-8")
        (ref_dir / f"{category}.json").write_text(
            json.dumps([h.to_dict() for h in hits], indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if references.syncretic:
        lines = ["# Syncretic Passages", "", "Passages weaving together two or more traditions.", ""]
        for hit in references.syncretic:
            lines += [
                f"### {hit.category} — Chapter {hit.chapter_number}",
                "",
                f"> {hit.excerpt}",
                "",
                f"**Terms:** {', '.join(hit.terms)}",
                "",
            ]
        (ref_dir / "syncretic.md").write_text("\n".join(lines), encoding="utf-8")
        (ref_dir / "syncretic.json").write_text(
            json.dumps([h.to_dict() for h in references.syncretic], indent=2, ensure_ascii=False), encoding="utf-8"
        )


def _write_insights_md(result: PipelineResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    insights = result.insights
    lines = [f"# {result.book_name} — Insights", ""]
    for kind, heading in (
        ("thematic", "Thematic"),
        ("textual", "Textual"),
        ("general", "Structural"),
        ("quality", "Scan Quality"),
    ):
        block = [i for i in insights.insights if i.kind == kind]
        if not block:
            continue
        lines += [f"## {heading}", ""]
        for insight in block:
            lines += [f"### {insight.title}", "", insight.detail, ""]
    if insights.notable_quotes:
        lines += ["## Notable Passages", ""]
        for quote in insights.notable_quotes:
            lines += [f"> {quote}", ""]
    if insights.recurring_figures:
        lines += ["## Recurring Names & Figures", ""]
        for name, count in insights.recurring_figures:
            lines += [f"- {name} — {count} mentions"]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
