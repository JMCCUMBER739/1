"""End-to-end orchestration: scanned PDF in, restored/analyzed book out.

The pipeline is designed to be driven from a GUI: every stage reports
progress through an optional ``progress_callback(fraction: float, message:
str)`` so a Streamlit progress bar (or a CLI, or tests) can observe it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image

from core import analytics, insights as insights_mod, ocr, pdf_io, references as references_mod
from core.chapters import Chapter, PageText, split_into_chapters
from core.config import PipelineOptions
from core.image_cleaning import clean_page_image
from core.language import detect_language, language_name, to_tesseract_code
from core.report import build_html_report, build_pdf_report
from core.summarization import summarize_chapter
from core.translation import TranslationResult, translate_text
from core.utils import ensure_dir, safe_page_number, slugify, write_json
from core.visualizations import (
    chapter_length_bar,
    entity_frequency_bar,
    generate_wordcloud_image,
    language_distribution_pie,
    noise_reduction_bar,
    readability_gauge,
    reference_category_pie,
    sentiment_arc_line,
    word_frequency_bar,
)

ProgressCallback = Optional[Callable[[float, str], None]]


@dataclass
class PipelineResult:
    book_title: str
    output_dir: Path
    page_count: int
    original_image_paths: List[Path] = field(default_factory=list)
    cleaned_image_paths: List[Path] = field(default_factory=list)
    cleaned_pdf_path: Optional[Path] = None
    page_texts: List[str] = field(default_factory=list)
    page_languages: List[Optional[str]] = field(default_factory=list)
    detected_primary_language: Optional[str] = None
    full_text: str = ""
    translation: Optional[TranslationResult] = None
    chapters: List[Chapter] = field(default_factory=list)
    references_by_category: Dict[str, list] = field(default_factory=dict)
    reference_stats: Dict[str, dict] = field(default_factory=dict)
    stats: Optional[analytics.TextStatistics] = None
    sentiment_by_chapter: List[dict] = field(default_factory=list)
    word_frequency: List[tuple] = field(default_factory=list)
    image_quality: Optional[analytics.ImageQualityStats] = None
    cleaning_metrics: List[dict] = field(default_factory=list)
    insights: Optional[insights_mod.BookInsights] = None
    html_report_path: Optional[Path] = None
    pdf_report_path: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)
    manifest_path: Optional[Path] = None


def _report_progress(cb: ProgressCallback, fraction: float, message: str) -> None:
    if cb:
        cb(min(fraction, 1.0), message)


class BookRestorationPipeline:
    def run(
        self,
        pdf_path: Path,
        options: PipelineOptions,
        progress_callback: ProgressCallback = None,
    ) -> PipelineResult:
        pdf_path = Path(pdf_path)
        book_title = options.book_title or pdf_path.stem
        slug = slugify(book_title)
        book_output_dir = ensure_dir(Path(options.output_dir) / slug)

        images_original_dir = ensure_dir(book_output_dir / "images" / "original")
        images_cleaned_dir = ensure_dir(book_output_dir / "images" / "cleaned")
        text_dir = ensure_dir(book_output_dir / "text")
        chapters_dir = ensure_dir(book_output_dir / "chapters")
        references_dir = ensure_dir(book_output_dir / "references")
        analytics_dir = ensure_dir(book_output_dir / "analytics")
        reports_dir = ensure_dir(book_output_dir / "reports")

        result = PipelineResult(book_title=book_title, output_dir=book_output_dir, page_count=0)

        # ---- 1. Render source PDF pages to images -------------------------------------------------
        _report_progress(progress_callback, 0.02, "Rendering scanned pages from PDF...")
        rendered_pages = pdf_io.render_pdf_to_images(pdf_path, dpi=options.cleaning.dpi)
        result.page_count = len(rendered_pages)
        if not rendered_pages:
            result.warnings.append("No pages could be rendered from the PDF.")
            return result

        # ---- 2. Clean each page, saving both original and cleaned images --------------------------
        cleaned_pil_images: List[Image.Image] = []
        for i, rendered in enumerate(rendered_pages):
            page_no = safe_page_number(i)
            original_path = images_original_dir / f"page_{page_no}.png"
            rendered.image.save(original_path)
            result.original_image_paths.append(original_path)

            cleaning_result = clean_page_image(rendered.image, options.cleaning)
            cleaned_path = images_cleaned_dir / f"page_{page_no}.png"
            cleaning_result.image.save(cleaned_path)
            result.cleaned_image_paths.append(cleaned_path)
            result.cleaning_metrics.append(cleaning_result.metrics)
            cleaned_pil_images.append(cleaning_result.image)

            fraction = 0.05 + 0.30 * ((i + 1) / len(rendered_pages))
            _report_progress(progress_callback, fraction, f"Cleaning page {i + 1}/{len(rendered_pages)}...")

        result.image_quality = analytics.compute_image_quality_stats(result.cleaning_metrics)

        # ---- 3. Rebuild a cleaned PDF ------------------------------------------------------------
        _report_progress(progress_callback, 0.37, "Assembling cleaned PDF...")
        try:
            result.cleaned_pdf_path = pdf_io.images_to_pdf(result.cleaned_image_paths, book_output_dir / "cleaned.pdf")
        except Exception as exc:
            result.warnings.append(f"Could not assemble cleaned PDF: {exc}")

        # ---- 4. OCR every cleaned page ------------------------------------------------------------
        ocr_languages = options.ocr.languages or ["eng"]
        for i, image in enumerate(cleaned_pil_images):
            try:
                ocr_result = ocr.ocr_image(image, languages=ocr_languages, psm=options.ocr.psm)
                page_text = ocr_result.text
            except ocr.TesseractNotFoundError as exc:
                result.warnings.append(str(exc))
                page_text = ""
            result.page_texts.append(page_text)
            (text_dir / "per_page").mkdir(parents=True, exist_ok=True)
            (text_dir / "per_page" / f"page_{safe_page_number(i)}.txt").write_text(page_text, encoding="utf-8")

            page_lang = detect_language(page_text) if options.ocr.auto_detect_language else None
            result.page_languages.append(page_lang)

            fraction = 0.37 + 0.20 * ((i + 1) / len(cleaned_pil_images))
            _report_progress(progress_callback, fraction, f"Running OCR on page {i + 1}/{len(cleaned_pil_images)}...")

        result.full_text = "\n\n".join(result.page_texts)
        (text_dir / "full_text_original.txt").write_text(result.full_text, encoding="utf-8")

        non_null_langs = [l for l in result.page_languages if l]
        result.detected_primary_language = max(set(non_null_langs), key=non_null_langs.count) if non_null_langs else None

        # If auto-detect found a dominant non-English language and the user didn't
        # explicitly pick OCR languages beyond the default, re-OCR isn't repeated
        # here (kept as a single pass for performance); the detected language is
        # still surfaced for translation/summarization defaults.

        # ---- 5. Chapter detection -----------------------------------------------------------------
        _report_progress(progress_callback, 0.58, "Detecting chapter boundaries...")
        pages = [PageText(index=i, text=t) for i, t in enumerate(result.page_texts)]
        result.chapters = split_into_chapters(pages)

        # ---- 6. Translation (optional) -----------------------------------------------------------
        if options.translation.enabled and result.full_text.strip():
            _report_progress(progress_callback, 0.60, "Translating full text...")
            translation = translate_text(
                result.full_text,
                target_language=options.translation.target_language,
                source_language=result.detected_primary_language,
            )
            result.translation = translation
            if translation.warning:
                result.warnings.append(translation.warning)
            out_name = f"full_text_translated_{options.translation.target_language}.txt"
            (text_dir / out_name).write_text(translation.text, encoding="utf-8")

            if options.translation.translate_chapters_individually:
                for i, chapter in enumerate(result.chapters):
                    chap_translation = translate_text(
                        chapter.text,
                        target_language=options.translation.target_language,
                        source_language=result.detected_primary_language,
                    )
                    chapter.translated_text = chap_translation.text
                    fraction = 0.60 + 0.10 * ((i + 1) / max(1, len(result.chapters)))
                    _report_progress(progress_callback, fraction, f"Translating chapter {i + 1}/{len(result.chapters)}...")

        # ---- 7. Chapter synopses -------------------------------------------------------------------
        for i, chapter in enumerate(result.chapters):
            chapter.synopsis = summarize_chapter(
                chapter.text,
                title=chapter.title,
                sentence_count=options.analysis.summary_sentence_count,
                iso_lang=result.detected_primary_language,
                prefer_llm=options.analysis.use_llm_if_available,
            )
            chapter_dir = ensure_dir(chapters_dir / f"chapter_{chapter.number:02d}")
            (chapter_dir / "text.txt").write_text(chapter.text, encoding="utf-8")
            (chapter_dir / "synopsis.txt").write_text(chapter.synopsis, encoding="utf-8")
            if chapter.translated_text:
                (chapter_dir / "translated.txt").write_text(chapter.translated_text, encoding="utf-8")
            fraction = 0.72 + 0.10 * ((i + 1) / max(1, len(result.chapters)))
            _report_progress(progress_callback, fraction, f"Summarizing chapter {i + 1}/{len(result.chapters)}...")

        # ---- 8. Reference extraction ----------------------------------------------------------------
        _report_progress(progress_callback, 0.83, "Scanning for esoteric, theological & scientific references...")
        pages_text_map = {i: t for i, t in enumerate(result.page_texts)}
        result.references_by_category = references_mod.find_all_references(
            pages_text_map,
            categories=options.analysis.reference_categories,
            context_chars=options.analysis.reference_context_chars,
        )
        for category, refs in result.references_by_category.items():
            write_json(
                references_dir / f"{category}.json",
                [ref.__dict__ for ref in refs],
            )
        result.reference_stats = analytics.compute_reference_stats(result.references_by_category)

        # ---- 9. Analytics -----------------------------------------------------------------------------
        _report_progress(progress_callback, 0.89, "Computing advanced analytics...")
        result.stats = analytics.compute_text_statistics(result.full_text)
        result.sentiment_by_chapter = analytics.compute_sentiment_by_chapter(result.chapters)
        result.word_frequency = analytics.compute_word_frequency(result.full_text, top_n=50)
        write_json(analytics_dir / "stats.json", result.stats.__dict__)
        write_json(analytics_dir / "sentiment_by_chapter.json", result.sentiment_by_chapter)
        write_json(analytics_dir / "word_frequency.json", result.word_frequency)

        # ---- 10. Insights -----------------------------------------------------------------------------
        _report_progress(progress_callback, 0.93, "Generating insights and thematic analysis...")
        result.insights = insights_mod.generate_insights(
            result.full_text,
            result.sentiment_by_chapter,
            iso_lang=result.detected_primary_language,
            top_n_keywords=options.analysis.top_keywords,
            use_llm=options.analysis.use_llm_if_available,
        )
        write_json(
            book_output_dir / "insights" / "insights.json",
            {
                "themes": result.insights.themes,
                "narrative_summary": result.insights.narrative_summary,
                "sentiment_arc_description": result.insights.sentiment_arc_description,
                "top_keywords": result.insights.top_keywords,
                "notable_entities": result.insights.notable_entities,
            },
        )

        # ---- 11. Visualizations + reports ----------------------------------------------------------
        _report_progress(progress_callback, 0.96, "Rendering visualizations and building reports...")
        figures = {
            "Word Frequency": word_frequency_bar(result.word_frequency[:25]),
            "Sentiment Arc by Chapter": sentiment_arc_line(result.sentiment_by_chapter),
            "Reference Mentions by Category": reference_category_pie(result.reference_stats),
            "Flesch Reading Ease": readability_gauge(result.stats.flesch_reading_ease),
            "Chapter Length": chapter_length_bar(result.chapters),
            "Detected Language per Page": language_distribution_pie(
                analytics.compute_language_distribution(result.page_languages)
            ),
            "Notable Entities": entity_frequency_bar(result.insights.notable_entities[:20]),
            "Per-page Noise Reduction": noise_reduction_bar(result.cleaning_metrics),
        }

        try:
            wc_image = generate_wordcloud_image(result.word_frequency)
            if wc_image is not None:
                wc_image.save(analytics_dir / "wordcloud.png")
        except Exception as exc:
            result.warnings.append(f"Word cloud generation failed: {exc}")

        try:
            result.html_report_path = build_html_report(
                analytics_dir / "analytics_report.html",
                title=book_title,
                stats=result.stats,
                narrative_summary=result.insights.narrative_summary,
                sentiment_arc=result.insights.sentiment_arc_description,
                figures=figures,
            )
        except Exception as exc:
            result.warnings.append(f"HTML report generation failed: {exc}")

        try:
            result.pdf_report_path = build_pdf_report(
                reports_dir / "full_report.pdf",
                title=book_title,
                stats=result.stats,
                chapters=result.chapters,
                insights=result.insights,
                reference_stats=result.reference_stats,
                figures=figures,
            )
        except Exception as exc:
            result.warnings.append(f"PDF report generation failed: {exc}")

        # ---- 12. Manifest -----------------------------------------------------------------------------
        manifest = {
            "book_title": book_title,
            "page_count": result.page_count,
            "detected_primary_language": language_name(result.detected_primary_language),
            "warnings": result.warnings,
            "outputs": {
                "cleaned_pdf": str(result.cleaned_pdf_path) if result.cleaned_pdf_path else None,
                "html_report": str(result.html_report_path) if result.html_report_path else None,
                "pdf_report": str(result.pdf_report_path) if result.pdf_report_path else None,
                "images_original_dir": str(images_original_dir),
                "images_cleaned_dir": str(images_cleaned_dir),
                "text_dir": str(text_dir),
                "chapters_dir": str(chapters_dir),
                "references_dir": str(references_dir),
                "analytics_dir": str(analytics_dir),
            },
            "chapter_titles": [c.title for c in result.chapters],
            "reference_stats": result.reference_stats,
        }
        result.manifest_path = book_output_dir / "manifest.json"
        write_json(result.manifest_path, manifest)

        _report_progress(progress_callback, 1.0, "Done.")
        return result
