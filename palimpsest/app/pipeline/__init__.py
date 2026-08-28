"""End-to-end orchestration of the Palimpsest pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.config import OCR_LANG_MAP, SETTINGS
from app.pipeline.analytics import build_analytics
from app.pipeline.chapters import detect_chapters, enrich_chapters
from app.pipeline.clean import clean_pages
from app.pipeline.export import export_all
from app.pipeline.ingest import render_pages
from app.pipeline.models import JobResult
from app.pipeline.ocr import extract_texts
from app.pipeline.references import find_references
from app.pipeline.translate import translate_pages

ProgressCb = Callable[[float, str], None]


def run_pipeline(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int = 200,
    max_pages: int | None = None,
    clean_mode: str = "auto",
    binarize: bool = True,
    enable_ocr: bool = True,
    ocr_lang_code: str = "en",
    enable_translation: bool = True,
    translate_target: str = "en",
    progress: ProgressCb | None = None,
) -> JobResult:
    def report(pct: float, msg: str) -> None:
        if progress:
            progress(pct, msg)

    pdf_path = Path(pdf_path).expanduser().resolve()
    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    out = Path(output_dir).expanduser().resolve() / pdf_path.stem / job_id
    out.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    report(0.02, "Rasterizing pages…")
    pages = render_pages(pdf_path, out, dpi=dpi, max_pages=max_pages)

    report(0.18, "Cleaning scanned images…")
    pages = clean_pages(pages, out, mode=clean_mode, binarize=binarize)

    ocr_lang = OCR_LANG_MAP.get(ocr_lang_code, SETTINGS.ocr_lang)
    report(0.35, "Extracting / OCR text…")
    texts = extract_texts(
        pages,
        pdf_path,
        ocr_lang=ocr_lang,
        enable_ocr=enable_ocr,
        max_pages=max_pages,
    )
    if enable_ocr and all(not t.cleaned_text for t in texts):
        warnings.append("OCR produced no text. Install Tesseract for scanned books, or provide a text-layer PDF.")

    if enable_translation:
        report(0.55, "Translating pages…")
        try:
            texts = translate_pages(texts, target=translate_target, enabled=True)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Translation partially failed: {exc}")
    else:
        report(0.55, "Skipping translation…")

    report(0.68, "Detecting chapters & writing synopses…")
    chapters = enrich_chapters(detect_chapters(texts))

    report(0.78, "Mining esoteric, theological & scientific references…")
    references = find_references(texts, chapters)

    report(0.86, "Computing analytics…")
    analytics = build_analytics(pages, texts, chapters, references)

    result = JobResult(
        job_id=job_id,
        source_pdf=pdf_path,
        output_dir=out,
        created_at=datetime.now(timezone.utc),
        pages=pages,
        texts=texts,
        chapters=chapters,
        references=references,
        analytics=analytics,
        warnings=warnings,
    )

    report(0.92, "Exporting PDFs, reports & assets…")
    result = export_all(result)
    result.manifest = {
        "job_id": job_id,
        "source": str(pdf_path),
        "output_dir": str(out),
        "cleaned_pdf": str(result.cleaned_pdf) if result.cleaned_pdf else None,
        "translated_pdf": str(result.translated_pdf) if result.translated_pdf else None,
        "report_html": str(result.report_html) if result.report_html else None,
    }
    report(1.0, "Complete")
    return result
