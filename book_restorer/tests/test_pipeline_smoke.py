import shutil
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from core.config import AnalysisOptions, CleaningOptions, OCROptions, PipelineOptions, TranslationOptions
from core.ocr import tesseract_available
from core.pipeline import BookRestorationPipeline


def _make_sample_pdf(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=LETTER)
    width, height = LETTER

    c.setFont("Helvetica-Bold", 22)
    c.drawString(72, height - 100, "CHAPTER I")
    c.setFont("Helvetica", 13)
    text_lines = [
        "In the beginning the old alchemist studied the philosopher's stone in secret.",
        "He believed the hypothesis of transmutation could be proven by careful experiment.",
        "The prophet in the village spoke often of salvation and divine grace.",
        "This wonderful and joyful chapter set a hopeful tone for the whole book.",
    ]
    y = height - 160
    for line in text_lines * 3:
        c.drawString(72, y, line)
        y -= 20
        if y < 72:
            c.showPage()
            c.setFont("Helvetica", 13)
            y = height - 72
    c.showPage()

    c.setFont("Helvetica-Bold", 22)
    c.drawString(72, height - 100, "CHAPTER II")
    c.setFont("Helvetica", 13)
    text_lines_2 = [
        "The tarot cards were laid out upon the table beside sacred geometry diagrams.",
        "Later the scientist described atomic theory and the scientific method in detail.",
        "This terrible and awful turn of events darkened the mood considerably.",
    ]
    y = height - 160
    for line in text_lines_2 * 3:
        c.drawString(72, y, line)
        y -= 20
        if y < 72:
            c.showPage()
            c.setFont("Helvetica", 13)
            y = height - 72
    c.showPage()
    c.save()


@pytest.mark.skipif(not tesseract_available(), reason="tesseract binary not installed")
def test_full_pipeline_end_to_end(tmp_path: Path):
    pdf_path = tmp_path / "sample_book.pdf"
    _make_sample_pdf(pdf_path)

    output_dir = tmp_path / "output"
    options = PipelineOptions(
        output_dir=output_dir,
        cleaning=CleaningOptions(dpi=150),
        ocr=OCROptions(languages=["eng"], auto_detect_language=True),
        translation=TranslationOptions(enabled=False),
        analysis=AnalysisOptions(summary_sentence_count=2, use_llm_if_available=False),
        book_title="Test Book",
    )

    progress_events = []

    def on_progress(fraction, message):
        progress_events.append((fraction, message))

    pipeline = BookRestorationPipeline()
    result = pipeline.run(pdf_path, options, progress_callback=on_progress)

    assert result.page_count == 2
    assert len(result.original_image_paths) == 2
    assert len(result.cleaned_image_paths) == 2
    assert all(p.exists() for p in result.original_image_paths)
    assert all(p.exists() for p in result.cleaned_image_paths)
    assert result.cleaned_pdf_path is not None and result.cleaned_pdf_path.exists()

    assert result.full_text.strip() != ""
    assert len(result.chapters) >= 1
    assert result.stats is not None and result.stats.word_count > 0

    assert "esoteric" in result.reference_stats
    assert "theological" in result.reference_stats
    assert "scientific" in result.reference_stats

    assert result.manifest_path is not None and result.manifest_path.exists()
    assert result.html_report_path is not None and result.html_report_path.exists()

    assert progress_events[-1][0] == 1.0
    shutil.rmtree(output_dir, ignore_errors=True)
