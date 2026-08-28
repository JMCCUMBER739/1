from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.assets.make_sample_pdf import build_sample_pdf
from app.pipeline import run_pipeline
from app.pipeline.chapters import detect_chapters, enrich_chapters
from app.pipeline.models import PageText
from app.pipeline.ocr import normalize_text
from app.pipeline.references import find_references


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("pdf") / "sample.pdf"
    return build_sample_pdf(path)


def test_normalize_text_hyphenation():
    assert "knowledge" in normalize_text("knowl-\nedge of the wise")


def test_chapter_detection_and_synopsis():
    texts = [
        PageText(0, "Chapter I: Beginnings\nOnce upon a hermetic night the initiate sought gnosis.", "Chapter I: Beginnings\nOnce upon a hermetic night the initiate sought gnosis."),
        PageText(1, "More text about the sacred covenant and divine grace in theology.", "More text about the sacred covenant and divine grace in theology."),
        PageText(2, "Chapter II: Science\nThe experiment measured gravity and force with care.", "Chapter II: Science\nThe experiment measured gravity and force with care."),
    ]
    chapters = enrich_chapters(detect_chapters(texts))
    assert len(chapters) >= 2
    assert chapters[0].synopsis
    assert chapters[0].insights


def test_reference_extraction():
    texts = [
        PageText(
            0,
            "",
            "The alchemist studied the philosopher's stone and Hermes Trismegistus. "
            "Theology of the Trinity and salvation. Gravity and the electron in scientific study.",
        )
    ]
    hits = find_references(texts)
    kinds = {h.kind.value for h in hits}
    assert "esoteric" in kinds
    assert "theological" in kinds
    assert "scientific" in kinds
    assert all(h.excerpt for h in hits)


def test_pipeline_end_to_end(sample_pdf: Path, tmp_path: Path):
    result = run_pipeline(
        sample_pdf,
        tmp_path / "out",
        dpi=120,
        max_pages=4,
        enable_ocr=False,
        enable_translation=False,
        binarize=True,
    )
    assert result.cleaned_pdf and result.cleaned_pdf.exists()
    assert result.translated_pdf and result.translated_pdf.exists()
    assert result.report_html and result.report_html.exists()
    assert (result.output_dir / "images" / "original").exists()
    assert (result.output_dir / "images" / "cleaned").exists()
    assert (result.output_dir / "synopses" / "chapters.md").exists()
    assert (result.output_dir / "references" / "all_references.csv").exists()
    assert (result.output_dir / "analytics" / "report.json").exists()
    assert result.analytics.page_count >= 1
    assert result.analytics.word_count > 50
    assert len(result.references) >= 5
    assert len(result.chapters) >= 1
