"""Shared data models for the processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ReferenceKind(str, Enum):
    ESOTERIC = "esoteric"
    THEOLOGICAL = "theological"
    SCIENTIFIC = "scientific"


@dataclass
class PageImage:
    page_index: int
    original_path: Path
    cleaned_path: Path | None = None
    width: int = 0
    height: int = 0
    skew_degrees: float = 0.0
    quality_score: float = 0.0


@dataclass
class PageText:
    page_index: int
    raw_text: str
    cleaned_text: str
    translated_text: str | None = None
    detected_language: str | None = None
    confidence: float = 0.0


@dataclass
class Chapter:
    index: int
    title: str
    start_page: int
    end_page: int
    text: str
    synopsis: str = ""
    insights: list[str] = field(default_factory=list)
    word_count: int = 0


@dataclass
class ReferenceHit:
    kind: ReferenceKind
    category: str
    term: str
    excerpt: str
    page_index: int
    chapter_title: str | None = None
    context_score: float = 1.0


@dataclass
class AnalyticsReport:
    page_count: int
    word_count: int
    unique_words: int
    avg_words_per_page: float
    chapter_count: int
    language_distribution: dict[str, float]
    readability_score: float
    lexical_diversity: float
    reference_counts: dict[str, int]
    top_terms: list[tuple[str, int]]
    quality_by_page: list[float]
    sentiment_proxy: float
    thematic_weights: dict[str, float]


@dataclass
class JobResult:
    job_id: str
    source_pdf: Path
    output_dir: Path
    created_at: datetime
    pages: list[PageImage]
    texts: list[PageText]
    chapters: list[Chapter]
    references: list[ReferenceHit]
    analytics: AnalyticsReport
    cleaned_pdf: Path | None = None
    translated_pdf: Path | None = None
    report_html: Path | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
