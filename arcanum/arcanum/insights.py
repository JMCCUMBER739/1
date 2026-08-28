"""Insight generation: notable findings distilled from the analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re

from .analytics import BookAnalytics
from .chapters import Chapter
from .references import ReferenceReport
from .synopsis import summarize_text
from .textutils import STOPWORDS


@dataclass
class Insight:
    title: str
    detail: str
    kind: str = "general"  # general | thematic | textual | quality

    def to_dict(self) -> dict:
        return {"title": self.title, "detail": self.detail, "kind": self.kind}


@dataclass
class InsightReport:
    insights: list[Insight] = field(default_factory=list)
    notable_quotes: list[str] = field(default_factory=list)
    recurring_figures: list[tuple[str, int]] = field(default_factory=list)


_PROPER_RE = re.compile(r"\b([A-Z][a-z]{2,})(?:\s+([A-Z][a-z]{2,}))?\b")


def _recurring_figures(text: str) -> list[tuple[str, int]]:
    """Frequent capitalized names that are not sentence-initial noise."""
    counts: Counter = Counter()
    for m in _PROPER_RE.finditer(text):
        name = m.group(0)
        if name.lower() in STOPWORDS:
            continue
        counts[name] += 1
    return [(n, c) for n, c in counts.most_common(30) if c >= 3][:12]


def build_insights(
    full_text: str, chapters: list[Chapter], analytics: BookAnalytics, references: ReferenceReport
) -> InsightReport:
    report = InsightReport()
    ins = report.insights

    # --- Thematic profile -------------------------------------------------
    counts = analytics.category_counts
    if any(counts.values()):
        dominant = max(counts, key=counts.get)
        share = counts[dominant] / max(1, sum(counts.values()))
        ins.append(
            Insight(
                title=f"Dominant register: {dominant}",
                detail=(
                    f"Of {sum(counts.values())} mined reference passages, "
                    f"{counts[dominant]} ({share:.0%}) are {dominant}. "
                    f"Full breakdown — esoteric: {counts.get('esoteric', 0)}, "
                    f"theological: {counts.get('theological', 0)}, "
                    f"scientific: {counts.get('scientific', 0)}."
                ),
                kind="thematic",
            )
        )
    if references.syncretic:
        top = references.syncretic[0]
        ins.append(
            Insight(
                title=f"{len(references.syncretic)} syncretic passages blend traditions",
                detail=(
                    f"Passages mixing categories were found; the strongest "
                    f"({top.category}, ch. {top.chapter_number}) reads: "
                    f"\u201c{top.excerpt[:220]}\u2026\u201d"
                ),
                kind="thematic",
            )
        )

    for cat, terms in analytics.top_terms_by_category.items():
        if terms:
            listed = ", ".join(f"{t} ({n})" for t, n in terms[:5])
            ins.append(
                Insight(
                    title=f"Signature {cat} vocabulary",
                    detail=f"Most recurrent {cat} terms: {listed}.",
                    kind="thematic",
                )
            )

    # --- Language profile -------------------------------------------------
    ins.append(
        Insight(
            title=f"Reading level: {analytics.reading_level}",
            detail=(
                f"Flesch reading ease {analytics.flesch_score} with an "
                f"average sentence length of {analytics.avg_sentence_length} "
                f"words and lexical diversity of "
                f"{analytics.lexical_diversity:.2%} "
                f"({analytics.unique_words:,} unique of "
                f"{analytics.word_count:,} words)."
            ),
            kind="textual",
        )
    )
    if analytics.top_bigrams:
        listed = ", ".join(f"\u201c{b}\u201d ({n}\u00d7)" for b, n in analytics.top_bigrams[:5])
        ins.append(
            Insight(title="Recurring phrases", detail=f"Most repeated two-word phrases: {listed}.", kind="textual")
        )

    report.recurring_figures = _recurring_figures(full_text)
    if report.recurring_figures:
        listed = ", ".join(f"{n} ({c}\u00d7)" for n, c in report.recurring_figures[:6])
        ins.append(
            Insight(
                title="Recurring names and figures",
                detail=f"Frequently mentioned proper names: {listed}.",
                kind="textual",
            )
        )

    # --- Structure --------------------------------------------------------
    if chapters:
        longest = max(chapters, key=lambda c: len(c.text.split()))
        shortest = min(chapters, key=lambda c: len(c.text.split()))
        ins.append(
            Insight(
                title=f"Structure: {len(chapters)} chapters detected",
                detail=(
                    f"Longest is \u201c{longest.title}\u201d "
                    f"(~{len(longest.text.split()):,} words, pages "
                    f"{longest.start_page}\u2013{longest.end_page}); shortest is "
                    f"\u201c{shortest.title}\u201d "
                    f"(~{len(shortest.text.split()):,} words)."
                ),
                kind="general",
            )
        )

    # --- Scan / restoration quality ----------------------------------------
    if analytics.ocr_confidence_by_page:
        worst = min(range(len(analytics.ocr_confidence_by_page)), key=lambda i: analytics.ocr_confidence_by_page[i])
        ins.append(
            Insight(
                title=f"OCR confidence averaged {analytics.mean_ocr_confidence}%",
                detail=(
                    f"Weakest page is {worst + 1} at "
                    f"{analytics.ocr_confidence_by_page[worst]}% — consider "
                    f"re-scanning it at a higher resolution."
                ),
                kind="quality",
            )
        )
    if analytics.contrast_before and analytics.contrast_after:
        before = sum(analytics.contrast_before) / len(analytics.contrast_before)
        after = sum(analytics.contrast_after) / len(analytics.contrast_after)
        if before > 0:
            ins.append(
                Insight(
                    title=f"Restoration lifted contrast {((after - before) / before):+.0%}",
                    detail=(
                        f"Mean page contrast improved from {before:.0f} to "
                        f"{after:.0f}; median residual noise fell from "
                        f"{_safe_median(analytics.noise_before):.2f} to "
                        f"{_safe_median(analytics.noise_after):.2f}."
                    ),
                    kind="quality",
                )
            )
    corrected = [s for s in analytics.skew_by_page if abs(s) > 0.15]
    if corrected:
        ins.append(
            Insight(
                title=f"Deskewed {len(corrected)} pages",
                detail=(f"Largest correction was " f"{max(corrected, key=abs):+.2f}\u00b0."),
                kind="quality",
            )
        )

    # --- Notable quotes -----------------------------------------------------
    _, key_sentences, _ = summarize_text(full_text, max_sentences=6)
    report.notable_quotes = key_sentences
    return report


def _safe_median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
