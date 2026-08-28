"""Book-level analytics aggregation.

Combines OCR quality, restoration metrics, linguistic statistics and
reference-mining results into a single serializable analytics object
that feeds the charts, the HTML report and the GUI dashboard.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .chapters import Chapter
from .imaging import CleanResult
from .ocr import PageOCR
from .references import ReferenceReport
from .textutils import content_words, flesch_reading_ease, reading_level_label, split_sentences, tokenize


@dataclass
class BookAnalytics:
    page_count: int = 0
    word_count: int = 0
    unique_words: int = 0
    sentence_count: int = 0
    lexical_diversity: float = 0.0
    avg_sentence_length: float = 0.0
    flesch_score: float = 0.0
    reading_level: str = ""
    mean_ocr_confidence: float = 0.0
    ocr_confidence_by_page: list[float] = field(default_factory=list)
    words_by_page: list[int] = field(default_factory=list)
    words_by_chapter: dict[str, int] = field(default_factory=dict)
    top_words: list[tuple[str, int]] = field(default_factory=list)
    top_bigrams: list[tuple[str, int]] = field(default_factory=list)
    contrast_before: list[float] = field(default_factory=list)
    contrast_after: list[float] = field(default_factory=list)
    sharpness_before: list[float] = field(default_factory=list)
    sharpness_after: list[float] = field(default_factory=list)
    noise_before: list[float] = field(default_factory=list)
    noise_after: list[float] = field(default_factory=list)
    skew_by_page: list[float] = field(default_factory=list)
    category_counts: dict[str, int] = field(default_factory=dict)
    category_density_by_chapter: dict[str, dict[int, float]] = field(default_factory=dict)
    top_terms_by_category: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    syncretic_count: int = 0

    def to_dict(self) -> dict:
        return {
            "document": {
                "pages": self.page_count,
                "words": self.word_count,
                "unique_words": self.unique_words,
                "sentences": self.sentence_count,
                "lexical_diversity": self.lexical_diversity,
                "avg_sentence_length": self.avg_sentence_length,
                "flesch_reading_ease": self.flesch_score,
                "reading_level": self.reading_level,
            },
            "ocr": {
                "mean_confidence": self.mean_ocr_confidence,
                "confidence_by_page": self.ocr_confidence_by_page,
                "words_by_page": self.words_by_page,
            },
            "restoration": {
                "contrast_before": self.contrast_before,
                "contrast_after": self.contrast_after,
                "sharpness_before": self.sharpness_before,
                "sharpness_after": self.sharpness_after,
                "noise_before": self.noise_before,
                "noise_after": self.noise_after,
                "skew_corrected_by_page": self.skew_by_page,
            },
            "language": {
                "top_words": self.top_words,
                "top_bigrams": self.top_bigrams,
                "words_by_chapter": self.words_by_chapter,
            },
            "references": {
                "category_counts": self.category_counts,
                "density_per_1k_words_by_chapter": self.category_density_by_chapter,
                "top_terms_by_category": self.top_terms_by_category,
                "syncretic_passages": self.syncretic_count,
            },
        }


def compute_analytics(
    page_texts: list[str],
    ocr_pages: list[PageOCR],
    clean_results: list[CleanResult],
    chapters: list[Chapter],
    references: ReferenceReport,
) -> BookAnalytics:
    a = BookAnalytics()
    full_text = "\n".join(page_texts)

    words = tokenize(full_text)
    contents = content_words(full_text)
    sentences = split_sentences(full_text)

    a.page_count = len(page_texts)
    a.word_count = len(words)
    a.unique_words = len(set(words))
    a.sentence_count = len(sentences)
    a.lexical_diversity = round(len(set(words)) / len(words), 4) if words else 0.0
    a.avg_sentence_length = round(len(words) / len(sentences), 1) if sentences else 0.0
    a.flesch_score = flesch_reading_ease(full_text)
    a.reading_level = reading_level_label(a.flesch_score)

    a.ocr_confidence_by_page = [p.confidence for p in ocr_pages]
    a.words_by_page = [p.word_count for p in ocr_pages]
    if ocr_pages:
        a.mean_ocr_confidence = round(sum(p.confidence for p in ocr_pages) / len(ocr_pages), 1)

    for c in clean_results:
        a.contrast_before.append(round(c.before.contrast, 1))
        a.contrast_after.append(round(c.after.contrast, 1))
        a.sharpness_before.append(round(c.before.sharpness, 1))
        a.sharpness_after.append(round(c.after.sharpness, 1))
        a.noise_before.append(round(c.before.noise, 2))
        a.noise_after.append(round(c.after.noise, 2))
        a.skew_by_page.append(c.skew_angle)

    word_freq = Counter(contents)
    a.top_words = word_freq.most_common(25)

    bigrams = Counter(f"{contents[i]} {contents[i + 1]}" for i in range(len(contents) - 1))
    a.top_bigrams = [(b, n) for b, n in bigrams.most_common(15) if n > 1]

    for chapter in chapters:
        label = f"Ch. {chapter.number}" if chapter.number else "Front"
        a.words_by_chapter[label] = len(tokenize(chapter.text))

    a.category_counts = {cat: references.total(cat) for cat in references.hits}
    a.category_density_by_chapter = references.chapter_density
    a.top_terms_by_category = {
        cat: sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:12]
        for cat, counts in references.term_counts.items()
    }
    a.syncretic_count = len(references.syncretic)
    return a
