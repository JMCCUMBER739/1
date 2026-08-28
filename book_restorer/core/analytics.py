"""Advanced text analytics: readability, sentiment, frequency, language mix."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from core.utils import word_tokenize_basic

_analyzer = SentimentIntensityAnalyzer()


@dataclass
class TextStatistics:
    word_count: int = 0
    unique_word_count: int = 0
    sentence_count: int = 0
    character_count: int = 0
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0
    flesch_reading_ease: float = 0.0
    flesch_kincaid_grade: float = 0.0
    gunning_fog: float = 0.0
    estimated_reading_minutes: float = 0.0
    lexical_diversity: float = 0.0


def compute_text_statistics(text: str) -> TextStatistics:
    text = text.strip()
    if not text:
        return TextStatistics()

    words = word_tokenize_basic(text)
    unique_words = set(words)
    sentence_count = max(textstat.sentence_count(text), 1)
    word_count = len(words) or 1

    try:
        flesch = textstat.flesch_reading_ease(text)
        fk_grade = textstat.flesch_kincaid_grade(text)
        fog = textstat.gunning_fog(text)
    except Exception:
        flesch, fk_grade, fog = 0.0, 0.0, 0.0

    return TextStatistics(
        word_count=word_count,
        unique_word_count=len(unique_words),
        sentence_count=sentence_count,
        character_count=len(text),
        avg_word_length=sum(len(w) for w in words) / word_count,
        avg_sentence_length=word_count / sentence_count,
        flesch_reading_ease=flesch,
        flesch_kincaid_grade=fk_grade,
        gunning_fog=fog,
        estimated_reading_minutes=word_count / 200.0,
        lexical_diversity=len(unique_words) / word_count,
    )


def compute_word_frequency(text: str, top_n: int = 50, stopwords: Optional[set] = None) -> List[tuple]:
    stopwords = stopwords or set()
    words = [w for w in word_tokenize_basic(text) if len(w) > 2 and w not in stopwords]
    return Counter(words).most_common(top_n)


def compute_sentiment(text: str) -> Dict[str, float]:
    if not text.strip():
        return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
    return _analyzer.polarity_scores(text)


def compute_sentiment_by_chapter(chapters) -> List[Dict]:
    results = []
    for chapter in chapters:
        scores = compute_sentiment(chapter.text)
        results.append({"chapter": chapter.number, "title": chapter.title, **scores})
    return results


def compute_language_distribution(page_languages: List[Optional[str]]) -> Dict[str, int]:
    counts = Counter(lang or "unknown" for lang in page_languages)
    return dict(counts)


def compute_reference_stats(references_by_category: Dict[str, list]) -> Dict[str, Dict]:
    stats = {}
    for category, refs in references_by_category.items():
        term_counts = Counter(r.term for r in refs)
        stats[category] = {
            "total_matches": len(refs),
            "unique_terms": len(term_counts),
            "top_terms": term_counts.most_common(10),
        }
    return stats


@dataclass
class ImageQualityStats:
    pages_processed: int = 0
    avg_skew_correction_degrees: float = 0.0
    avg_noise_reduction_pct: float = 0.0


def compute_image_quality_stats(page_metrics: List[Dict]) -> ImageQualityStats:
    if not page_metrics:
        return ImageQualityStats()
    skews = [abs(m.get("skew_angle_degrees", 0.0)) for m in page_metrics]
    noise_reductions = []
    for m in page_metrics:
        before, after = m.get("noise_before", 0.0), m.get("noise_after", 0.0)
        if before > 0:
            noise_reductions.append(max(0.0, (before - after) / before * 100))
    return ImageQualityStats(
        pages_processed=len(page_metrics),
        avg_skew_correction_degrees=sum(skews) / len(skews),
        avg_noise_reduction_pct=(sum(noise_reductions) / len(noise_reductions)) if noise_reductions else 0.0,
    )
