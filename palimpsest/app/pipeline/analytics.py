"""Advanced book analytics."""

from __future__ import annotations

import math
import re
from collections import Counter

from app.pipeline.models import AnalyticsReport, Chapter, PageImage, PageText, ReferenceHit

_STOP = {
    "the", "and", "for", "that", "with", "this", "from", "have", "are", "was",
    "were", "been", "being", "their", "there", "which", "would", "could", "should",
    "about", "into", "when", "what", "your", "them", "than", "then", "also", "only",
    "some", "such", "very", "more", "most", "other", "upon", "shall", "will", "these",
    "those", "his", "her", "its", "our", "you", "not", "but", "all", "any", "can",
    "had", "has", "may", "who", "whom", "how", "why", "did", "does", "done", "out",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']{3,}", text.lower())


def flesch_reading_ease(text: str) -> float:
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)
    words = _tokens(text)
    if not words:
        return 0.0
    syllables = sum(max(1, len(re.findall(r"[aeiouy]+", w))) for w in words)
    score = 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))
    return round(max(0.0, min(100.0, score)), 2)


def sentiment_proxy(text: str) -> float:
    """Simple lexicon polarity in [-1, 1]."""
    pos = {
        "good", "great", "beauty", "divine", "light", "truth", "love", "wisdom",
        "holy", "grace", "harmony", "virtue", "hope", "peace", "glorious", "sacred",
    }
    neg = {
        "evil", "dark", "death", "fear", "false", "sin", "chaos", "pain", "wrath",
        "corruption", "ignorance", "despair", "hell", "curse", "violence",
    }
    toks = _tokens(text)
    if not toks:
        return 0.0
    p = sum(1 for t in toks if t in pos)
    n = sum(1 for t in toks if t in neg)
    return round((p - n) / math.sqrt(len(toks)), 3)


def thematic_weights(references: list[ReferenceHit]) -> dict[str, float]:
    counts = Counter(r.kind.value for r in references)
    total = sum(counts.values()) or 1
    return {k: round(v / total, 3) for k, v in counts.items()}


def build_analytics(
    pages: list[PageImage],
    texts: list[PageText],
    chapters: list[Chapter],
    references: list[ReferenceHit],
) -> AnalyticsReport:
    full = "\n".join((t.translated_text or t.cleaned_text or "") for t in texts)
    tokens = _tokens(full)
    word_count = len(tokens)
    unique = len(set(tokens))
    page_count = len(pages) or 1

    lang_counts: Counter[str] = Counter()
    for t in texts:
        if t.detected_language:
            lang_counts[t.detected_language] += 1
    lang_total = sum(lang_counts.values()) or 1
    lang_dist = {k: round(v / lang_total, 3) for k, v in lang_counts.items()}

    content_words = [w for w in tokens if w not in _STOP]
    top_terms = Counter(content_words).most_common(25)

    ref_counts = Counter(r.kind.value for r in references)

    return AnalyticsReport(
        page_count=len(pages),
        word_count=word_count,
        unique_words=unique,
        avg_words_per_page=round(word_count / page_count, 1),
        chapter_count=len(chapters),
        language_distribution=lang_dist,
        readability_score=flesch_reading_ease(full),
        lexical_diversity=round(unique / max(word_count, 1), 3),
        reference_counts=dict(ref_counts),
        top_terms=top_terms,
        quality_by_page=[p.quality_score for p in pages],
        sentiment_proxy=sentiment_proxy(full),
        thematic_weights=thematic_weights(references),
    )
