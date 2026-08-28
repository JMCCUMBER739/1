"""Extractive synopsis generation, per chapter.

Sentences are scored with a frequency-based centrality model (word
frequencies normalized against the chapter's vocabulary), boosted for
early position and penalized for extreme lengths; the top sentences are
re-ordered by appearance to form a coherent abstract.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .chapters import Chapter
from .textutils import content_words, split_sentences, tokenize


@dataclass
class ChapterSynopsis:
    chapter_number: int
    chapter_title: str
    start_page: int
    end_page: int
    word_count: int
    summary: str
    key_sentences: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


def _score_sentences(sentences: list[str], frequencies: Counter) -> list[float]:
    if not frequencies:
        return [0.0] * len(sentences)
    top = max(frequencies.values())
    scores: list[float] = []
    total = len(sentences)
    for index, sentence in enumerate(sentences):
        words = content_words(sentence)
        if not words:
            scores.append(0.0)
            continue
        base = sum(frequencies[w] / top for w in words) / len(words)
        position_bonus = 0.15 if index < max(3, total // 10) else 0.0
        length = len(tokenize(sentence))
        length_penalty = 0.25 if (length < 6 or length > 60) else 0.0
        scores.append(base + position_bonus - length_penalty)
    return scores


def summarize_text(text: str, max_sentences: int = 5) -> tuple[str, list[str], list[str]]:
    """Return (summary paragraph, key sentences, keywords) for a text."""
    sentences = split_sentences(text)
    if not sentences:
        return "", [], []
    frequencies = Counter(content_words(text))
    scores = _score_sentences(sentences, frequencies)

    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    chosen = sorted(ranked[:max_sentences])
    picked = [sentences[i] for i in chosen]
    keywords = [w for w, _ in frequencies.most_common(10)]
    return " ".join(picked), picked, keywords


def build_synopses(chapters: list[Chapter], max_sentences: int = 5) -> list[ChapterSynopsis]:
    result: list[ChapterSynopsis] = []
    for chapter in chapters:
        summary, key_sentences, keywords = summarize_text(chapter.text, max_sentences=max_sentences)
        result.append(
            ChapterSynopsis(
                chapter_number=chapter.number,
                chapter_title=chapter.title,
                start_page=chapter.start_page,
                end_page=chapter.end_page,
                word_count=len(tokenize(chapter.text)),
                summary=summary,
                key_sentences=key_sentences,
                keywords=keywords,
            )
        )
    return result
