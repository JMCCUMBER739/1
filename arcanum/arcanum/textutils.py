"""Lightweight, dependency-free text utilities (tokenizing, sentences)."""

from __future__ import annotations

import re

STOPWORDS: frozenset[str] = frozenset("""
a about above after again against all am an and any are as at be because
been before being below between both but by can cannot could did do does
doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just
me more most my myself no nor not now of off on once only or other our
ours ourselves out over own same she should so some such than that the
their theirs them themselves then there these they this those through to
too under until up upon very was we were what when where which while who
whom why will with would you your yours yourself yourselves shall unto
thee thou thy hath doth ye may might must one two also yet even said
""".split())

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u00c0-\u00dc\"'])")
_WORD_RE = re.compile(r"[A-Za-z\u00c0-\u024f']+")


def normalize_ocr_text(text: str) -> str:
    """Repair common OCR artifacts: hyphenated line breaks, stray gaps."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    flat = re.sub(r"\s*\n\s*", " ", text)
    parts = _SENTENCE_RE.split(flat)
    return [p.strip() for p in parts if len(p.strip()) > 2]


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def content_words(text: str) -> list[str]:
    return [w for w in tokenize(text) if w not in STOPWORDS and len(w) > 2]


def count_syllables(word: str) -> int:
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def flesch_reading_ease(text: str) -> float:
    sentences = split_sentences(text)
    words = tokenize(text)
    if not sentences or not words:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
    score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words))
    return round(max(0.0, min(120.0, score)), 1)


def reading_level_label(score: float) -> str:
    if score >= 80:
        return "Easy (general audience)"
    if score >= 60:
        return "Standard (conversational)"
    if score >= 40:
        return "Moderately difficult (literary)"
    if score >= 20:
        return "Difficult (academic)"
    return "Very difficult (scholarly / archaic)"
