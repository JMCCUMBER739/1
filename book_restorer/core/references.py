"""Esoteric / theological / scientific reference detection with excerpts.

Each category is backed by a curated lexicon in ``data/lexicons``. Matching
is whole-word/phrase, case-insensitive, and returns the surrounding sentence
(or a character window as fallback) as the citable excerpt, along with the
page number where the excerpt was found.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

from core.config import LEXICON_DIR, REFERENCE_CATEGORIES
from core.utils import load_lexicon

_LEXICON_FILES = {
    "esoteric": LEXICON_DIR / "esoteric_terms.txt",
    "theological": LEXICON_DIR / "theological_terms.txt",
    "scientific": LEXICON_DIR / "scientific_terms.txt",
}


@dataclass
class Reference:
    category: str
    term: str
    excerpt: str
    page: int
    chapter_title: str = ""


@lru_cache(maxsize=None)
def _compiled_patterns(category: str):
    terms = load_lexicon(_LEXICON_FILES[category])
    # Longer phrases first so multi-word matches win over single-word
    # substrings (e.g. "sacred geometry" before "sacred").
    terms = sorted(set(terms), key=len, reverse=True)
    patterns = []
    for term in terms:
        escaped = re.escape(term)
        pattern = re.compile(rf"(?<![\w]){escaped}(?![\w])", re.IGNORECASE)
        patterns.append((term, pattern))
    return patterns


def available_categories() -> List[str]:
    return list(REFERENCE_CATEGORIES)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _extract_excerpt(text: str, match_start: int, match_end: int, context_chars: int) -> str:
    # Prefer full-sentence context; fall back to a character window.
    window_start = max(0, match_start - context_chars)
    window_end = min(len(text), match_end + context_chars)
    window = text[window_start:window_end]

    sentences = _SENTENCE_SPLIT_RE.split(window)
    if len(sentences) > 1:
        # Find the sentence(s) actually containing the match by re-scanning.
        cursor = 0
        for sentence in sentences:
            s_start = window.find(sentence, cursor)
            s_end = s_start + len(sentence)
            cursor = s_end
            abs_start = window_start + s_start
            abs_end = window_start + s_end
            if abs_start <= match_start <= abs_end or abs_start <= match_end <= abs_end:
                return sentence.strip()
    prefix = "..." if window_start > 0 else ""
    suffix = "..." if window_end < len(text) else ""
    return f"{prefix}{window.strip()}{suffix}"


def find_references_in_text(
    text: str,
    category: str,
    page: int = 0,
    chapter_title: str = "",
    context_chars: int = 320,
    max_matches: int = 200,
) -> List[Reference]:
    results: List[Reference] = []
    if not text.strip():
        return results
    covered = [False] * len(text)
    for term, pattern in _compiled_patterns(category):
        for match in pattern.finditer(text):
            if any(covered[match.start(): match.end()]):
                continue
            for i in range(match.start(), match.end()):
                covered[i] = True
            excerpt = _extract_excerpt(text, match.start(), match.end(), context_chars)
            results.append(
                Reference(
                    category=category,
                    term=term,
                    excerpt=excerpt,
                    page=page,
                    chapter_title=chapter_title,
                )
            )
            if len(results) >= max_matches:
                return results
    return results


def find_all_references(
    pages_text: Dict[int, str],
    categories: List[str],
    context_chars: int = 320,
) -> Dict[str, List[Reference]]:
    """Scan every page for every requested category."""
    out: Dict[str, List[Reference]] = {c: [] for c in categories}
    for category in categories:
        for page_idx, text in pages_text.items():
            out[category].extend(
                find_references_in_text(text, category=category, page=page_idx, context_chars=context_chars)
            )
    return out
