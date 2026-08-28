"""Esoteric, theological, and scientific reference extraction."""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.pipeline.models import Chapter, PageText, ReferenceHit, ReferenceKind

_LEXICON_DIR = Path(__file__).resolve().parent.parent / "lexicons"


def _load_lexicon(name: str) -> dict[str, list[str]]:
    path = _LEXICON_DIR / f"{name}.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("categories", {})


def _excerpt_around(text: str, start: int, end: int, radius: int = 120) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    snippet = text[left:right].replace("\n", " ")
    snippet = re.sub(r"\s+", " ", snippet).strip()
    if left > 0:
        snippet = "…" + snippet
    if right < len(text):
        snippet = snippet + "…"
    return snippet


def _chapter_for_page(chapters: list[Chapter], page_index: int) -> str | None:
    for ch in chapters:
        if ch.start_page <= page_index <= ch.end_page:
            return ch.title
    return None


def find_references(
    texts: list[PageText],
    chapters: list[Chapter] | None = None,
) -> list[ReferenceHit]:
    chapters = chapters or []
    lexicons = {
        ReferenceKind.ESOTERIC: _load_lexicon("esoteric"),
        ReferenceKind.THEOLOGICAL: _load_lexicon("theological"),
        ReferenceKind.SCIENTIFIC: _load_lexicon("scientific"),
    }

    hits: list[ReferenceHit] = []
    seen: set[tuple[str, str, int]] = set()

    for page in texts:
        body = page.translated_text or page.cleaned_text or ""
        if not body.strip():
            continue
        lower = body.lower()
        for kind, categories in lexicons.items():
            for category, terms in categories.items():
                for term in terms:
                    term_l = term.lower()
                    for m in re.finditer(re.escape(term_l), lower):
                        key = (kind.value, term_l, page.page_index)
                        if key in seen:
                            continue
                        seen.add(key)
                        excerpt = _excerpt_around(body, m.start(), m.end())
                        hits.append(
                            ReferenceHit(
                                kind=kind,
                                category=category,
                                term=term,
                                excerpt=excerpt,
                                page_index=page.page_index,
                                chapter_title=_chapter_for_page(chapters, page.page_index),
                                context_score=1.0 + min(len(term.split()) * 0.2, 1.0),
                            )
                        )
    hits.sort(key=lambda h: (h.kind.value, h.page_index, h.term))
    return hits
