"""Chapter boundary detection for OCR'd book text.

Scanned books rarely carry structural metadata (no bookmarks/outline), so we
rely on textual heuristics: explicit "Chapter N" / "Book N" / "Part N"
headings, standalone Roman numerals, and large all-caps lines that look like
section titles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from core.utils import clean_whitespace

_ROMAN_RE = r"[IVXLCDM]+"
_HEADING_PATTERNS = [
    re.compile(rf"^\s*(chapter)\s+({_ROMAN_RE}|\d+|[A-Za-z]+)\.?\s*[:.\-]?\s*(.*)$", re.IGNORECASE),
    re.compile(rf"^\s*(book)\s+({_ROMAN_RE}|\d+|[A-Za-z]+)\.?\s*[:.\-]?\s*(.*)$", re.IGNORECASE),
    re.compile(rf"^\s*(part)\s+({_ROMAN_RE}|\d+|[A-Za-z]+)\.?\s*[:.\-]?\s*(.*)$", re.IGNORECASE),
    re.compile(rf"^\s*({_ROMAN_RE})\.?\s*$"),
]


@dataclass
class Chapter:
    number: int
    title: str
    text: str
    start_page: int
    end_page: int
    translated_text: str = ""
    synopsis: str = ""


@dataclass
class PageText:
    index: int
    text: str


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    for pattern in _HEADING_PATTERNS:
        if pattern.match(stripped):
            return True
    # All-caps short line (common section title rendering in OCR output).
    letters = [c for c in stripped if c.isalpha()]
    if letters and stripped.upper() == stripped and 3 <= len(letters) <= 60 and len(stripped.split()) <= 8:
        return True
    return False


def split_into_chapters(pages: List[PageText], min_chapter_chars: int = 400) -> List[Chapter]:
    """Detect chapter headings across the whole book and split accordingly.

    If no headings are found the entire book is returned as a single
    "Chapter 1" so downstream summarization/analytics always have at least
    one section to work with.
    """
    headings: List[tuple] = []  # (page_index, line_no_within_page, title)
    for page in pages:
        lines = page.text.splitlines()
        for line in lines:
            if _looks_like_heading(line):
                headings.append((page.index, line.strip()))

    full_pages_text = [clean_whitespace(p.text) for p in pages]

    if not headings:
        combined = "\n\n".join(full_pages_text)
        return [
            Chapter(
                number=1,
                title="Full Text",
                text=combined,
                start_page=pages[0].index if pages else 0,
                end_page=pages[-1].index if pages else 0,
            )
        ]

    chapters: List[Chapter] = []
    for i, (page_idx, title) in enumerate(headings):
        start_page = page_idx
        end_page = headings[i + 1][0] if i + 1 < len(headings) else pages[-1].index
        chapter_pages = [p for p in pages if start_page <= p.index <= end_page]
        text = "\n\n".join(clean_whitespace(p.text) for p in chapter_pages)
        # Drop the heading line itself from the body for a cleaner synopsis.
        text = re.sub(re.escape(title), "", text, count=1).strip()
        chapters.append(
            Chapter(
                number=len(chapters) + 1,
                title=title[:100] or f"Chapter {len(chapters) + 1}",
                text=text,
                start_page=start_page,
                end_page=end_page,
            )
        )

    # Merge trivially short "chapters" (false positive headings) into the
    # previous chapter so synopsis/reference extraction stays meaningful.
    merged: List[Chapter] = []
    for chapter in chapters:
        if merged and len(chapter.text) < min_chapter_chars:
            merged[-1].text = clean_whitespace(merged[-1].text + "\n\n" + chapter.text)
            merged[-1].end_page = chapter.end_page
        else:
            merged.append(chapter)

    for idx, chapter in enumerate(merged, start=1):
        chapter.number = idx

    return merged or chapters
