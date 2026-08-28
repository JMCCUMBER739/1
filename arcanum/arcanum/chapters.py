"""Chapter detection from OCR text.

Detects heading lines such as "CHAPTER IV", "Chapter Nine", "BOOK II",
"PART THE FIRST" or bare roman numerals, using their page positions to
segment the book. Falls back to even segmentation when a book carries no
detectable chapter structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass
class Chapter:
    number: int
    title: str
    start_page: int  # 1-based, inclusive
    end_page: int  # 1-based, inclusive
    text: str = ""
    pages: list[int] = field(default_factory=list)


_NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    "thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth"
)

_HEADING_PATTERNS = [
    re.compile(
        rf"^\s*(chapter|book|part|canto|section)\s+" rf"([IVXLCDM]+|\d+|{_NUMBER_WORDS})\b[.:\-\s]*(.{{0,80}})$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*([IVXLCDM]{1,7})[.\s]\s*(.{0,80})$"),
]


def _match_heading(line: str) -> str | None:
    stripped = line.strip()
    if not (3 <= len(stripped) <= 90):
        return None
    m = _HEADING_PATTERNS[0].match(stripped)
    if m:
        return stripped
    m = _HEADING_PATTERNS[1].match(stripped)
    if m and stripped.isupper():
        return stripped
    return None


def detect_chapters(page_texts: list[str], fallback_segments: int = 6) -> list[Chapter]:
    """Segment a book into chapters given per-page OCR text."""
    headings: list[tuple[int, str]] = []  # (page index, title)
    for page_index, text in enumerate(page_texts):
        # Headings live near the top of a page; inspect the first lines.
        for line in text.splitlines()[:8]:
            title = _match_heading(line)
            if title:
                headings.append((page_index, title))
                break

    total_pages = len(page_texts)
    chapters: list[Chapter] = []

    if len(headings) >= 2:
        for i, (page_index, title) in enumerate(headings):
            end = headings[i + 1][0] - 1 if i + 1 < len(headings) else total_pages - 1
            end = max(end, page_index)
            chapters.append(
                Chapter(
                    number=i + 1,
                    title=re.sub(r"\s+", " ", title).strip(" .:-"),
                    start_page=page_index + 1,
                    end_page=end + 1,
                )
            )
        # Front matter before the first heading becomes a preface chapter.
        if headings[0][0] > 0:
            chapters.insert(
                0,
                Chapter(
                    number=0,
                    title="Front Matter",
                    start_page=1,
                    end_page=headings[0][0],
                ),
            )
    else:
        segments = min(fallback_segments, max(1, total_pages))
        size = max(1, round(total_pages / segments))
        start = 0
        number = 1
        while start < total_pages:
            end = min(start + size - 1, total_pages - 1)
            if total_pages - end <= size // 2:
                end = total_pages - 1
            chapters.append(
                Chapter(
                    number=number,
                    title=f"Section {number}",
                    start_page=start + 1,
                    end_page=end + 1,
                )
            )
            number += 1
            start = end + 1

    for chapter in chapters:
        chapter.pages = list(range(chapter.start_page, chapter.end_page + 1))
        chapter.text = "\n".join(page_texts[p - 1] for p in chapter.pages).strip()

    return [c for c in chapters if c.text]
