"""Chapter segmentation, synopses, and insight extraction."""

from __future__ import annotations

import re
from collections import Counter

from app.pipeline.models import Chapter, PageText

_CHAPTER_RE = re.compile(
    r"^(?:"
    r"chapter\s+([ivxlcdm\d]+|[a-z]+)"
    r"|chapt?\.?\s*([ivxlcdm\d]+)"
    r"|book\s+([ivxlcdm\d]+)"
    r"|part\s+([ivxlcdm\d]+)"
    r"|section\s+(\d+)"
    r"|libro\s+([ivxlcdm\d]+)"
    r"|chapitre\s+([ivxlcdm\d]+)"
    r"|kapitel\s+([ivxlcdm\d]+)"
    r")(?:[:.\-–—]\s*(.+))?$",
    re.IGNORECASE,
)

_TITLE_LINE = re.compile(r"^[A-Z][A-Z0-9 ,;:'\"\-]{8,80}$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _page_body(page: PageText) -> str:
    return (page.translated_text or page.cleaned_text or "").strip()


def detect_chapters(texts: list[PageText]) -> list[Chapter]:
    markers: list[tuple[int, str]] = []
    for page in texts:
        body = _page_body(page)
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        for ln in lines[:12]:  # chapter headings usually near top
            m = _CHAPTER_RE.match(ln)
            if m:
                title = ln
                markers.append((page.page_index, title))
                break
            if _TITLE_LINE.match(ln) and len(ln.split()) <= 10:
                # All-caps title heuristic — only if rare
                if page.page_index == 0 or (markers and page.page_index - markers[-1][0] > 1):
                    markers.append((page.page_index, ln.title()))
                    break

    if not markers:
        # Fall back to equal page bands
        n = len(texts)
        if n == 0:
            return []
        band = max(n // 3, 1)
        chapters: list[Chapter] = []
        for i, start in enumerate(range(0, n, band)):
            end = min(start + band - 1, n - 1)
            chunk = "\n\n".join(_page_body(t) for t in texts[start : end + 1])
            chapters.append(
                Chapter(
                    index=i + 1,
                    title=f"Section {i + 1}",
                    start_page=start,
                    end_page=end,
                    text=chunk,
                    word_count=len(chunk.split()),
                )
            )
        return chapters

    # Deduplicate consecutive same page
    dedup: list[tuple[int, str]] = []
    for m in markers:
        if not dedup or dedup[-1][0] != m[0]:
            dedup.append(m)

    chapters = []
    for i, (start, title) in enumerate(dedup):
        end = dedup[i + 1][0] - 1 if i + 1 < len(dedup) else texts[-1].page_index
        chunk = "\n\n".join(_page_body(t) for t in texts if start <= t.page_index <= end)
        chapters.append(
            Chapter(
                index=i + 1,
                title=title,
                start_page=start,
                end_page=end,
                text=chunk,
                word_count=len(chunk.split()),
            )
        )
    return chapters


def _key_sentences(text: str, limit: int = 3) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if len(s.strip()) > 40]
    if not sentences:
        return []
    words = re.findall(r"[A-Za-z']{4,}", text.lower())
    freq = Counter(words)
    stop = {
        "that", "this", "with", "from", "have", "were", "which", "their", "there",
        "would", "could", "should", "about", "into", "when", "what", "your", "them",
        "than", "then", "been", "also", "only", "some", "such", "very", "more",
        "most", "other", "upon", "shall", "will", "these", "those", "being",
    }
    scored: list[tuple[float, str]] = []
    for s in sentences:
        tokens = re.findall(r"[A-Za-z']{4,}", s.lower())
        score = sum(freq[t] for t in tokens if t not in stop) / (len(tokens) + 1)
        score += min(len(s) / 300.0, 0.5)
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    # Preserve reading order of top picks
    top = {s for _, s in scored[:limit]}
    return [s for s in sentences if s in top][:limit]


def synthesize_synopsis(chapter: Chapter) -> str:
    sentences = _key_sentences(chapter.text, limit=3)
    if not sentences:
        preview = chapter.text[:280].strip()
        return preview + ("…" if len(chapter.text) > 280 else "")
    return " ".join(sentences)


def extract_insights(chapter: Chapter) -> list[str]:
    insights: list[str] = []
    text = chapter.text
    words = re.findall(r"[A-Za-z']{4,}", text.lower())
    if not words:
        return ["Insufficient text for insight extraction."]

    freq = Counter(w for w in words if len(w) > 5)
    top = [w for w, _ in freq.most_common(5)]
    if top:
        insights.append(f"Dominant themes: {', '.join(top)}.")

    questions = len(re.findall(r"\?", text))
    if questions:
        insights.append(f"Contains {questions} interrogative passage(s), suggesting dialectical or catechetical structure.")

    quotes = len(re.findall(r"[\"'].{12,}?[\"']", text))
    if quotes:
        insights.append(f"Identified ~{quotes} quoted or cited fragment(s).")

    long_sents = [s for s in _SENTENCE_SPLIT.split(text) if len(s.split()) > 35]
    if long_sents:
        insights.append("Prose leans toward long, periodic sentences typical of classical or scholastic style.")

    if chapter.word_count:
        density = len(set(words)) / max(len(words), 1)
        if density > 0.55:
            insights.append("High lexical diversity — exploratory or encyclopedic register.")
        elif density < 0.35:
            insights.append("Repetitive lexicon — didactic, liturgical, or formulaic register.")

    return insights[:5] or ["No strong structural insights detected."]


def enrich_chapters(chapters: list[Chapter]) -> list[Chapter]:
    for ch in chapters:
        ch.synopsis = synthesize_synopsis(ch)
        ch.insights = extract_insights(ch)
        ch.word_count = len(ch.text.split())
    return chapters
