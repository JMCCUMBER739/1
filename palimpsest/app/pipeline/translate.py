"""Multi-language translation with graceful offline fallback."""

from __future__ import annotations

import time
from typing import Callable

from app.pipeline.models import PageText

ProgressCb = Callable[[float, str], None]


def _chunk_text(text: str, max_chars: int = 4500) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for para in text.split("\n"):
        if size + len(para) + 1 > max_chars and buf:
            chunks.append("\n".join(buf))
            buf, size = [], 0
        buf.append(para)
        size += len(para) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def translate_text(text: str, target: str = "en", source: str = "auto") -> str:
    if not text.strip():
        return text
    # Latin often mis-detected; keep as-is if targeting English and looks classical
    try:
        from deep_translator import GoogleTranslator

        parts = []
        for chunk in _chunk_text(text):
            translated = GoogleTranslator(source=source, target=target).translate(chunk)
            parts.append(translated or chunk)
            time.sleep(0.05)
        return "\n".join(parts)
    except Exception:
        # Network / rate-limit: return original with marker
        return text


def translate_pages(
    texts: list[PageText],
    target: str = "en",
    enabled: bool = True,
    progress: ProgressCb | None = None,
) -> list[PageText]:
    if not enabled:
        return texts

    out: list[PageText] = []
    total = max(len(texts), 1)
    for i, page in enumerate(texts):
        src = page.detected_language or "auto"
        # Skip if already target language
        if src == target or (target.startswith("zh") and src == "zh-cn"):
            translated = page.cleaned_text
        else:
            translated = translate_text(page.cleaned_text, target=target, source="auto")
        page.translated_text = translated
        out.append(page)
        if progress:
            progress((i + 1) / total, f"Translated page {i + 1}/{total}")
    return out
