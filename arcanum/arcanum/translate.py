"""Multi-language translation with automatic source detection.

Uses deep-translator's Google backend (no API key required). Text is
translated in sentence-aligned chunks below the service limit. Failures
(e.g. no network) degrade gracefully: the pipeline reports the problem
and continues with everything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from deep_translator import GoogleTranslator

# Curated display list for the GUI; any ISO code accepted by the backend works.
SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "pl": "Polish",
    "sv": "Swedish",
    "el": "Greek",
    "la": "Latin",
    "iw": "Hebrew",
    "yi": "Yiddish",
    "ar": "Arabic",
    "sa": "Sanskrit",
    "hi": "Hindi",
    "zh-CN": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "tr": "Turkish",
}

# Well-known source languages for old books, offered in the GUI/CLI.
# "auto" lets the backend detect the language per chunk.
SOURCE_LANGUAGES: dict[str, str] = {
    "auto": "Auto-detect",
    **SUPPORTED_LANGUAGES,
}

# Codes the backend spells differently from common usage.
_CODE_ALIASES = {"he": "iw", "zh": "zh-CN", "grc": "el"}

# Languages with no machine-translation support in the backend. OCR can
# still read them (e.g. Tesseract `syr` for Syriac-script Aramaic); we
# fail with a clear message instead of a confusing backend error.
UNSUPPORTED_SOURCES = {
    "arc": "Aramaic",
    "syr": "Syriac",
    "cop": "Coptic",
}


def normalize_code(code: str) -> str:
    return _CODE_ALIASES.get(code, code)


_CHUNK_LIMIT = 4200


@dataclass
class TranslationResult:
    language: str
    language_name: str
    ok: bool
    text: str = ""
    error: str = ""
    chapter_summaries: dict[str, str] = field(default_factory=dict)


def _chunk_text(text: str, limit: int = _CHUNK_LIMIT) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in text.split("\n"):
        # Never exceed the limit even for a single huge paragraph.
        while len(paragraph) > limit:
            head, paragraph = paragraph[:limit], paragraph[limit:]
            if current:
                chunks.append("\n".join(current))
                current, size = [], 0
            chunks.append(head)
        if size + len(paragraph) + 1 > limit and current:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph) + 1
    if current:
        chunks.append("\n".join(current))
    return [c for c in chunks if c.strip()]


_ERROR_MARKERS = ("that’s an error", "that's an error", "error 500", "error 502", "error 503")


def _looks_like_error_page(text: str) -> bool:
    head = text[:300].lower()
    return any(marker in head for marker in _ERROR_MARKERS)


def _translate_chunk(translator: GoogleTranslator, chunk: str, attempts: int = 6) -> str:
    """Translate one chunk with retries; the service occasionally 500s."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            result = translator.translate(chunk) or ""
            if not _looks_like_error_page(result):
                return result
            last_error = RuntimeError("service returned an error page")
        except Exception as exc:
            last_error = exc
        time.sleep(2.0 * (attempt + 1))
    raise last_error or RuntimeError("translation failed")


def translate_text(text: str, target: str, source: str = "auto", progress=None) -> TranslationResult:
    """Translate `text` into `target`; `progress(done, total)` is optional."""
    target = normalize_code(target)
    source = normalize_code(source)
    name = SUPPORTED_LANGUAGES.get(target, target)
    if not text.strip():
        return TranslationResult(target, name, ok=True, text="")
    if source in UNSUPPORTED_SOURCES:
        return TranslationResult(
            target,
            name,
            ok=False,
            error=(
                f"{UNSUPPORTED_SOURCES[source]} is not supported by the "
                f"translation backend. The text can still be OCR-ed and "
                f"analyzed; consider specialist tooling for translation."
            ),
        )
    try:
        translator = GoogleTranslator(source=source, target=target)
        chunks = _chunk_text(text)
        out: list[str] = []
        for i, chunk in enumerate(chunks):
            out.append(_translate_chunk(translator, chunk))
            if progress:
                progress(i + 1, len(chunks))
        return TranslationResult(target, name, ok=True, text="\n".join(out))
    except Exception as exc:  # network failure, quota, bad code, ...
        return TranslationResult(target, name, ok=False, error=f"{type(exc).__name__}: {exc}")
