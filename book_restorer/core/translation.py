"""Translation backend abstraction.

Layered strategy so the app works with zero configuration and gets better
as optional extras are available:

1. If ``OPENAI_API_KEY`` is set, use an LLM for higher quality, context aware
   translation (best for archaic/esoteric vocabulary).
2. Otherwise fall back to ``deep_translator`` (free, keyless Google Translate
   web endpoint) - requires outbound internet access.
3. If neither works (offline sandbox, rate limiting, etc.) the original text
   is returned untouched and the failure is surfaced to the caller so the
   GUI can show a clear warning instead of silently failing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from core.utils import chunk_text


@dataclass
class TranslationResult:
    text: str
    backend: str
    success: bool
    warning: Optional[str] = None


def _translate_with_openai(text: str, target_language: str, source_language: Optional[str]) -> str:
    from openai import OpenAI  # imported lazily; optional dependency

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    src = f" from {source_language}" if source_language else ""
    prompt = (
        f"Translate the following excerpt of an old book{src} into {target_language}. "
        "Preserve tone, archaic style where meaningful, and paragraph breaks. "
        "Return only the translation, no commentary.\n\n" + text
    )
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_TRANSLATE_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def _translate_with_deep_translator(text: str, target_language: str, source_language: Optional[str]) -> str:
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source=source_language or "auto", target=target_language)
    chunks = chunk_text(text, max_chars=4500)
    translated_chunks = [translator.translate(chunk) or "" for chunk in chunks]
    return "\n".join(translated_chunks)


def translate_text(
    text: str,
    target_language: str = "en",
    source_language: Optional[str] = None,
) -> TranslationResult:
    """Translate ``text`` into ``target_language`` (ISO 639-1 code)."""
    if not text or not text.strip():
        return TranslationResult(text="", backend="none", success=True)

    if os.environ.get("OPENAI_API_KEY"):
        try:
            translated = "\n".join(
                _translate_with_openai(chunk, target_language, source_language)
                for chunk in chunk_text(text, max_chars=6000)
            )
            return TranslationResult(text=translated, backend="openai", success=True)
        except Exception as exc:  # pragma: no cover - network/key dependent
            fallback_warning = f"OpenAI translation failed ({exc}); falling back to Google Translate."
    else:
        fallback_warning = None

    try:
        translated = _translate_with_deep_translator(text, target_language, source_language)
        return TranslationResult(
            text=translated,
            backend="google_translate",
            success=True,
            warning=fallback_warning,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        return TranslationResult(
            text=text,
            backend="none",
            success=False,
            warning=f"Translation unavailable ({exc}). Showing original text.",
        )


def supported_target_languages() -> List[str]:
    try:
        from deep_translator import GoogleTranslator

        return sorted(GoogleTranslator().get_supported_languages())
    except Exception:
        return ["english", "spanish", "french", "german", "italian", "portuguese", "russian"]
