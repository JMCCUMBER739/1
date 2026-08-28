"""Language identification helpers, used to pick OCR/translation/summary
settings automatically from noisy scanned text.
"""

from __future__ import annotations

from typing import Optional

from langdetect import DetectorFactory, LangDetectException, detect

# Deterministic results across runs.
DetectorFactory.seed = 0

_ISO_TO_NAME = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "la": "Latin",
    "el": "Greek",
    "ru": "Russian",
    "ar": "Arabic",
    "he": "Hebrew",
    "nl": "Dutch",
    "sv": "Swedish",
    "pl": "Polish",
}

_ISO_TO_TESSERACT = {
    "en": "eng",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "la": "lat",
    "el": "grc",
    "ru": "rus",
    "ar": "ara",
    "he": "heb",
}


def detect_language(text: str) -> Optional[str]:
    """Best-effort ISO 639-1 language code, or None if detection fails
    (e.g. text too short or pure noise from a bad OCR pass).
    """
    sample = text.strip()
    if len(sample) < 20:
        return None
    try:
        return detect(sample)
    except LangDetectException:
        return None


def language_name(iso_code: Optional[str]) -> str:
    if not iso_code:
        return "Unknown"
    return _ISO_TO_NAME.get(iso_code, iso_code.upper())


def to_tesseract_code(iso_code: Optional[str]) -> str:
    if not iso_code:
        return "eng"
    return _ISO_TO_TESSERACT.get(iso_code, "eng")
