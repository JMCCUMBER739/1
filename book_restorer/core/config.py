"""Configuration objects shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
LEXICON_DIR = PACKAGE_ROOT / "data" / "lexicons"

REFERENCE_CATEGORIES = ("esoteric", "theological", "scientific")

#: Languages we ship curated stopword-free NLP support for out of the box.
#: Anything else still works for OCR/translation, just with lighter-weight
#: heuristics for summarization/analytics.
SUPPORTED_SUMMARY_LANGUAGES = {
    "en": "english",
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
}

#: Mapping of human-friendly language names to the ISO codes used by both
#: langdetect and deep-translator, plus the Tesseract 3-letter code.
LANGUAGE_TABLE = {
    "English": ("en", "eng"),
    "German": ("de", "deu"),
    "French": ("fr", "fra"),
    "Spanish": ("es", "spa"),
    "Italian": ("it", "ita"),
    "Portuguese": ("pt", "por"),
    "Latin": ("la", "lat"),
    "Ancient Greek": ("el", "grc"),
    "Russian": ("ru", "rus"),
    "Arabic": ("ar", "ara"),
    "Hebrew": ("he", "heb"),
}


@dataclass
class CleaningOptions:
    """Controls for the scanned-page image restoration pipeline."""

    dpi: int = 300
    denoise: bool = True
    denoise_strength: int = 8
    deskew: bool = True
    enhance_contrast: bool = True
    clahe_clip_limit: float = 2.5
    binarize: bool = True
    adaptive_block_size: int = 35
    adaptive_c: int = 15
    remove_speckles: bool = True
    crop_borders: bool = True
    border_margin_px: int = 8


@dataclass
class OCROptions:
    languages: List[str] = field(default_factory=lambda: ["eng"])
    auto_detect_language: bool = True
    psm: int = 3  # tesseract page segmentation mode


@dataclass
class TranslationOptions:
    enabled: bool = False
    target_language: str = "en"
    translate_chapters_individually: bool = True


@dataclass
class AnalysisOptions:
    summary_sentence_count: int = 6
    top_keywords: int = 20
    reference_categories: List[str] = field(default_factory=lambda: list(REFERENCE_CATEGORIES))
    reference_context_chars: int = 320
    use_llm_if_available: bool = True


@dataclass
class PipelineOptions:
    output_dir: Path
    cleaning: CleaningOptions = field(default_factory=CleaningOptions)
    ocr: OCROptions = field(default_factory=OCROptions)
    translation: TranslationOptions = field(default_factory=TranslationOptions)
    analysis: AnalysisOptions = field(default_factory=AnalysisOptions)
    book_title: Optional[str] = None
    keep_intermediate_pages: bool = True
