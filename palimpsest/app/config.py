"""Application configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PALIMPSEST_", env_file=".env", extra="ignore")

    app_name: str = "Palimpsest"
    default_dpi: int = 200
    max_pages: int = 500
    ocr_lang: str = "eng"
    translate_target: str = "en"
    output_root: Path = Field(default_factory=lambda: Path("outputs"))
    clean_mode: Literal["auto", "gentle", "aggressive"] = "auto"
    enable_translation: bool = True
    enable_ocr: bool = True
    theme_accent: str = "#C45C26"
    theme_ink: str = "#1A1612"
    theme_paper: str = "#F7F1E8"
    theme_slate: str = "#2C3E50"
    theme_sage: str = "#4A6B5C"


SETTINGS = Settings()

SUPPORTED_TRANSLATE_LANGS = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "la": "Latin",
    "el": "Greek",
    "he": "Hebrew",
    "ar": "Arabic",
    "ru": "Russian",
    "zh-CN": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "sv": "Swedish",
    "tr": "Turkish",
    "hi": "Hindi",
}

OCR_LANG_MAP = {
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
    "it": "ita",
    "pt": "por",
    "la": "lat",
    "el": "ell",
    "he": "heb",
    "ar": "ara",
    "ru": "rus",
    "zh-CN": "chi_sim",
    "ja": "jpn",
    "ko": "kor",
    "nl": "nld",
    "pl": "pol",
    "sv": "swe",
    "tr": "tur",
    "hi": "hin",
}
