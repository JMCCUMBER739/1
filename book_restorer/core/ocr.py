"""OCR wrapper around pytesseract with multi-language support."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import List, Optional

import pytesseract
from PIL import Image


class TesseractNotFoundError(RuntimeError):
    pass


@dataclass
class PageOCRResult:
    index: int
    text: str
    mean_confidence: float
    detected_lang_hint: Optional[str] = None


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def get_installed_languages() -> List[str]:
    """Return tesseract language codes installed on this system."""
    if not tesseract_available():
        return []
    try:
        langs = pytesseract.get_languages(config="")
    except Exception:
        return ["eng"]
    return [lang for lang in langs if lang != "osd"]


def ocr_image(image: Image.Image, languages: List[str], psm: int = 3) -> PageOCRResult:
    if not tesseract_available():
        raise TesseractNotFoundError(
            "Tesseract OCR binary was not found on PATH. Install it with "
            "'sudo apt-get install tesseract-ocr tesseract-ocr-<lang>'."
        )
    lang_code = "+".join(languages) if languages else "eng"
    config = f"--psm {psm}"
    try:
        text = pytesseract.image_to_string(image, lang=lang_code, config=config)
    except pytesseract.TesseractError:
        # Fall back to english if a requested language pack is missing.
        text = pytesseract.image_to_string(image, lang="eng", config=config)

    mean_conf = 0.0
    try:
        data = pytesseract.image_to_data(image, lang=lang_code, config=config, output_type=pytesseract.Output.DICT)
        confidences = [float(c) for c in data.get("conf", []) if c not in ("-1", -1)]
        if confidences:
            mean_conf = sum(confidences) / len(confidences)
    except Exception:
        pass

    return PageOCRResult(index=-1, text=text, mean_confidence=mean_conf)


def ocr_pages(images: List[Image.Image], languages: List[str], psm: int = 3) -> List[PageOCRResult]:
    results = []
    for i, image in enumerate(images):
        result = ocr_image(image, languages=languages, psm=psm)
        result.index = i
        results.append(result)
    return results
