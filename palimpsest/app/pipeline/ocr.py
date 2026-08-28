"""OCR and text normalization."""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from app.pipeline.ingest import extract_embedded_text
from app.pipeline.models import PageImage, PageText

_WS_RE = re.compile(r"[ \t]+")
_HYPHEN_RE = re.compile(r"(\w)-\n(\w)")
_MULTI_NL = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_RE.sub(r"\1\2", text)
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    # Fix common OCR ligature / scan artifacts
    replacements = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "—": "-",
        "–": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.strip()


def _tesseract_available() -> bool:
    try:
        import shutil

        import pytesseract

        return shutil.which("tesseract") is not None and pytesseract.get_tesseract_version() is not None
    except Exception:
        return False


def ocr_image(path: Path, lang: str = "eng") -> tuple[str, float]:
    """OCR a single image. Returns (text, confidence 0–100)."""
    if not _tesseract_available():
        return "", 0.0
    import pytesseract

    with Image.open(path) as im:
        data = pytesseract.image_to_data(im, lang=lang, output_type=pytesseract.Output.DICT)
        text = pytesseract.image_to_string(im, lang=lang)
    confs = [float(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit() and float(c) >= 0]
    confidence = sum(confs) / len(confs) if confs else 0.0
    return text, confidence


def detect_language(text: str) -> str | None:
    sample = (text or "").strip()
    if len(sample) < 40:
        return None
    try:
        from langdetect import detect

        return detect(sample)
    except Exception:
        return None


def extract_texts(
    pages: list[PageImage],
    pdf_path: Path,
    ocr_lang: str = "eng",
    enable_ocr: bool = True,
    max_pages: int | None = None,
) -> list[PageText]:
    embedded = extract_embedded_text(pdf_path, max_pages=max_pages)
    results: list[PageText] = []

    for page in pages:
        idx = page.page_index
        embedded_text = embedded[idx] if idx < len(embedded) else ""
        raw = embedded_text.strip()
        confidence = 95.0 if len(raw) > 40 else 0.0

        if enable_ocr and len(raw) < 40:
            img_path = page.cleaned_path or page.original_path
            ocr_text, confidence = ocr_image(img_path, lang=ocr_lang)
            raw = ocr_text if ocr_text.strip() else raw

        cleaned = normalize_text(raw)
        lang = detect_language(cleaned)
        results.append(
            PageText(
                page_index=idx,
                raw_text=raw,
                cleaned_text=cleaned,
                detected_language=lang,
                confidence=confidence,
            )
        )
    return results
