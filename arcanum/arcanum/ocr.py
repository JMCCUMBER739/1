"""OCR layer built on Tesseract via pytesseract."""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image
import numpy as np
import pytesseract


@dataclass
class PageOCR:
    """OCR output for one page."""

    text: str
    confidence: float  # mean word confidence 0-100
    word_count: int
    lines: list[str] = field(default_factory=list)
    searchable_pdf: bytes | None = None


def _to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 3:
        return Image.fromarray(image[:, :, ::-1])
    return Image.fromarray(image)


def available_languages() -> list[str]:
    try:
        return sorted(pytesseract.get_languages(config=""))
    except Exception:
        return ["eng"]


def ocr_page(image: np.ndarray, lang: str = "eng", make_pdf: bool = True) -> PageOCR:
    """Run OCR on a cleaned page image.

    Returns extracted text, mean confidence, and (optionally) a one-page
    searchable PDF containing the image with an invisible text layer.
    """
    pil = _to_pil(image)
    config = "--oem 3 --psm 3"

    data = pytesseract.image_to_data(
        pil,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    words: list[str] = []
    confidences: list[float] = []
    line_map: dict[tuple, list[str]] = {}
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        conf = float(data["conf"][i])
        if conf < 0:
            continue
        words.append(word)
        confidences.append(conf)
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        line_map.setdefault(key, []).append(word)

    lines = [" ".join(ws) for _, ws in sorted(line_map.items())]
    text = "\n".join(lines)
    mean_conf = float(np.mean(confidences)) if confidences else 0.0

    pdf_bytes: bytes | None = None
    if make_pdf:
        pdf_bytes = pytesseract.image_to_pdf_or_hocr(
            pil,
            lang=lang,
            config=config,
            extension="pdf",
        )

    return PageOCR(
        text=text,
        confidence=round(mean_conf, 1),
        word_count=len(words),
        lines=lines,
        searchable_pdf=pdf_bytes,
    )
