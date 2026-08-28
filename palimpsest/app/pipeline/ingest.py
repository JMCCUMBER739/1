"""PDF ingestion and page rasterization."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import numpy as np
from PIL import Image

from app.pipeline.models import PageImage


def open_pdf(path: Path) -> pymupdf.Document:
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError("Input must be a PDF file")
    return pymupdf.open(path)


def page_count(path: Path) -> int:
    with open_pdf(path) as doc:
        return doc.page_count


def render_pages(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 200,
    max_pages: int | None = None,
) -> list[PageImage]:
    """Rasterize each PDF page to PNG under output_dir/images/original."""
    original_dir = output_dir / "images" / "original"
    original_dir.mkdir(parents=True, exist_ok=True)

    pages: list[PageImage] = []
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)

    with open_pdf(pdf_path) as doc:
        total = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
        for i in range(total):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out_path = original_dir / f"page_{i + 1:04d}.png"
            pix.save(str(out_path))
            pages.append(
                PageImage(
                    page_index=i,
                    original_path=out_path,
                    width=pix.width,
                    height=pix.height,
                )
            )
    return pages


def load_image_array(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def extract_embedded_text(pdf_path: Path, max_pages: int | None = None) -> list[str]:
    """Pull any existing text layer (useful for born-digital or already-OCR'd PDFs)."""
    texts: list[str] = []
    with open_pdf(pdf_path) as doc:
        total = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
        for i in range(total):
            texts.append(doc.load_page(i).get_text("text") or "")
    return texts
