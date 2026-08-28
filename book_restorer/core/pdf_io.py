"""PDF <-> image conversion utilities built on PyMuPDF and img2pdf.

Scanned books are treated as a sequence of raster page images: we render
each PDF page to a high-DPI image for cleaning/OCR, and re-assemble cleaned
images into a new lossless-ish PDF at the end.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import img2pdf
from PIL import Image


@dataclass
class RenderedPage:
    index: int
    image: Image.Image
    width: int
    height: int
    source_dpi: int


def get_page_count(pdf_path: Path) -> int:
    with fitz.open(pdf_path) as doc:
        return doc.page_count


def render_pdf_to_images(pdf_path: Path, dpi: int = 300) -> List[RenderedPage]:
    """Render every page of ``pdf_path`` to a PIL image at ``dpi``."""
    pages: List[RenderedPage] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(RenderedPage(index=i, image=image, width=pix.width, height=pix.height, source_dpi=dpi))
    return pages


def render_pdf_pages_iter(pdf_path: Path, dpi: int = 300):
    """Generator variant of :func:`render_pdf_to_images` for large books, so
    callers can stream page-by-page without holding the whole book in RAM.
    """
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            yield RenderedPage(index=i, image=image, width=pix.width, height=pix.height, source_dpi=dpi)


def images_to_pdf(image_paths: List[Path], output_pdf_path: Path) -> Path:
    """Assemble a list of image files into a single PDF (used for the
    "cleaned" output book).
    """
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pdf_path, "wb") as f:
        f.write(img2pdf.convert([str(p) for p in image_paths]))
    return output_pdf_path
