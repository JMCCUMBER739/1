"""PDF input/output built on PyMuPDF.

Responsibilities: render scanned pages to images, extract any embedded
images in their original format, and assemble the cleaned, searchable
output PDF (cleaned page image + invisible OCR text layer).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator

from PIL import Image
import numpy as np
import pymupdf


def open_pdf(path: str | Path) -> pymupdf.Document:
    return pymupdf.open(str(path))


def page_count(path: str | Path) -> int:
    with pymupdf.open(str(path)) as doc:
        return doc.page_count


def render_page(doc: pymupdf.Document, index: int, dpi: int = 300) -> np.ndarray:
    """Render one page to a grayscale-capable BGR numpy array."""
    page = doc[index]
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB)
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = arr.reshape(pix.height, pix.width, pix.n)
    return arr[:, :, ::-1].copy()  # RGB -> BGR for OpenCV


def iter_pages(doc: pymupdf.Document, dpi: int = 300) -> Iterator[tuple[int, np.ndarray]]:
    for i in range(doc.page_count):
        yield i, render_page(doc, i, dpi=dpi)


def extract_embedded_images(pdf_path: str | Path, out_dir: str | Path) -> list[Path]:
    """Extract every embedded raster image in its original encoded format."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    with pymupdf.open(str(pdf_path)) as doc:
        seen: set[int] = set()
        for page_index in range(doc.page_count):
            for img in doc[page_index].get_images(full=True):
                xref = img[0]
                if xref in seen:
                    continue
                seen.add(xref)
                info = doc.extract_image(xref)
                ext = info["ext"]
                name = f"page_{page_index + 1:04d}_img_{xref}.{ext}"
                path = out_dir / name
                path.write_bytes(info["image"])
                saved.append(path)
    return saved


def save_image_png(image: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 3:
        pil = Image.fromarray(image[:, :, ::-1])
    else:
        pil = Image.fromarray(image)
    pil.save(str(path))


def build_searchable_pdf(page_pdf_bytes: list[bytes], out_path: str | Path) -> None:
    """Merge per-page searchable PDFs (from tesseract) into one document."""
    out = pymupdf.open()
    for chunk in page_pdf_bytes:
        with pymupdf.open(stream=chunk, filetype="pdf") as part:
            out.insert_pdf(part)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(str(out_path), deflate=True, garbage=3)
    out.close()


def build_image_pdf(images: list[np.ndarray], out_path: str | Path, dpi: int = 300) -> None:
    """Fallback: build a plain (non-searchable) PDF from page images."""
    pil_pages = []
    for image in images:
        if image.ndim == 3:
            pil_pages.append(Image.fromarray(image[:, :, ::-1]).convert("RGB"))
        else:
            pil_pages.append(Image.fromarray(image).convert("RGB"))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    pil_pages[0].save(buf, format="PDF", save_all=True, append_images=pil_pages[1:], resolution=dpi)
    Path(out_path).write_bytes(buf.getvalue())
