"""Generate a synthetic "old scanned book" PDF for demoing/testing Book Restorer.

Produces a multi-chapter PDF with aged-paper texture, slight per-page skew,
and speckle noise, so the cleaning pipeline has something real to do. The
text is written to intentionally include esoteric, theological, and
scientific vocabulary so the reference-detection features have material to
find.

Usage:
    python examples/generate_sample_scan.py [output_path]
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

CHAPTERS = [
    (
        "CHAPTER I: THE ALCHEMIST'S JOURNAL",
        """In the beginning the old alchemist labored in secret, seeking the philosopher's stone
        through long nights of patient transmutation. He wrote often of hermeticism and the sacred
        geometry hidden within the great work, believing that as above so below, the microcosm
        mirrored the macrocosm in every alchemical symbol he inscribed upon the wall. His grimoire
        spoke of the astral plane and the kundalini rising like a serpent of fire, while the tarot
        cards he laid out each evening whispered of hidden knowledge yet to be revealed. It was a
        wonderful and hopeful beginning to a long and winding study, one that would occupy his mind
        for the whole of his remaining years.""",
    ),
    (
        "CHAPTER II: THE PROPHET'S SERMON",
        """The prophet rose before the congregation and spoke of salvation, of divine grace, and of
        the covenant renewed between god and his people. He reminded them of the resurrection and
        the promise of eternal life, urging repentance and faith above all worldly things. The
        gospel, he said, was a light unto every soul lost in sin, and the church stood as a
        sanctuary of doctrine and sacrament for all who sought it. Yet even in the sermon there was
        a terrible warning of judgment day, a darker note that unsettled many in the pews that
        morning.""",
    ),
    (
        "CHAPTER III: THE NATURALIST'S HYPOTHESIS",
        """The naturalist proposed a bold hypothesis regarding atomic structure, arguing that
        careful experiment and observation, not superstition, would reveal the true laws of nature.
        He described the periodic table, the behavior of energy under thermodynamics, and the slow
        march of evolution through natural selection. His scientific method demanded rigor: every
        theorem tested, every equation checked against the data of the laboratory. It was, in his
        view, a wonderful example of how the scientific method could illuminate even the most
        mysterious corners of the natural world.""",
    ),
]


def _aged_background(width: int, height: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.normal(228, 6, (height, width)).astype(np.uint8)
    img = Image.fromarray(base, mode="L").convert("RGB")
    # Sepia tint.
    arr = np.array(img).astype(np.float32)
    arr[:, :, 0] *= 1.02
    arr[:, :, 1] *= 0.97
    arr[:, :, 2] *= 0.85
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(0.6))


def _add_speckle_noise(img: Image.Image, seed: int, amount: int = 400) -> Image.Image:
    rng = np.random.default_rng(seed + 1)
    draw = ImageDraw.Draw(img)
    for _ in range(amount):
        x, y = rng.integers(0, img.width), rng.integers(0, img.height)
        shade = int(rng.integers(80, 180))
        draw.point((x, y), fill=(shade, shade, shade))
    return img


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def make_page(title: str, body: str, page_number: int, seed: int) -> Image.Image:
    width, height = 1275, 1650  # ~8.5x11in @150dpi
    img = _aged_background(width, height, seed)
    draw = ImageDraw.Draw(img)

    title_font = _load_font(34)
    body_font = _load_font(22)

    y = 90
    if title:
        for line in textwrap.wrap(title, width=40):
            draw.text((90, y), line, font=title_font, fill=(25, 20, 15))
            y += 46
        y += 30

    for paragraph in body.split("\n\n") if body else []:
        wrapped = textwrap.wrap(" ".join(paragraph.split()), width=78)
        for line in wrapped:
            draw.text((90, y), line, font=body_font, fill=(35, 28, 20))
            y += 32
        y += 20

    draw.text((width // 2 - 10, height - 60), str(page_number), font=body_font, fill=(60, 50, 40))

    img = _add_speckle_noise(img, seed)
    angle = ((seed % 7) - 3) * 0.6  # small deterministic skew per page
    img = img.rotate(angle, expand=False, fillcolor=(228, 221, 204))
    return img


def build_sample_pdf(output_path: Path) -> Path:
    pages = []
    page_number = 1
    for title, body in CHAPTERS:
        pages.append(make_page(title, "", page_number, seed=page_number))
        page_number += 1
        # Split long chapter body across a couple of pages for realism.
        words = body.split()
        mid = len(words) // 2
        first_half = " ".join(words[:mid])
        second_half = " ".join(words[mid:])
        pages.append(make_page("", first_half, page_number, seed=page_number))
        page_number += 1
        pages.append(make_page("", second_half, page_number, seed=page_number))
        page_number += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Embed correct DPI metadata so the resulting PDF pages have realistic
    # physical dimensions (Letter size), matching how real scanners /
    # img2pdf produce scanned-book PDFs. Without this, PDF viewers/renderers
    # treat the raw pixel image as if it were rendered at 72 DPI, making the
    # "page" far larger than a real sheet of paper.
    pages[0].save(output_path, save_all=True, append_images=pages[1:], resolution=150.0)
    return output_path


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "sample_old_book.pdf"
    result = build_sample_pdf(out)
    print(f"Wrote sample scanned book PDF to: {result}")
