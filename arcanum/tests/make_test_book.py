"""Generate a synthetic 'aged scan' book PDF for end-to-end testing.

Renders chapters of period-flavored text to images, then degrades them
like an old scan: sepia paper tone, uneven illumination, gaussian noise,
slight rotation. Output is an image-only PDF (no text layer), exactly
like a real scanned book.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np

CHAPTERS = [
    (
        "CHAPTER I",
        "Of the Hermetic Art",
        """
The alchemist labours nightly in his laboratory, seeking the
philosopher's stone by which base metals may suffer transmutation
into gold. The hermetic doctrine, as above so below, teaches that
the celestial spheres govern the terrestrial. By careful experiment
and observation, the adept records each distillation in his grimoire,
trusting that divine providence shall reward patient study.
The emerald tablet speaks of the prima materia, the first matter
of creation, hidden from the profane. Many a natural philosopher
has sought this arcane knowledge through sacred geometry and the
mysteries of numerology. The magnum opus proceeds through blackening,
whitening, and reddening, until the elixir of life is obtained.
Mercury and sulphur, joined in the crucible, reveal the secret
doctrine of the ancients. The furnace must be tended with prayer
and vigilance, for the work is both chemical and spiritual.
""",
    ),
    (
        "CHAPTER II",
        "Of Divine Providence",
        """
The theologians of the age held that salvation proceeds from grace
alone, and that the soul, being immortal, must render account at the
last judgment. Scripture teaches in the gospel that faith without
works is dead. The priest at the temple offered prayer and sacrament,
baptism and communion, that the congregation might attain heaven
and escape the torments of hell. Angels and archangels attend the
throne of the Almighty, while demons tempt the faithful to sin
and blasphemy. The prophet spake of resurrection and eternal life,
of a new covenant written not on tablets of stone but upon the heart.
The monastery preserved the psalms and the testament through ages
of darkness, and the monks copied scripture by candlelight.
Heresy was feared above plague, and many a martyr and saint was made
in those centuries. Miracles were reported at holy wells, and the
blessed relics drew pilgrims from every land seeking redemption.
""",
    ),
    (
        "CHAPTER III",
        "Of Natural Philosophy",
        """
The astronomer with his telescope observed the planets in their
orbits, and by mathematics computed the eclipse to the very hour.
Natural philosophy advanced by hypothesis and experiment, by careful
measurement with instrument and apparatus. The chemist in his
laboratory studied combustion, and found that oxygen, not phlogiston,
sustains the flame. The physician by dissection learned the anatomy
of the nervous system and the circulation of the blood. Electricity
and magnetism were shown to be twin phenomena, and the spectrum of
light was divided by the prism into its elements. The naturalist
collected specimens of every species, and the geologist read in the
strata the deep history of the earth. Gravity binds the comet to its
course as surely as it holds the apple to the bough. The theory of
evolution by natural selection would in time explain the origin of
species by empirical observation and rational deduction.
""",
    ),
    (
        "CHAPTER IV",
        "Of the Union of Wisdom",
        """
The wise man perceives that alchemy and theology and science are
three lamps lit from one flame. The astrologer casts his horoscope
by the same celestial mathematics the astronomer employs, and the
zodiac is mapped with the geometer's instrument. Is not the divine
creation itself the greatest experiment, and the scripture of nature
a testament writ in atoms and orbits? The mystic in his meditation
and the physician at his dissection both seek the hidden order of
the soul and of matter. The occult and the empirical are estranged
brothers, and gnosis and logic two roads up a single mountain.
The angel and the atom alike declare the glory of the creator,
and prophecy and hypothesis are each a wager against the dark.
Thus the initiate learns that the philosopher's stone is wisdom
itself, the transmutation not of mercury but of the mind, and the
elixir of life the eternal life promised by grace and by knowledge.
""",
    ),
]


def _wrap(draw, text, font, max_width):
    lines = []
    for paragraph in text.strip().split("\n"):
        words = paragraph.split()
        if not words:
            continue
        line = words[0]
        for word in words[1:]:
            trial = f"{line} {word}"
            if draw.textlength(trial, font=font) <= max_width:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def _load_font(size):
    for name in ("DejaVuSerif.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_page(heading, subtitle, body_lines, w=1240, h=1754, aged=True, seed=0):
    page = Image.new("L", (w, h), 235)
    draw = ImageDraw.Draw(page)
    font_h = _load_font(44)
    font_s = _load_font(30)
    font_b = _load_font(26)

    y = 130
    if heading:
        tw = draw.textlength(heading, font=font_h)
        draw.text(((w - tw) / 2, y), heading, font=font_h, fill=25)
        y += 80
        tw = draw.textlength(subtitle, font=font_s)
        draw.text(((w - tw) / 2, y), subtitle, font=font_s, fill=40)
        y += 90

    for line in body_lines:
        draw.text((140, y), line, font=font_b, fill=30)
        y += 42
        if y > h - 140:
            break

    if not aged:
        return page.convert("RGB")

    rng = np.random.default_rng(seed)
    arr = np.asarray(page).astype(np.float32)

    yy, xx = np.mgrid[0:h, 0:w]
    vignette = 18 * np.sin(xx / w * np.pi) * np.sin(yy / h * np.pi) - 26 + 14 * (xx / w)
    arr = arr + vignette
    arr += rng.normal(0, 9, arr.shape)  # scanner grain
    for _ in range(rng.integers(2, 5)):  # foxing stains
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        r = rng.integers(40, 140)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r**2
        arr[mask] -= rng.integers(8, 22)
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    aged_img = Image.fromarray(arr)
    angle = float(rng.uniform(-1.6, 1.6))
    aged_img = aged_img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=210)

    rgb = np.asarray(aged_img.convert("RGB")).astype(np.float32)
    rgb[:, :, 0] *= 1.00  # sepia tone
    rgb[:, :, 1] *= 0.94
    rgb[:, :, 2] *= 0.80
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def build_book(out_path: Path, pages_per_chapter=2):
    pages = []
    seed = 0
    for heading, subtitle, body in CHAPTERS:
        probe = Image.new("L", (10, 10))
        draw = ImageDraw.Draw(probe)
        lines = _wrap(draw, body, _load_font(26), 960)
        per_page = max(1, (len(lines) + pages_per_chapter - 1) // pages_per_chapter)
        for i in range(pages_per_chapter):
            chunk = lines[i * per_page : (i + 1) * per_page]
            if not chunk and i > 0:
                break
            pages.append(make_page(heading if i == 0 else "", subtitle if i == 0 else "", chunk, seed=seed))
            seed += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(out_path, format="PDF", save_all=True, append_images=pages[1:], resolution=150)
    print(f"wrote {out_path} ({len(pages)} pages)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="/tmp/arcanum_test/old_book.pdf")
    parser.add_argument("--pages-per-chapter", type=int, default=2)
    args = parser.parse_args()
    build_book(Path(args.output), args.pages_per_chapter)
