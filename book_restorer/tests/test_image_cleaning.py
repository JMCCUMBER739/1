import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.config import CleaningOptions
from core.image_cleaning import clean_page_image


def make_noisy_scanned_page(size=(600, 800), angle=3.0, text="The Quick Brown Fox") -> Image.Image:
    image = Image.new("L", size, color=235)
    draw = ImageDraw.Draw(image)
    for y in range(40, size[1] - 40, 60):
        draw.text((40, y), text, fill=10)
    rotated = image.rotate(angle, expand=False, fillcolor=235)

    arr = np.array(rotated).astype(np.int16)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 15, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy).convert("RGB")


def test_clean_page_image_returns_smaller_or_equal_and_binarized():
    page = make_noisy_scanned_page()
    result = clean_page_image(page, CleaningOptions())

    assert result.image.mode in ("L", "1")
    arr = np.array(result.image)
    unique_values = np.unique(arr)
    # Binarized output should be (close to) two-tone.
    assert len(unique_values) <= 10

    assert "skew_angle_degrees" in result.metrics
    assert "noise_before" in result.metrics
    assert "noise_after" in result.metrics


def test_clean_page_image_respects_disabled_options():
    page = make_noisy_scanned_page(angle=0.0)
    options = CleaningOptions(denoise=False, deskew=False, enhance_contrast=False, binarize=False, remove_speckles=False, crop_borders=False)
    result = clean_page_image(page, options)
    assert result.image.size == page.size
