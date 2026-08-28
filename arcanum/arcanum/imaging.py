"""Image restoration for scanned book pages.

Every page goes through a restoration chain tuned for aged paper:
grayscale conversion, illumination flattening (removes yellowing and
uneven lighting), non-local-means denoising, automatic deskewing and a
gentle contrast stretch. Quality metrics are captured before and after
so the analytics layer can quantify the improvement.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PageQuality:
    """Quality metrics for a single page image."""

    brightness: float
    contrast: float
    sharpness: float
    noise: float

    def to_dict(self) -> dict:
        return {
            "brightness": round(self.brightness, 2),
            "contrast": round(self.contrast, 2),
            "sharpness": round(self.sharpness, 2),
            "noise": round(self.noise, 2),
        }


@dataclass
class CleanResult:
    """Result of cleaning one page."""

    image: np.ndarray  # cleaned grayscale image
    skew_angle: float  # degrees corrected
    before: PageQuality
    after: PageQuality


def measure_quality(gray: np.ndarray) -> PageQuality:
    """Compute brightness / contrast / sharpness / noise estimates."""
    brightness = float(gray.mean())
    contrast = float(gray.std())
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(lap.var())
    # Noise estimate: median absolute deviation of the high-pass residual.
    blur = cv2.medianBlur(gray, 3)
    residual = gray.astype(np.int16) - blur.astype(np.int16)
    noise = float(np.median(np.abs(residual)))
    return PageQuality(brightness, contrast, sharpness, noise)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide by an estimated background to remove stains and shading."""
    kernel = max(gray.shape) // 20 | 1  # odd kernel ~5% of page size
    background = cv2.medianBlur(gray, min(kernel, 99))
    background = np.where(background == 0, 1, background)
    flat = cv2.divide(gray, background, scale=255)
    return flat


def estimate_skew(gray: np.ndarray) -> float:
    """Estimate page skew in degrees using text-line Hough analysis."""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=gray.shape[1] // 4,
        maxLineGap=20,
    )
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4)[:200]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 15:  # near-horizontal text lines only
            angles.append(angle)
    if not angles:
        return 0.0
    return float(np.median(angles))


def _rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def clean_page(image: np.ndarray, deskew: bool = True) -> CleanResult:
    """Run the full restoration chain on a page image (BGR or gray)."""
    gray = _to_gray(image)
    before = measure_quality(gray)

    flat = _flatten_illumination(gray)
    denoised = cv2.fastNlMeansDenoising(flat, None, h=12, templateWindowSize=7, searchWindowSize=21)

    skew = 0.0
    if deskew:
        skew = estimate_skew(denoised)
        if abs(skew) > 0.15:
            denoised = _rotate(denoised, skew)
        else:
            skew = 0.0

    # Gentle contrast stretch anchored on robust percentiles.
    lo, hi = np.percentile(denoised, (2, 98))
    if hi - lo > 10:
        stretched = np.clip((denoised.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    else:
        stretched = denoised

    after = measure_quality(stretched)
    return CleanResult(image=stretched, skew_angle=round(skew, 2), before=before, after=after)


def binarize_for_ocr(gray: np.ndarray) -> np.ndarray:
    """Adaptive binarization used only as OCR input (not for display)."""
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=35,
        C=15,
    )
