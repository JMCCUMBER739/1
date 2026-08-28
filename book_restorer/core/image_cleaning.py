"""Scanned-page image restoration pipeline built on OpenCV.

The pipeline is intentionally modular: each step can be toggled through
:class:`core.config.CleaningOptions` so the GUI can expose an "aggressiveness"
control without the caller needing to know OpenCV internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import cv2
import numpy as np
from PIL import Image

from core.config import CleaningOptions


@dataclass
class CleaningResult:
    image: Image.Image
    metrics: Dict[str, float] = field(default_factory=dict)


def _pil_to_cv(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv_to_pil(mat: np.ndarray) -> Image.Image:
    if mat.ndim == 2:
        return Image.fromarray(mat)
    return Image.fromarray(cv2.cvtColor(mat, cv2.COLOR_BGR2RGB))


def _projection_variance_score(binary: np.ndarray, angle: float) -> float:
    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(binary, rot_matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    row_sums = rotated.sum(axis=1).astype(np.float64)
    return float(np.var(row_sums))


def _estimate_skew_angle(gray: np.ndarray, dpi: int = 300, max_angle: float = 10.0) -> float:
    """Estimate the dominant skew angle of the page's text lines.

    Uses the classic projection-profile method: the image is binarized,
    downscaled for speed, then rotated over a range of candidate angles.
    The angle whose horizontal (row-wise) ink projection has the highest
    variance is chosen, since perfectly horizontal text lines produce sharp
    alternating peaks/valleys (high variance) while skewed text smears the
    projection out (low variance).

    This is considerably more robust to scan speckle/foxing noise and to
    ragged paragraph margins than fitting a single bounding rectangle
    (``minAreaRect``) over all ink pixels, which can be thrown off by the
    overall *shape* of a text block rather than the true line orientation.
    """
    inverted = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Suppress isolated speckle noise before measuring projections.
    denoised = cv2.medianBlur(thresh, 3)

    if cv2.countNonZero(denoised) < 200:
        return 0.0

    # Downscale for a fast coarse search; the angle is resolution-independent.
    scale = 700.0 / max(denoised.shape)
    if scale < 1.0:
        small = cv2.resize(denoised, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = denoised

    coarse_angles = np.arange(-max_angle, max_angle + 0.001, 0.5)
    scores = [_projection_variance_score(small, a) for a in coarse_angles]
    best_coarse = float(coarse_angles[int(np.argmax(scores))])

    fine_angles = np.arange(best_coarse - 0.6, best_coarse + 0.601, 0.1)
    fine_scores = [_projection_variance_score(small, a) for a in fine_angles]
    best_angle = float(fine_angles[int(np.argmax(fine_scores))])

    # A flat score landscape (near-blank page, or pure noise) means we
    # couldn't confidently detect a skew; better to leave the page alone.
    baseline_score = _projection_variance_score(small, 0.0)
    best_score = max(fine_scores)
    if baseline_score <= 0 or best_score < baseline_score * 1.05:
        return 0.0

    return best_angle


def _rotate(mat: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.05:
        return mat
    (h, w) = mat.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        mat,
        rot_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _crop_borders(gray: np.ndarray, margin: int) -> np.ndarray:
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return gray
    x, y, w, h = cv2.boundingRect(coords)
    x0 = max(x - margin, 0)
    y0 = max(y - margin, 0)
    x1 = min(x + w + margin, gray.shape[1])
    y1 = min(y + h + margin, gray.shape[0])
    if x1 <= x0 or y1 <= y0:
        return gray
    return gray[y0:y1, x0:x1]


def estimate_noise(gray: np.ndarray) -> float:
    """Rough noise estimate via the Laplacian variance of high frequencies."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def clean_page_image(image: Image.Image, options: CleaningOptions) -> CleaningResult:
    """Run the full restoration pipeline on a single scanned page.

    Steps: grayscale -> deskew -> denoise -> contrast enhancement (CLAHE)
    -> adaptive binarization -> speckle removal -> border crop.
    """
    metrics: Dict[str, float] = {}
    mat = _pil_to_cv(image)
    gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)

    metrics["noise_before"] = estimate_noise(gray)

    skew_angle = 0.0
    if options.deskew:
        skew_angle = _estimate_skew_angle(gray, dpi=options.dpi)
        if abs(skew_angle) > 0.1:
            gray = _rotate(gray, skew_angle)
    metrics["skew_angle_degrees"] = skew_angle

    if options.denoise:
        strength = max(1, min(30, options.denoise_strength))
        gray = cv2.fastNlMeansDenoising(gray, h=strength, templateWindowSize=7, searchWindowSize=21)

    if options.enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=options.clahe_clip_limit, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    if options.binarize:
        block_size = options.adaptive_block_size
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(3, block_size)
        gray = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            options.adaptive_c,
        )

    if options.remove_speckles:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        gray = cv2.medianBlur(gray, 3)

    if options.crop_borders:
        gray = _crop_borders(gray, options.border_margin_px)

    metrics["noise_after"] = estimate_noise(gray)
    metrics["output_width"] = gray.shape[1]
    metrics["output_height"] = gray.shape[0]

    return CleaningResult(image=_cv_to_pil(gray), metrics=metrics)
