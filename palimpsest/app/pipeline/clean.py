"""Scanned-page image cleaning: deskew, denoise, contrast, binarize."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.pipeline.models import PageImage


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def estimate_skew(gray: np.ndarray) -> float:
    """Estimate skew angle in degrees using min-area rect on edges."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    coords = np.column_stack(np.where(edges > 0))
    if len(coords) < 100:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    # OpenCV returns angle of the long side; clamp small noise
    if abs(angle) > 15:
        return 0.0
    return float(angle)


def deskew(gray: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.15:
        return gray
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def denoise(gray: np.ndarray, mode: str = "auto") -> np.ndarray:
    if mode == "gentle":
        return cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)
    if mode == "aggressive":
        return cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
    # auto: light bilateral for text preservation
    return cv2.bilateralFilter(gray, 5, 50, 50)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )


def quality_score(gray: np.ndarray) -> float:
    """Heuristic 0–100 quality: sharpness + contrast."""
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = float(gray.std())
    sharp = min(lap / 500.0, 1.0)
    cont = min(contrast / 60.0, 1.0)
    return round(100.0 * (0.55 * sharp + 0.45 * cont), 2)


def clean_page(
    rgb: np.ndarray,
    mode: str = "auto",
    binarize: bool = True,
) -> tuple[np.ndarray, float, float]:
    gray = _to_gray(rgb)
    angle = estimate_skew(gray)
    gray = deskew(gray, angle)
    gray = denoise(gray, mode=mode)
    gray = enhance_contrast(gray)
    if binarize:
        out = adaptive_binarize(gray)
    else:
        out = gray
    score = quality_score(out if not binarize else gray)
    return out, angle, score


def clean_pages(
    pages: list[PageImage],
    output_dir: Path,
    mode: str = "auto",
    binarize: bool = True,
) -> list[PageImage]:
    cleaned_dir = output_dir / "images" / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    updated: list[PageImage] = []
    for page in pages:
        with Image.open(page.original_path) as im:
            rgb = np.array(im.convert("RGB"))
        cleaned, angle, score = clean_page(rgb, mode=mode, binarize=binarize)
        out_path = cleaned_dir / f"page_{page.page_index + 1:04d}.png"
        Image.fromarray(cleaned).save(out_path)
        updated.append(
            PageImage(
                page_index=page.page_index,
                original_path=page.original_path,
                cleaned_path=out_path,
                width=page.width,
                height=page.height,
                skew_degrees=angle,
                quality_score=score,
            )
        )
    return updated
