"""Small shared helpers used across the core package."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, List


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(value: str, max_length: int = 60) -> str:
    """Turn an arbitrary book title / filename into a filesystem-safe slug."""
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    value = re.sub(r"[-\s]+", "-", value)
    value = value.strip("-") or "book"
    return value[:max_length].rstrip("-") or "book"


def load_lexicon(path: Path) -> List[str]:
    """Load a newline-delimited term list, skipping blanks/comments."""
    if not path.exists():
        return []
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        terms.append(line)
    return terms


def chunk_text(text: str, max_chars: int = 4500) -> List[str]:
    """Split text into chunks close to ``max_chars`` without breaking words,
    preferring paragraph/sentence boundaries. Used for translation APIs that
    cap request size.
    """
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: List[str] = []
    paragraphs = text.split("\n")
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(para) <= max_chars:
            current = para
        else:
            for sentence in re.split(r"(?<=[.!?])\s+", para):
                cand2 = f"{current} {sentence}".strip() if current else sentence
                if len(cand2) <= max_chars:
                    current = cand2
                else:
                    if current:
                        chunks.append(current)
                    current = sentence[:max_chars]
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def safe_page_number(index: int) -> str:
    return f"{index + 1:04d}"


def clean_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_tokenize_basic(text: str) -> List[str]:
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", text.lower())


def batched(iterable: Iterable, n: int):
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch
