#!/usr/bin/env python3
"""CLI entry point for Palimpsest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="palimpsest",
        description="Restore scanned books, translate, extract references, and analyze.",
    )
    p.add_argument("pdf", type=Path, help="Input PDF path")
    p.add_argument("-o", "--output", type=Path, default=Path("outputs"), help="Output folder")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--max-pages", type=int, default=None)
    p.add_argument("--clean-mode", choices=["auto", "gentle", "aggressive"], default="auto")
    p.add_argument("--no-binarize", action="store_true")
    p.add_argument("--no-ocr", action="store_true")
    p.add_argument("--ocr-lang", default="en")
    p.add_argument("--no-translate", action="store_true")
    p.add_argument("--translate-to", default="en")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def progress(pct: float, msg: str) -> None:
        bar = int(pct * 30)
        sys.stdout.write(f"\r[{'#' * bar}{'.' * (30 - bar)}] {pct:5.1%}  {msg[:60]:<60}")
        sys.stdout.flush()
        if pct >= 1.0:
            sys.stdout.write("\n")

    result = run_pipeline(
        args.pdf,
        args.output,
        dpi=args.dpi,
        max_pages=args.max_pages,
        clean_mode=args.clean_mode,
        binarize=not args.no_binarize,
        enable_ocr=not args.no_ocr,
        ocr_lang_code=args.ocr_lang,
        enable_translation=not args.no_translate,
        translate_target=args.translate_to,
        progress=progress,
    )
    print(json.dumps(result.manifest, indent=2))
    for w in result.warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
