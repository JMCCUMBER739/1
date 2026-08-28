"""Command-line interface: `python -m arcanum <input.pdf> -o <folder>`."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from . import __app_name__, __tagline__, __version__
from .pipeline import PipelineOptions, run_pipeline
from .translate import SUPPORTED_LANGUAGES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arcanum",
        description=f"{__app_name__} — {__tagline__} (v{__version__})",
    )
    parser.add_argument("input", nargs="?", help="Path to the scanned book PDF")
    parser.add_argument("-o", "--output", default="arcanum_output", help="Output folder (default: ./arcanum_output)")
    parser.add_argument(
        "-t",
        "--translate",
        action="append",
        default=[],
        metavar="LANG",
        help="Target language code (repeatable), e.g. -t es -t fr",
    )
    parser.add_argument("--ocr-lang", default="eng", help="Tesseract OCR language (default: eng)")
    parser.add_argument("--dpi", type=int, default=250, help="Rendering DPI for page images (default: 250)")
    parser.add_argument("--max-pages", type=int, default=None, help="Process only the first N pages")
    parser.add_argument("--no-deskew", action="store_true", help="Disable automatic deskewing")
    parser.add_argument("--synopsis-sentences", type=int, default=5, help="Sentences per chapter synopsis (default: 5)")
    parser.add_argument("--gui", action="store_true", help="Launch the graphical interface")
    parser.add_argument("--list-languages", action="store_true", help="List supported translation languages")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list_languages:
        for code, name in SUPPORTED_LANGUAGES.items():
            print(f"  {code:6s} {name}")
        return 0

    if args.gui or not args.input:
        from .gui.main_window import launch

        return launch()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 2

    options = PipelineOptions(
        input_pdf=str(input_path),
        output_dir=args.output,
        dpi=args.dpi,
        ocr_language=args.ocr_lang,
        translate_to=args.translate,
        deskew=not args.no_deskew,
        max_pages=args.max_pages,
        synopsis_sentences=args.synopsis_sentences,
    )

    def progress(pct: int, msg: str) -> None:
        bar = "█" * (pct // 4) + "░" * (25 - pct // 4)
        sys.stdout.write(f"\r[{bar}] {pct:3d}%  {msg:<55.55s}")
        sys.stdout.flush()
        if pct >= 100:
            sys.stdout.write("\n")

    result = run_pipeline(options, progress=progress)

    print(f"\n{__app_name__} finished in {result.elapsed_seconds:.0f}s")
    print(f"  Output folder : {result.output_dir}")
    if result.cleaned_pdf:
        print(f"  Cleaned PDF   : {result.cleaned_pdf}")
    if result.report_html:
        print(f"  Report        : {result.report_html}")
    if result.analytics:
        counts = result.analytics.category_counts
        print(
            f"  References    : esoteric {counts.get('esoteric', 0)}, "
            f"theological {counts.get('theological', 0)}, "
            f"scientific {counts.get('scientific', 0)}"
        )
    for warning in result.warnings:
        print(f"  warning: {warning.splitlines()[0]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
