# Palimpsest

**Book Restoration & Intelligence Studio** — a production-ready tool for cleaning scanned book PDFs, translating multi-language wording, extracting chapter synopses and insights, mining esoteric / theological / scientific references with excerpts, and delivering advanced analytics with professional visualizations.

## Features

- **Input / output selectors** — Streamlit GUI with PDF upload or local path, plus output folder picker
- **Image restoration** — deskew, denoise, CLAHE contrast, adaptive binarization; original + cleaned page images exported
- **Cleaned PDF** — rebuilt from restored pages
- **OCR** — Tesseract when installed; falls back to embedded PDF text layers
- **Translation** — multi-language via Google Translate (`deep-translator`); configurable target
- **Chapter intelligence** — automatic chapter detection, synopses, and insight bullets
- **Reference mining** — esoteric, theological, and scientific lexicons with page excerpts (CSV + HTML)
- **Advanced analytics** — readability, lexical diversity, sentiment proxy, language mix, quality curves, thematic radar, sunburst taxonomy
- **CLI** — headless batch processing for pipelines

## Quick start

```bash
cd palimpsest
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional but recommended for true scanned OCR:
#   sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
chmod +x run_gui.sh
./run_gui.sh
```

Open the URL Streamlit prints (default `http://localhost:8501`).

### CLI

```bash
python run_cli.py samples/sample_ancient_miscellany.pdf -o outputs --no-translate
```

Generate the sample volume:

```bash
python -m app.assets.make_sample_pdf
```

## Output layout

```
outputs/<book>/<job_id>/
  images/original/          # raw rasterized pages
  images/cleaned/           # restored pages
  pdf/cleaned.pdf
  pdf/translated.pdf
  text/cleaned_full.txt
  text/translated_full.txt
  synopses/chapters.md
  synopses/chapters.json
  references/all_references.csv
  references/esoteric_references.csv
  references/theological_references.csv
  references/scientific_references.csv
  analytics/report.json
  report.html
  manifest.json
```

## Tests

```bash
pip install pytest
pytest -q
```

## Architecture

| Module | Role |
|--------|------|
| `app/pipeline/ingest.py` | PDF → page images |
| `app/pipeline/clean.py` | Scan restoration |
| `app/pipeline/ocr.py` | Text extraction / OCR |
| `app/pipeline/translate.py` | Multi-language translation |
| `app/pipeline/chapters.py` | Chapters, synopses, insights |
| `app/pipeline/references.py` | Domain reference extraction |
| `app/pipeline/analytics.py` | Metrics |
| `app/pipeline/export.py` | PDFs, CSV, HTML, manifest |
| `app/gui/main.py` | Professional Streamlit studio |
| `app/viz/charts.py` | Plotly dashboards |

## Notes

- Translation requires outbound network access. Use `--no-translate` / uncheck the GUI option for fully offline runs.
- Without Tesseract, born-digital or already-OCR'd PDFs still process via their text layer.
