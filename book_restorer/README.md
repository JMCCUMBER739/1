# 📜 Book Restorer — AI Manuscript Studio

A production-style tool for restoring scanned historical books and turning them into rich,
analyzable digital artifacts. Upload a scanned PDF and Book Restorer will:

- **Clean & restore** every page (deskew, denoise, contrast-enhance, binarize, crop borders)
  and rebuild a crisp cleaned PDF, while preserving the original scans untouched.
- **OCR** the text (multi-language, powered by Tesseract) with automatic per-page language
  detection.
- **Translate** the book (full text and/or chapter-by-chapter) into any target language.
- **Detect chapters** and generate an AI synopsis for each one.
- **Surface esoteric, theological, and scientific references** anywhere in the book, each with
  the matched term, page number, and a citable excerpt.
- **Compute advanced analytics**: readability scores, sentiment arc across chapters, word
  frequency, lexical diversity, language mix, keyword/entity extraction, and more.
- **Generate insights**: key themes, a narrative overview, and a description of the emotional
  arc of the text.
- Export everything (cleaned images, cleaned PDF, transcripts, translations, chapter synopses,
  reference excerpts, an interactive analytics dashboard, and a polished PDF report) to a
  folder you choose.
- All of the above is driven from a polished, professional **Streamlit GUI** with input/output
  selectors, live progress, and interactive Plotly visualizations.

The tool works **out of the box with no API keys** (OCR, cleaning, reference detection,
extractive summarization, and translation via a free keyless backend all work offline/without
credentials). If you set `OPENAI_API_KEY`, synopses, insights, and translation are automatically
upgraded to use an LLM for noticeably richer, more abstractive output.

## How it works

```
Scanned PDF
    │
    ▼
Render pages (PyMuPDF) ──► original page images
    │
    ▼
Clean pages (OpenCV: deskew → denoise → CLAHE contrast → adaptive threshold → despeckle → crop)
    │
    ├──► cleaned page images
    └──► reassembled cleaned.pdf (img2pdf)
    │
    ▼
OCR (Tesseract, multi-language) + per-page language detection
    │
    ▼
Chapter detection (heading heuristics) ──► per-chapter text
    │
    ├──► Translation (optional; OpenAI if configured, else Google Translate)
    ├──► Chapter synopses (LexRank extractive summarization, or LLM if configured)
    ├──► Esoteric / Theological / Scientific reference scanning (curated lexicons + excerpts)
    ├──► Analytics (readability, sentiment, word frequency, language mix, image-quality stats)
    └──► Insights (themes, keywords, notable entities, narrative + sentiment-arc summary)
    │
    ▼
Reports: interactive HTML dashboard (Plotly) + polished PDF report (ReportLab)
    │
    ▼
Organized output folder + Streamlit dashboard
```

## Output folder layout

Every processed book gets its own subfolder (named after the book title) inside the output
directory you choose in the GUI:

```
<output_dir>/<book-slug>/
├── manifest.json                     # summary of everything produced
├── cleaned.pdf                       # restored, cleaned version of the whole book
├── images/
│   ├── original/page_0001.png ...    # untouched scans
│   └── cleaned/page_0001.png ...     # restored scans
├── text/
│   ├── full_text_original.txt
│   ├── full_text_translated_<lang>.txt
│   └── per_page/page_0001.txt ...
├── chapters/
│   └── chapter_01/
│       ├── text.txt
│       ├── translated.txt
│       └── synopsis.txt
├── references/
│   ├── esoteric.json
│   ├── theological.json
│   └── scientific.json
├── analytics/
│   ├── stats.json
│   ├── word_frequency.json
│   ├── sentiment_by_chapter.json
│   ├── wordcloud.png
│   └── analytics_report.html         # interactive, self-contained dashboard
├── insights/
│   └── insights.json
└── reports/
    └── full_report.pdf               # polished, shareable summary report
```

## Requirements

- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) with language packs for whichever
  languages your books use, plus Poppler (used transitively by PyMuPDF-adjacent tooling):

  ```bash
  sudo apt-get update
  sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu tesseract-ocr-fra \
      tesseract-ocr-spa tesseract-ocr-ita tesseract-ocr-por tesseract-ocr-lat tesseract-ocr-grc \
      tesseract-ocr-rus tesseract-ocr-ara tesseract-ocr-heb poppler-utils
  ```

  (macOS: `brew install tesseract tesseract-lang poppler`)

## Setup

```bash
cd book_restorer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# One-time NLTK data download used by the summarizer/analytics
python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt', 'punkt_tab', 'stopwords')]"
```

## Run the GUI

```bash
streamlit run app.py
```

Then open the printed local URL in your browser. In the sidebar:

1. Choose your cleaning preset, DPI, OCR languages, translation target (optional), and which
   reference categories to scan for.
2. Upload one or more scanned book PDFs.
3. Pick/confirm the output folder.
4. Click **Run Restoration** and watch live progress.
5. Explore the **Overview**, **Page Gallery** (before/after), **Text & Translation**,
   **Chapters & Synopses**, **References**, **Analytics Dashboard**, **Insights**, and
   **Downloads** tabs.

### Try it with a sample book

No scanned book handy? Generate a synthetic "aged scan" with speckle noise, sepia tint, and
per-page skew (containing esoteric, theological, and scientific vocabulary to exercise every
feature):

```bash
python examples/generate_sample_scan.py
```

This writes `examples/sample_old_book.pdf`, which you can upload directly in the GUI.

## Using it without the GUI

The whole pipeline is a plain Python API, so you can script it directly:

```python
from pathlib import Path
from core.config import PipelineOptions, TranslationOptions
from core.pipeline import BookRestorationPipeline

options = PipelineOptions(
    output_dir=Path("./output"),
    translation=TranslationOptions(enabled=True, target_language="es"),
)
result = BookRestorationPipeline().run(Path("my_scanned_book.pdf"), options)

print(result.stats)
print(result.reference_stats)
for chapter in result.chapters:
    print(chapter.number, chapter.title, "->", chapter.synopsis)
```

## Optional: enable LLM-powered enrichment

Set an environment variable before launching to unlock richer, abstractive synopses/insights
and higher quality translation:

```bash
export OPENAI_API_KEY=sk-...
streamlit run app.py
```

Without it, the tool automatically falls back to fully offline extractive summarization
(LexRank), frequency-based keyword/entity extraction, and free Google Translate — no
functionality is blocked, quality is simply enhanced further when a key is present.

## Testing

```bash
pip install pytest
python -m pytest -q
```

The test suite covers image cleaning (deskew/denoise/binarize), chapter-boundary detection,
reference extraction for all three categories, analytics computations, and a full end-to-end
pipeline smoke test (skipped automatically if Tesseract isn't installed).

## Project layout

```
book_restorer/
├── app.py                # Streamlit GUI
├── core/                 # All processing logic (no GUI dependencies)
│   ├── pdf_io.py          # PDF <-> image conversion
│   ├── image_cleaning.py  # OpenCV restoration pipeline
│   ├── ocr.py             # Tesseract wrapper
│   ├── language.py        # Language detection
│   ├── translation.py     # Translation backends (LLM / Google Translate)
│   ├── chapters.py        # Chapter boundary detection
│   ├── summarization.py   # Chapter synopses
│   ├── references.py      # Esoteric / theological / scientific reference detection
│   ├── insights.py        # Keywords, entities, themes, narrative overview
│   ├── analytics.py       # Readability, sentiment, frequency, language mix
│   ├── visualizations.py  # Plotly chart builders + word cloud
│   ├── report.py          # HTML dashboard + PDF report generation
│   └── pipeline.py        # End-to-end orchestration
├── data/lexicons/        # Curated term lists for reference detection
├── gui/styles.py         # Custom CSS for the Streamlit UI
├── examples/             # Sample-book generator for demos/testing
└── tests/                # Unit + smoke tests
```
