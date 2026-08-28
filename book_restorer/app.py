"""Book Restorer — AI Manuscript Studio.

A production-style Streamlit GUI for restoring, translating, and deeply
analyzing scanned historical/esoteric/theological/scientific books.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from core import analytics, ocr
from core.config import (
    AnalysisOptions,
    CleaningOptions,
    OCROptions,
    PipelineOptions,
    REFERENCE_CATEGORIES,
    TranslationOptions,
)
from core.language import language_name
from core.pipeline import BookRestorationPipeline, PipelineResult
from core.visualizations import (
    chapter_length_bar,
    entity_frequency_bar,
    generate_wordcloud_image,
    language_distribution_pie,
    noise_reduction_bar,
    readability_gauge,
    reference_category_pie,
    sentiment_arc_line,
    word_frequency_bar,
)
from gui.styles import CUSTOM_CSS, metric_card_html

TRANSLATION_TARGETS = {
    "English": "en",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Arabic": "ar",
    "Hebrew": "iw",
    "Dutch": "nl",
    "Polish": "pl",
    "Chinese (Simplified)": "zh-CN",
    "Japanese": "ja",
    "Latin": "la",
}

OCR_LANGUAGE_NAMES = {
    "eng": "English",
    "deu": "German",
    "fra": "French",
    "spa": "Spanish",
    "ita": "Italian",
    "por": "Portuguese",
    "lat": "Latin",
    "grc": "Ancient Greek",
    "rus": "Russian",
    "ara": "Arabic",
    "heb": "Hebrew",
    "osd": "Orientation/Script Detection",
}

CLEANING_PRESETS = {
    "Light (already fairly clean scans)": CleaningOptions(
        denoise_strength=4, adaptive_block_size=45, adaptive_c=8, clahe_clip_limit=1.5, remove_speckles=False
    ),
    "Standard (typical aged scans)": CleaningOptions(),
    "Aggressive (heavily degraded / foxed pages)": CleaningOptions(
        denoise_strength=16, adaptive_block_size=25, adaptive_c=22, clahe_clip_limit=3.5, remove_speckles=True
    ),
}


st.set_page_config(
    page_title="Book Restorer — AI Manuscript Studio",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _init_state():
    st.session_state.setdefault("results", {})
    st.session_state.setdefault("active_book", None)
    st.session_state.setdefault("work_dir", tempfile.mkdtemp(prefix="book_restorer_uploads_"))


_init_state()


def render_header():
    st.markdown(
        """
        <div class="hero-banner">
          <h1>📜 Book Restorer — AI Manuscript Studio</h1>
          <p>Upload scanned historical books to automatically clean and restore the pages, OCR and translate the
          text, generate chapter-by-chapter synopses, surface esoteric, theological &amp; scientific references
          with excerpts, and produce a stunning interactive analytics dashboard — all in one click.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_options_from_sidebar() -> tuple[PipelineOptions, str, Path]:
    st.sidebar.markdown("## ⚙️ Restoration Settings")

    output_dir_str = st.sidebar.text_input(
        "Output folder",
        value=str(Path.cwd() / "book_restorer_output"),
        help="All cleaned images, PDFs, transcripts, references, and analytics will be written here.",
    )

    st.sidebar.markdown("### 🧼 Image Cleaning")
    preset_name = st.sidebar.selectbox("Cleaning preset", list(CLEANING_PRESETS.keys()), index=1)
    cleaning = CLEANING_PRESETS[preset_name]
    dpi = st.sidebar.slider("Render DPI", min_value=150, max_value=600, value=300, step=50)
    cleaning.dpi = dpi
    deskew = st.sidebar.checkbox("Auto-deskew crooked scans", value=True)
    cleaning.deskew = deskew
    crop_borders = st.sidebar.checkbox("Auto-crop dark borders", value=True)
    cleaning.crop_borders = crop_borders

    st.sidebar.markdown("### 🔤 OCR")
    installed_langs = ocr.get_installed_languages() or ["eng"]
    lang_options = [l for l in installed_langs if l in OCR_LANGUAGE_NAMES] or ["eng"]
    default_langs = ["eng"] if "eng" in lang_options else lang_options[:1]
    selected_langs = st.sidebar.multiselect(
        "OCR languages (select all languages present in the book)",
        options=lang_options,
        default=default_langs,
        format_func=lambda code: f"{OCR_LANGUAGE_NAMES.get(code, code)} ({code})",
    )
    auto_detect = st.sidebar.checkbox("Auto-detect dominant language per page", value=True)

    st.sidebar.markdown("### 🌍 Translation")
    translate_enabled = st.sidebar.checkbox("Translate the book", value=False)
    target_lang_name = st.sidebar.selectbox(
        "Target language", list(TRANSLATION_TARGETS.keys()), index=0, disabled=not translate_enabled
    )
    translate_chapters = st.sidebar.checkbox(
        "Translate each chapter separately (for side-by-side view)", value=True, disabled=not translate_enabled
    )

    st.sidebar.markdown("### 🔎 Reference Scanning")
    categories = st.sidebar.multiselect(
        "Categories to detect",
        options=list(REFERENCE_CATEGORIES),
        default=list(REFERENCE_CATEGORIES),
        format_func=lambda c: c.title(),
    )

    st.sidebar.markdown("### 🧠 Analysis")
    summary_sentences = st.sidebar.slider("Synopsis length (sentences per chapter)", 3, 12, 6)
    top_keywords = st.sidebar.slider("Top keywords to extract", 10, 40, 20)
    use_llm = st.sidebar.checkbox(
        "Use LLM enrichment if OPENAI_API_KEY is set (richer synopses & insights)", value=True
    )

    options = PipelineOptions(
        output_dir=Path(output_dir_str),
        cleaning=cleaning,
        ocr=OCROptions(languages=selected_langs or ["eng"], auto_detect_language=auto_detect),
        translation=TranslationOptions(
            enabled=translate_enabled,
            target_language=TRANSLATION_TARGETS[target_lang_name],
            translate_chapters_individually=translate_chapters,
        ),
        analysis=AnalysisOptions(
            summary_sentence_count=summary_sentences,
            top_keywords=top_keywords,
            reference_categories=categories or list(REFERENCE_CATEGORIES),
            use_llm_if_available=use_llm,
        ),
    )
    return options, output_dir_str, Path(st.session_state["work_dir"])


def run_pipeline_for_files(uploaded_files, options: PipelineOptions, work_dir: Path):
    pipeline = BookRestorationPipeline()
    for uploaded_file in uploaded_files:
        book_key = uploaded_file.name
        local_pdf_path = work_dir / uploaded_file.name
        local_pdf_path.write_bytes(uploaded_file.getvalue())

        st.markdown(f"#### Processing **{book_key}**")
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def on_progress(fraction: float, message: str, _status_text=status_text, _bar=progress_bar):
            _bar.progress(fraction)
            _status_text.markdown(f"`{message}`")

        book_options = PipelineOptions(
            output_dir=options.output_dir,
            cleaning=options.cleaning,
            ocr=options.ocr,
            translation=options.translation,
            analysis=options.analysis,
            book_title=Path(uploaded_file.name).stem,
        )

        try:
            result = pipeline.run(local_pdf_path, book_options, progress_callback=on_progress)
            st.session_state["results"][book_key] = result
            st.session_state["active_book"] = book_key
            status_text.markdown("✅ **Complete**")
        except Exception as exc:
            status_text.markdown(f"❌ **Failed:** {exc}")
            st.exception(exc)


def render_metrics_row(result: PipelineResult):
    stats = result.stats
    cols = st.columns(6)
    values = [
        (f"{result.page_count}", "Pages"),
        (f"{stats.word_count:,}" if stats else "0", "Words"),
        (f"{len(result.chapters)}", "Chapters"),
        (f"{stats.estimated_reading_minutes:.0f} min" if stats else "0", "Reading Time"),
        (language_name(result.detected_primary_language), "Primary Language"),
        (f"{sum(s['total_matches'] for s in result.reference_stats.values())}", "References Found"),
    ]
    for col, (value, label) in zip(cols, values):
        col.markdown(metric_card_html(value, label), unsafe_allow_html=True)


def render_overview_tab(result: PipelineResult):
    render_metrics_row(result)
    st.markdown("<br/>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📖 Narrative Summary")
        st.write(result.insights.narrative_summary if result.insights else "N/A")
        st.subheader("📈 Sentiment Arc")
        st.write(result.insights.sentiment_arc_description if result.insights else "N/A")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("🖼️ Image Quality")
        if result.image_quality:
            st.metric("Pages restored", result.image_quality.pages_processed)
            st.metric("Avg. skew corrected", f"{result.image_quality.avg_skew_correction_degrees:.2f}°")
            st.metric("Avg. noise reduced", f"{result.image_quality.avg_noise_reduction_pct:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    if result.warnings:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("⚠️ Warnings")
        for w in result.warnings:
            st.warning(w)
        st.markdown("</div>", unsafe_allow_html=True)


def render_gallery_tab(result: PipelineResult):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if not result.original_image_paths:
        st.info("No pages available.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    page_idx = st.slider("Page", 1, len(result.original_image_paths), 1) - 1
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original scan**")
        st.image(str(result.original_image_paths[page_idx]), use_container_width=True)
    with col2:
        st.markdown("**Cleaned / restored**")
        st.image(str(result.cleaned_image_paths[page_idx]), use_container_width=True)
    if page_idx < len(result.cleaning_metrics):
        m = result.cleaning_metrics[page_idx]
        st.caption(
            f"Skew corrected: {m.get('skew_angle_degrees', 0):.2f}° · "
            f"Noise before/after: {m.get('noise_before', 0):.1f} → {m.get('noise_after', 0):.1f}"
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_text_tab(result: PipelineResult):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    page_idx = st.slider("Page", 1, max(1, len(result.page_texts)), 1, key="text_page_slider") - 1
    lang = language_name(result.page_languages[page_idx]) if page_idx < len(result.page_languages) else "Unknown"
    st.caption(f"Detected language: {lang}")

    if result.translation and result.translation.text:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original OCR text (page)**")
            st.text_area("original", result.page_texts[page_idx], height=350, label_visibility="collapsed")
        with col2:
            st.markdown("**Full-book translation (excerpt)**")
            st.text_area("translated", result.translation.text[:4000], height=350, label_visibility="collapsed")
    else:
        st.markdown("**OCR text (page)**")
        st.text_area("original_only", result.page_texts[page_idx], height=400, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)


def render_chapters_tab(result: PipelineResult):
    for chapter in result.chapters:
        with st.expander(f"📗 Chapter {chapter.number}: {chapter.title}", expanded=(chapter.number == 1)):
            st.markdown("**Synopsis**")
            st.info(chapter.synopsis)
            cols = st.columns(2) if chapter.translated_text else [st.container()]
            with cols[0]:
                st.markdown("**Original text**")
                st.text_area(f"orig_{chapter.number}", chapter.text[:6000], height=250, label_visibility="collapsed")
            if chapter.translated_text:
                with cols[1]:
                    st.markdown("**Translated text**")
                    st.text_area(
                        f"trans_{chapter.number}", chapter.translated_text[:6000], height=250, label_visibility="collapsed"
                    )


def render_references_tab(result: PipelineResult):
    categories = list(result.references_by_category.keys())
    if not categories:
        st.info("No reference categories were scanned.")
        return
    tabs = st.tabs([c.title() for c in categories])
    icons = {"esoteric": "🔮", "theological": "⛪", "scientific": "🔬"}
    for tab, category in zip(tabs, categories):
        with tab:
            refs = result.references_by_category[category]
            stat = result.reference_stats.get(category, {})
            st.markdown(
                f"### {icons.get(category, '📌')} {category.title()} References "
                f"— {stat.get('total_matches', 0)} matches, {stat.get('unique_terms', 0)} unique terms"
            )
            if not refs:
                st.info(f"No {category} references were detected in this book.")
                continue

            top_terms = stat.get("top_terms", [])
            if top_terms:
                st.plotly_chart(
                    entity_frequency_bar(top_terms, title=f"Top {category.title()} Terms"),
                    use_container_width=True,
                    key=f"ref_chart_{category}",
                )

            df = pd.DataFrame(
                [{"Page": r.page + 1, "Term": r.term, "Chapter": r.chapter_title, "Excerpt": r.excerpt} for r in refs]
            )
            st.dataframe(df, use_container_width=True, height=280)

            st.markdown("#### Excerpts")
            for r in refs[:60]:
                st.markdown(
                    f'<span class="term-pill">{r.term}</span> <b>Page {r.page + 1}</b>'
                    f'<div class="excerpt-box">{r.excerpt}</div>',
                    unsafe_allow_html=True,
                )


def render_analytics_tab(result: PipelineResult):
    stats = result.stats
    if stats is None:
        st.info("No analytics available.")
        return

    cols = st.columns(5)
    for col, (value, label) in zip(
        cols,
        [
            (f"{stats.flesch_reading_ease:.1f}", "Flesch Reading Ease"),
            (f"{stats.flesch_kincaid_grade:.1f}", "Flesch-Kincaid Grade"),
            (f"{stats.gunning_fog:.1f}", "Gunning Fog Index"),
            (f"{stats.lexical_diversity:.2f}", "Lexical Diversity"),
            (f"{stats.avg_sentence_length:.1f}", "Avg. Words / Sentence"),
        ],
    ):
        col.markdown(metric_card_html(value, label), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(word_frequency_bar(result.word_frequency[:25]), use_container_width=True, key="wf_chart")
        st.plotly_chart(
            reference_category_pie(result.reference_stats), use_container_width=True, key="ref_pie_chart"
        )
        st.plotly_chart(
            noise_reduction_bar(result.cleaning_metrics), use_container_width=True, key="noise_chart"
        )
    with col2:
        st.plotly_chart(
            sentiment_arc_line(result.sentiment_by_chapter), use_container_width=True, key="sentiment_chart"
        )
        st.plotly_chart(readability_gauge(stats.flesch_reading_ease), use_container_width=True, key="gauge_chart")
        st.plotly_chart(
            language_distribution_pie(analytics.compute_language_distribution(result.page_languages)),
            use_container_width=True,
            key="lang_chart",
        )

    st.plotly_chart(chapter_length_bar(result.chapters), use_container_width=True, key="chlen_chart")

    st.markdown("#### Word Cloud")
    wc_image = generate_wordcloud_image(result.word_frequency)
    if wc_image is not None:
        st.image(wc_image, use_container_width=True)


def render_insights_tab(result: PipelineResult):
    insights = result.insights
    if insights is None:
        st.info("No insights available.")
        return
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧩 Key Themes")
    st.markdown(" ".join(f'<span class="term-pill">{t}</span>' for t in insights.themes), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("👤 Notable Names & Recurring Entities")
        if insights.notable_entities:
            st.plotly_chart(entity_frequency_bar(insights.notable_entities[:15]), use_container_width=True, key="ent_chart")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("🔑 Top Keywords")
        if insights.top_keywords:
            st.plotly_chart(word_frequency_bar(insights.top_keywords[:15], title="Top Keywords"), use_container_width=True, key="kw_chart")
        st.markdown("</div>", unsafe_allow_html=True)


def render_downloads_tab(result: PipelineResult):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("⬇️ Downloads")

    if result.cleaned_pdf_path and result.cleaned_pdf_path.exists():
        st.download_button(
            "Download Cleaned PDF",
            data=result.cleaned_pdf_path.read_bytes(),
            file_name=result.cleaned_pdf_path.name,
            mime="application/pdf",
        )
    if result.html_report_path and result.html_report_path.exists():
        st.download_button(
            "Download Interactive Analytics Report (HTML)",
            data=result.html_report_path.read_bytes(),
            file_name=result.html_report_path.name,
            mime="text/html",
        )
    if result.pdf_report_path and result.pdf_report_path.exists():
        st.download_button(
            "Download Full Report (PDF)",
            data=result.pdf_report_path.read_bytes(),
            file_name=result.pdf_report_path.name,
            mime="application/pdf",
        )

    st.markdown("---")
    st.caption(f"All outputs were written to: `{result.output_dir}`")
    if st.button("📦 Prepare full output ZIP"):
        with st.spinner("Zipping output folder..."):
            zip_base = Path(tempfile.gettempdir()) / f"{result.output_dir.name}_bundle"
            zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=result.output_dir)
        with open(zip_path, "rb") as f:
            st.download_button("Download Everything (.zip)", data=f.read(), file_name=f"{result.output_dir.name}.zip", mime="application/zip")
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    render_header()
    options, output_dir_str, work_dir = build_options_from_sidebar()

    st.sidebar.markdown("---")
    uploaded_files = st.sidebar.file_uploader(
        "📤 Upload scanned book PDF(s)", type=["pdf"], accept_multiple_files=True
    )
    run_clicked = st.sidebar.button("🚀 Run Restoration", use_container_width=True, disabled=not uploaded_files)

    if run_clicked and uploaded_files:
        Path(output_dir_str).mkdir(parents=True, exist_ok=True)
        run_pipeline_for_files(uploaded_files, options, work_dir)

    results = st.session_state["results"]
    if not results:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            "### 👋 Get started\n"
            "1. Upload one or more scanned book PDFs in the sidebar.\n"
            "2. Choose your cleaning preset, OCR languages, translation target, and reference categories.\n"
            "3. Click **Run Restoration** and watch the live progress.\n"
            "4. Explore cleaned pages, translations, chapter synopses, esoteric/theological/scientific "
            "references, and a full analytics dashboard — then download everything.\n"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    book_names = list(results.keys())
    active_book = st.selectbox(
        "📚 Select a processed book",
        book_names,
        index=book_names.index(st.session_state["active_book"]) if st.session_state["active_book"] in book_names else 0,
    )
    st.session_state["active_book"] = active_book
    result = results[active_book]

    tabs = st.tabs(
        ["Overview", "Page Gallery", "Text & Translation", "Chapters & Synopses", "References", "Analytics Dashboard", "Insights", "Downloads"]
    )
    with tabs[0]:
        render_overview_tab(result)
    with tabs[1]:
        render_gallery_tab(result)
    with tabs[2]:
        render_text_tab(result)
    with tabs[3]:
        render_chapters_tab(result)
    with tabs[4]:
        render_references_tab(result)
    with tabs[5]:
        render_analytics_tab(result)
    with tabs[6]:
        render_insights_tab(result)
    with tabs[7]:
        render_downloads_tab(result)


if __name__ == "__main__":
    main()
