"""Palimpsest Streamlit GUI — professional book restoration studio."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure package imports resolve when launched via `streamlit run`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import __version__
from app.config import SETTINGS, SUPPORTED_TRANSLATE_LANGS
from app.gui.theme import CUSTOM_CSS
from app.pipeline import run_pipeline
from app.pipeline.models import JobResult, ReferenceKind
from app.viz.charts import dashboard_figures, overview_strip

st.set_page_config(
    page_title="Palimpsest — Book Restoration Studio",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme() -> None:
    st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Book Restoration &amp; Intelligence Studio</div>
          <h1>Palimpsest</h1>
          <p>Restore scanned volumes, translate across languages, extract esoteric,
          theological, and scientific references, and explore advanced analytics —
          all from one professional workspace.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_row(result: JobResult) -> None:
    a = result.analytics
    cols = st.columns(6)
    items = [
        ("Pages", f"{a.page_count}"),
        ("Words", f"{a.word_count:,}"),
        ("Chapters", f"{a.chapter_count}"),
        ("Readability", f"{a.readability_score}"),
        ("Diversity", f"{a.lexical_diversity}"),
        ("References", f"{sum(a.reference_counts.values())}"),
    ]
    for col, (label, value) in zip(cols, items):
        col.markdown(
            f'<div class="metric-card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div></div>',
            unsafe_allow_html=True,
        )


def render_references(result: JobResult, kind: ReferenceKind) -> None:
    subset = [r for r in result.references if r.kind == kind]
    if not subset:
        st.info(f"No {kind.value} references detected.")
        return
    st.caption(f"{len(subset)} hits")
    for r in subset[:80]:
        st.markdown(
            f'<div class="ref-excerpt"><b>{r.term}</b> · {r.category} · page {r.page_index + 1}'
            f"{' · ' + r.chapter_title if r.chapter_title else ''}<br/>"
            f"<em>{r.excerpt}</em></div>",
            unsafe_allow_html=True,
        )


def render_results(result: JobResult) -> None:
    st.success(f"Job complete → `{result.output_dir}`")
    if result.warnings:
        for w in result.warnings:
            st.warning(w)

    metric_row(result)
    st.plotly_chart(overview_strip(result), use_container_width=True)

    figs = dashboard_figures(result)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(figs["quality"], use_container_width=True)
        st.plotly_chart(figs["chapters"], use_container_width=True)
        st.plotly_chart(figs["radar"], use_container_width=True)
    with c2:
        st.plotly_chart(figs["references"], use_container_width=True)
        st.plotly_chart(figs["terms"], use_container_width=True)
        st.plotly_chart(figs["languages"], use_container_width=True)

    tabs = st.tabs(
        [
            "Synopses & Insights",
            "Esoteric",
            "Theological",
            "Scientific",
            "Pages",
            "Exports",
        ]
    )

    with tabs[0]:
        for ch in result.chapters:
            st.markdown(f"### {ch.title}")
            st.caption(f"Pages {ch.start_page + 1}–{ch.end_page + 1} · {ch.word_count} words")
            st.write(ch.synopsis)
            for insight in ch.insights:
                st.markdown(f"- {insight}")
            st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)

    with tabs[1]:
        render_references(result, ReferenceKind.ESOTERIC)
    with tabs[2]:
        render_references(result, ReferenceKind.THEOLOGICAL)
    with tabs[3]:
        render_references(result, ReferenceKind.SCIENTIFIC)

    with tabs[4]:
        if result.pages:
            idx = st.slider("Page", 1, len(result.pages), 1) - 1
            page = result.pages[idx]
            left, right = st.columns(2)
            with left:
                st.caption("Original")
                st.image(str(page.original_path), use_container_width=True)
            with right:
                st.caption("Cleaned")
                if page.cleaned_path:
                    st.image(str(page.cleaned_path), use_container_width=True)
            text = result.texts[idx]
            st.markdown("**Extracted text**")
            st.text_area("cleaned", text.cleaned_text, height=160, label_visibility="collapsed")
            if text.translated_text and text.translated_text != text.cleaned_text:
                st.markdown("**Translated**")
                st.text_area("translated", text.translated_text, height=160, label_visibility="collapsed")

    with tabs[5]:
        st.markdown("#### Output folder contents")
        st.code(str(result.output_dir))
        files = {
            "Cleaned PDF": result.cleaned_pdf,
            "Translated PDF": result.translated_pdf,
            "HTML Report": result.report_html,
            "Manifest": result.output_dir / "manifest.json",
            "Chapter synopses": result.output_dir / "synopses" / "chapters.md",
            "All references CSV": result.output_dir / "references" / "all_references.csv",
            "Analytics JSON": result.output_dir / "analytics" / "report.json",
        }
        for label, path in files.items():
            if path and Path(path).exists():
                st.write(f"**{label}:** `{path}`")

        df = pd.DataFrame(
            [
                {
                    "kind": r.kind.value,
                    "category": r.category,
                    "term": r.term,
                    "page": r.page_index + 1,
                    "excerpt": r.excerpt,
                }
                for r in result.references
            ]
        )
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)


def sidebar_controls() -> dict:
    st.sidebar.markdown("## Workspace")
    st.sidebar.caption(f"Palimpsest v{__version__}")

    uploaded = st.sidebar.file_uploader("Input PDF (scanned book)", type=["pdf"])
    input_path = st.sidebar.text_input(
        "Or local PDF path",
        value="",
        placeholder="/path/to/book.pdf",
    )
    output_dir = st.sidebar.text_input(
        "Output folder",
        value=str(Path.cwd() / "outputs"),
    )

    st.sidebar.markdown("### Processing")
    dpi = st.sidebar.slider("Render DPI", 120, 300, SETTINGS.default_dpi, 20)
    max_pages = st.sidebar.number_input("Max pages (0 = all)", min_value=0, value=0, step=1)
    clean_mode = st.sidebar.selectbox("Clean mode", ["auto", "gentle", "aggressive"], index=0)
    binarize = st.sidebar.checkbox("Binarize pages", value=True)
    enable_ocr = st.sidebar.checkbox("Enable OCR", value=True)
    ocr_lang = st.sidebar.selectbox(
        "OCR / source language hint",
        options=list(SUPPORTED_TRANSLATE_LANGS.keys()),
        format_func=lambda k: SUPPORTED_TRANSLATE_LANGS[k],
        index=0,
    )
    enable_translation = st.sidebar.checkbox("Translate wording", value=True)
    translate_target = st.sidebar.selectbox(
        "Translate to",
        options=list(SUPPORTED_TRANSLATE_LANGS.keys()),
        format_func=lambda k: SUPPORTED_TRANSLATE_LANGS[k],
        index=0,
    )

    run = st.sidebar.button("Restore & Analyze", type="primary", use_container_width=True)

    return {
        "uploaded": uploaded,
        "input_path": input_path.strip(),
        "output_dir": output_dir.strip(),
        "dpi": dpi,
        "max_pages": None if max_pages == 0 else int(max_pages),
        "clean_mode": clean_mode,
        "binarize": binarize,
        "enable_ocr": enable_ocr,
        "ocr_lang": ocr_lang,
        "enable_translation": enable_translation,
        "translate_target": translate_target,
        "run": run,
    }


def resolve_input(ctrl: dict) -> Path | None:
    if ctrl["uploaded"] is not None:
        staging = Path(ctrl["output_dir"]).expanduser() / "_staging"
        staging.mkdir(parents=True, exist_ok=True)
        dest = staging / ctrl["uploaded"].name
        dest.write_bytes(ctrl["uploaded"].getvalue())
        return dest
    if ctrl["input_path"]:
        path = Path(ctrl["input_path"]).expanduser()
        if path.exists():
            return path
        st.error(f"PDF not found: {path}")
        return None
    return None


def main() -> None:
    inject_theme()
    hero()
    ctrl = sidebar_controls()

    if "result" in st.session_state and not ctrl["run"]:
        render_results(st.session_state["result"])
        return

    if not ctrl["run"]:
        st.markdown(
            """
            ### How it works
            1. Select a scanned PDF via the sidebar uploader or local path.
            2. Choose an output folder — Palimpsest writes originals, cleaned images,
               cleaned PDF, translated PDF, synopses, reference CSVs, analytics, and an HTML report.
            3. Tune DPI, cleaning, OCR language, and translation target.
            4. Click **Restore & Analyze** to run the full pipeline.
            """
        )
        st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Clean**\n\nDeskew, denoise, CLAHE contrast, adaptive binarization.")
        c2.markdown("**Understand**\n\nChapter synopses, insights, multi-language translation.")
        c3.markdown("**Reveal**\n\nEsoteric · theological · scientific reference excerpts.")
        return

    pdf_path = resolve_input(ctrl)
    if pdf_path is None:
        st.warning("Provide an input PDF to begin.")
        return

    progress = st.progress(0.0, text="Starting…")
    status = st.empty()

    def on_progress(pct: float, msg: str) -> None:
        progress.progress(min(max(pct, 0.0), 1.0), text=msg)
        status.caption(msg)

    try:
        result = run_pipeline(
            pdf_path,
            Path(ctrl["output_dir"]),
            dpi=ctrl["dpi"],
            max_pages=ctrl["max_pages"],
            clean_mode=ctrl["clean_mode"],
            binarize=ctrl["binarize"],
            enable_ocr=ctrl["enable_ocr"],
            ocr_lang_code=ctrl["ocr_lang"],
            enable_translation=ctrl["enable_translation"],
            translate_target=ctrl["translate_target"],
            progress=on_progress,
        )
        st.session_state["result"] = result
        progress.progress(1.0, text="Complete")
        render_results(result)
    except Exception as exc:  # noqa: BLE001
        st.exception(exc)


if __name__ == "__main__":
    main()
