"""Arcanum main window: configuration sidebar + results workspace."""

from __future__ import annotations

import html
from pathlib import Path
import sys
import webbrowser

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import style
from .. import __app_name__, __tagline__, __version__
from ..ocr import available_languages
from ..pipeline import PipelineOptions, PipelineResult
from ..translate import SUPPORTED_LANGUAGES
from .widgets import ChartPanel, PageComparator, StatCard
from .worker import PipelineWorker

_CHART_TITLES = {
    "ocr_confidence": "OCR Confidence by Page",
    "category_donut": "References by Tradition",
    "top_words": "Most Frequent Words",
    "chapter_words": "Words per Chapter",
    "density_heatmap": "Reference Density",
    "restoration": "Restoration Quality",
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{__app_name__} — {__tagline__}")
        self.resize(1460, 920)
        self._worker: PipelineWorker | None = None
        self._result: PipelineResult | None = None

        root = QSplitter(Qt.Horizontal)
        root.addWidget(self._build_sidebar())
        root.addWidget(self._build_workspace())
        root.setStretchFactor(1, 1)
        root.setSizes([360, 1100])
        self.setCentralWidget(root)
        self.statusBar().showMessage(
            f"{__app_name__} v{__version__} ready — choose a scanned book "
            f"PDF and an output folder, then press Analyze."
        )

    # ------------------------------------------------------------------
    # Sidebar (configuration)
    # ------------------------------------------------------------------
    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("Sidebar")
        sidebar.setAttribute(Qt.WA_StyledBackground, True)
        sidebar.setMinimumWidth(330)
        sidebar.setMaximumWidth(400)
        outer = QVBoxLayout(sidebar)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(22, 26, 22, 20)
        layout.setSpacing(9)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        title = QLabel(__app_name__.upper())
        title.setObjectName("AppTitle")
        tagline = QLabel(__tagline__.upper())
        tagline.setObjectName("AppTagline")
        layout.addWidget(title)
        layout.addWidget(tagline)
        layout.addSpacing(14)

        layout.addWidget(self._section("SOURCE"))
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Scanned book PDF…")
        layout.addLayout(self._picker_row(self.input_edit, self._pick_input))

        layout.addWidget(self._section("DESTINATION"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Output folder…")
        layout.addLayout(self._picker_row(self.output_edit, self._pick_output))

        layout.addWidget(self._section("OCR"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.addWidget(QLabel("Language"), 0, 0)
        self.ocr_lang_combo = QComboBox()
        for lang in available_languages():
            if lang != "osd":
                self.ocr_lang_combo.addItem(lang)
        index = self.ocr_lang_combo.findText("eng")
        if index >= 0:
            self.ocr_lang_combo.setCurrentIndex(index)
        grid.addWidget(self.ocr_lang_combo, 0, 1)
        grid.addWidget(QLabel("Detail (DPI)"), 1, 0)
        self.dpi_combo = QComboBox()
        for dpi in ("200", "250", "300", "400"):
            self.dpi_combo.addItem(dpi)
        self.dpi_combo.setCurrentText("250")
        grid.addWidget(self.dpi_combo, 1, 1)
        grid.addWidget(QLabel("Max pages"), 2, 0)
        self.max_pages_spin = QSpinBox()
        self.max_pages_spin.setRange(0, 10000)
        self.max_pages_spin.setSpecialValueText("All")
        grid.addWidget(self.max_pages_spin, 2, 1)
        layout.addLayout(grid)

        self.deskew_check = QCheckBox("Deskew && restore pages")
        self.deskew_check.setChecked(True)
        self.export_images_check = QCheckBox("Export page images")
        self.export_images_check.setChecked(True)
        self.embedded_check = QCheckBox("Extract embedded images " "(original format)")
        self.embedded_check.setChecked(True)
        layout.addWidget(self.deskew_check)
        layout.addWidget(self.export_images_check)
        layout.addWidget(self.embedded_check)

        layout.addWidget(self._section("TRANSLATE INTO"))
        self.language_list = QListWidget()
        self.language_list.setMaximumHeight(120)
        for code, name in SUPPORTED_LANGUAGES.items():
            item = QListWidgetItem(f"{name}  ({code})")
            item.setData(Qt.UserRole, code)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.language_list.addItem(item)
        layout.addWidget(self.language_list)

        layout.addWidget(self._section("SYNOPSIS"))
        row = QHBoxLayout()
        row.addWidget(QLabel("Sentences per chapter"))
        self.synopsis_spin = QSpinBox()
        self.synopsis_spin.setRange(2, 12)
        self.synopsis_spin.setValue(5)
        row.addWidget(self.synopsis_spin)
        layout.addLayout(row)

        layout.addSpacing(10)
        self.run_button = QPushButton("✦  Analyze Book")
        self.run_button.setObjectName("RunButton")
        self.run_button.clicked.connect(self._start)
        layout.addWidget(self.run_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("GhostButton")
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.setEnabled(False)
        layout.addWidget(self.cancel_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        self.stage_label = QLabel("Idle")
        self.stage_label.setStyleSheet(f"color: {style.MUTED}; font-size: 12px;")
        self.stage_label.setWordWrap(True)
        layout.addWidget(self.stage_label)

        layout.addStretch(1)
        return sidebar

    def _section(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("SectionLabel")
        return label

    def _picker_row(self, edit: QLineEdit, slot) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(edit, 1)
        button = QPushButton("Browse…")
        button.clicked.connect(slot)
        row.addWidget(button)
        return row

    def _pick_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a scanned book PDF", "", "PDF files (*.pdf)")
        if path:
            self.input_edit.setText(path)

    def _pick_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if path:
            self.output_edit.setText(path)

    # ------------------------------------------------------------------
    # Workspace (results)
    # ------------------------------------------------------------------
    def _build_workspace(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 16, 16, 12)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.tabs.addTab(self._build_dashboard_tab(), "Dashboard")
        self.page_comparator = PageComparator()
        self.tabs.addTab(self.page_comparator, "Pages")
        self.synopsis_view = self._make_browser()
        self.tabs.addTab(self.synopsis_view, "Synopsis")
        self.tabs.addTab(self._build_references_tab(), "References")
        self.insights_view = self._make_browser()
        self.tabs.addTab(self.insights_view, "Insights")
        self.translations_view = self._make_browser()
        self.tabs.addTab(self.translations_view, "Translations")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Monospace", 10))
        self.tabs.addTab(self.log_view, "Log")
        return container

    def _make_browser(self) -> QTextBrowser:
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(_placeholder_html("Run an analysis to populate " "this view."))
        return browser

    def _build_dashboard_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(4, 8, 4, 4)

        cards_row = QHBoxLayout()
        self.stat_cards: dict[str, StatCard] = {}
        for key, label in (
            ("pages", "Pages"),
            ("words", "Words"),
            ("chapters", "Chapters"),
            ("confidence", "OCR Confidence"),
            ("references", "References"),
            ("ease", "Reading Ease"),
        ):
            card = StatCard(label)
            self.stat_cards[key] = card
            cards_row.addWidget(card)
        outer.addLayout(cards_row)

        buttons_row = QHBoxLayout()
        self.open_report_button = QPushButton("Open Illustrated Report")
        self.open_report_button.clicked.connect(self._open_report)
        self.open_report_button.setEnabled(False)
        self.open_folder_button = QPushButton("Open Output Folder")
        self.open_folder_button.clicked.connect(self._open_folder)
        self.open_folder_button.setEnabled(False)
        buttons_row.addWidget(self.open_report_button)
        buttons_row.addWidget(self.open_folder_button)
        buttons_row.addStretch(1)
        outer.addLayout(buttons_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_host = QWidget()
        self.chart_grid = QGridLayout(grid_host)
        self.chart_grid.setSpacing(14)
        self.chart_panels: dict[str, ChartPanel] = {}
        for i, key in enumerate(_CHART_TITLES):
            panel = ChartPanel()
            self.chart_panels[key] = panel
            self.chart_grid.addWidget(panel, i // 2, i % 2)
        scroll.setWidget(grid_host)
        outer.addWidget(scroll, 1)
        return tab

    def _build_references_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 8, 4, 4)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Tradition:"))
        self.category_filter = QComboBox()
        self.category_filter.addItems(["All", "Esoteric", "Theological", "Scientific", "Syncretic"])
        self.category_filter.currentTextChanged.connect(lambda _: self._populate_references())
        controls.addWidget(self.category_filter)
        controls.addStretch(1)
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Vertical)
        self.reference_tree = QTreeWidget()
        self.reference_tree.setHeaderLabels(["Tradition", "Chapter", "Pages", "Terms", "Relevance"])
        self.reference_tree.setRootIsDecorated(False)
        self.reference_tree.setAlternatingRowColors(False)
        self.reference_tree.currentItemChanged.connect(self._show_excerpt)
        self.reference_tree.setColumnWidth(0, 130)
        self.reference_tree.setColumnWidth(1, 200)
        self.reference_tree.setColumnWidth(3, 340)
        splitter.addWidget(self.reference_tree)

        self.excerpt_view = QTextBrowser()
        self.excerpt_view.setHtml(_placeholder_html("Select a reference to read the excerpt."))
        splitter.addWidget(self.excerpt_view)
        splitter.setSizes([460, 220])
        layout.addWidget(splitter, 1)
        return tab

    # ------------------------------------------------------------------
    # Pipeline control
    # ------------------------------------------------------------------
    def _start(self) -> None:
        input_pdf = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        if not input_pdf or not Path(input_pdf).is_file():
            QMessageBox.warning(self, __app_name__, "Please choose an input PDF file.")
            return
        if not output_dir:
            QMessageBox.warning(self, __app_name__, "Please choose an output folder.")
            return

        targets = []
        for i in range(self.language_list.count()):
            item = self.language_list.item(i)
            if item.checkState() == Qt.Checked:
                targets.append(item.data(Qt.UserRole))

        options = PipelineOptions(
            input_pdf=input_pdf,
            output_dir=output_dir,
            dpi=int(self.dpi_combo.currentText()),
            ocr_language=self.ocr_lang_combo.currentText(),
            translate_to=targets,
            deskew=self.deskew_check.isChecked(),
            max_pages=self.max_pages_spin.value() or None,
            synopsis_sentences=self.synopsis_spin.value(),
            export_original_images=self.export_images_check.isChecked(),
            export_cleaned_images=self.export_images_check.isChecked(),
            extract_embedded=self.embedded_check.isChecked(),
        )

        self.log_view.clear()
        self._log(f"Starting analysis of {input_pdf}")
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)

        self._worker = PipelineWorker(options)
        self._worker.progressed.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.start()

    def _cancel(self) -> None:
        if self._worker:
            self._worker.cancel()
            self.stage_label.setText("Cancelling…")

    def _on_progress(self, pct: int, msg: str) -> None:
        self.progress_bar.setValue(pct)
        self.stage_label.setText(msg)
        self._log(f"[{pct:3d}%] {msg}")

    def _on_failed(self, trace: str) -> None:
        self._reset_controls("Failed.")
        self._log(trace)
        QMessageBox.critical(self, __app_name__, "Processing failed — see the Log tab for " "details.")

    def _on_cancelled(self) -> None:
        self._reset_controls("Cancelled.")
        self._log("Cancelled by user.")

    def _reset_controls(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.stage_label.setText(message)

    def _log(self, message: str) -> None:
        self.log_view.appendPlainText(message)

    # ------------------------------------------------------------------
    # Result rendering
    # ------------------------------------------------------------------
    def _on_finished(self, result: PipelineResult) -> None:
        self._result = result
        self._reset_controls(f"Complete in {result.elapsed_seconds:.0f}s — output in " f"{result.output_dir}")
        self._log(f"Finished in {result.elapsed_seconds:.0f}s")
        for warning in result.warnings:
            self._log(f"warning: {warning}")

        a = result.analytics
        if a:
            self.stat_cards["pages"].set_value(f"{a.page_count:,}")
            self.stat_cards["words"].set_value(f"{a.word_count:,}")
            self.stat_cards["chapters"].set_value(str(len(result.chapters)))
            self.stat_cards["confidence"].set_value(f"{a.mean_ocr_confidence:.0f}%")
            self.stat_cards["references"].set_value(f"{sum(a.category_counts.values()):,}")
            self.stat_cards["ease"].set_value(f"{a.flesch_score:.0f}")

        for key, panel in self.chart_panels.items():
            path = result.chart_paths.get(key)
            if path and Path(path).exists():
                panel.set_chart(path)

        self.page_comparator.load_pages(result.original_image_paths, result.cleaned_image_paths)
        self.synopsis_view.setHtml(_synopsis_html(result))
        self.insights_view.setHtml(_insights_html(result))
        self.translations_view.setHtml(_translations_html(result))
        self._populate_references()

        self.open_report_button.setEnabled(result.report_html is not None)
        self.open_folder_button.setEnabled(True)
        self.statusBar().showMessage(f"Analysis complete — {result.output_dir}")

    def _populate_references(self) -> None:
        self.reference_tree.clear()
        if not self._result or not self._result.references:
            return
        wanted = self.category_filter.currentText().lower()
        refs = self._result.references

        rows = []
        for category, hits in refs.hits.items():
            if wanted not in ("all", category):
                continue
            rows.extend(hits)
        if wanted in ("all", "syncretic"):
            rows.extend(refs.syncretic)

        for hit in rows:
            item = QTreeWidgetItem(
                [
                    hit.category,
                    f"{hit.chapter_number}. {hit.chapter_title}",
                    f"{hit.start_page}–{hit.end_page}",
                    ", ".join(hit.terms),
                    f"{hit.score:.1f}",
                ]
            )
            base = hit.category.split("+")[0]
            color = style.CATEGORY_COLORS.get(base if "+" not in hit.category else "syncretic", style.MUTED)
            item.setForeground(0, Qt.GlobalColor.white)
            item.setData(0, Qt.UserRole, hit.excerpt)
            item.setToolTip(3, ", ".join(hit.terms))
            from PySide6.QtGui import QBrush, QColor

            item.setForeground(0, QBrush(QColor(color)))
            self.reference_tree.addTopLevelItem(item)

    def _show_excerpt(self, current, _previous) -> None:
        if not current:
            return
        excerpt = current.data(0, Qt.UserRole) or ""
        category = current.text(0)
        chapter = current.text(1)
        color = style.CATEGORY_COLORS.get(category.split("+")[0], style.GOLD)
        self.excerpt_view.setHtml(f"""
        <div style="font-family: Georgia, serif; color: {style.INK};
                    padding: 8px;">
        <div style="color: {color}; font-size: 11px; letter-spacing: 2px;
                    text-transform: uppercase;">{html.escape(category)}
                    · {html.escape(chapter)}</div>
        <blockquote style="border-left: 3px solid {color};
                    margin: 10px 0; padding: 8px 14px; font-style: italic;
                    font-size: 15px;">{html.escape(excerpt)}</blockquote>
        </div>""")

    def _open_report(self) -> None:
        if self._result and self._result.report_html:
            webbrowser.open(Path(self._result.report_html).as_uri())

    def _open_folder(self) -> None:
        if self._result and self._result.output_dir:
            webbrowser.open(Path(self._result.output_dir).as_uri())


# ----------------------------------------------------------------------
# HTML fragments for text views
# ----------------------------------------------------------------------


def _base_css() -> str:
    return (
        f"body {{ font-family: Georgia, serif; color: {style.INK}; "
        f"background: {style.PANEL}; margin: 14px; }} "
        f"h2 {{ color: {style.GOLD}; font-weight: normal; }} "
        f"h3 {{ color: {style.INK}; margin-bottom: 2px; }} "
        f".meta {{ color: {style.MUTED}; font-size: 12px; }} "
        f".kw {{ color: {style.TEAL}; font-size: 13px; }} "
        f"blockquote {{ border-left: 3px solid {style.GOLD}; "
        f"margin: 8px 0; padding: 6px 14px; font-style: italic; }}"
    )


def _placeholder_html(message: str) -> str:
    return (
        f"<html><head><style>{_base_css()}</style></head><body>"
        f"<p style='color:{style.MUTED}; text-align:center; "
        f"margin-top:60px;'>{html.escape(message)}</p></body></html>"
    )


def _synopsis_html(result: PipelineResult) -> str:
    parts = [
        f"<html><head><style>{_base_css()}</style></head><body>",
        f"<h2>{html.escape(result.book_name)} — Synopsis by " f"Chapter</h2>",
    ]
    for s in result.synopses:
        parts.append(
            f"<h3>{s.chapter_number:02d}. {html.escape(s.chapter_title)}"
            f"</h3><div class='meta'>Pages {s.start_page}–{s.end_page} · "
            f"{s.word_count:,} words</div>"
            f"<p>{html.escape(s.summary) or '<i>No readable text.</i>'}</p>"
            f"<div class='kw'>Keywords: "
            f"{html.escape(', '.join(s.keywords))}</div><hr/>"
        )
    parts.append("</body></html>")
    return "".join(parts)


def _insights_html(result: PipelineResult) -> str:
    insights = result.insights
    parts = [f"<html><head><style>{_base_css()}</style></head><body>", "<h2>Insights</h2>"]
    if insights:
        for insight in insights.insights:
            parts.append(f"<h3>{html.escape(insight.title)}</h3>" f"<p>{html.escape(insight.detail)}</p>")
        if insights.notable_quotes:
            parts.append("<h2>Notable Passages</h2>")
            for quote in insights.notable_quotes:
                parts.append(f"<blockquote>{html.escape(quote)}" f"</blockquote>")
    parts.append("</body></html>")
    return "".join(parts)


def _translations_html(result: PipelineResult) -> str:
    parts = [f"<html><head><style>{_base_css()}</style></head><body>", "<h2>Translations</h2>"]
    if not result.translations:
        parts.append(f"<p class='meta'>No translation languages were " f"selected.</p>")
    for t in result.translations:
        status = "✓ complete" if t.ok else f"failed — {html.escape(t.error)}"
        parts.append(f"<h3>{html.escape(t.language_name)}</h3>" f"<div class='meta'>{status}</div>")
        if t.ok and t.text:
            preview = t.text[:1200]
            parts.append(
                f"<blockquote>{html.escape(preview)}…"
                f"</blockquote>"
                f"<div class='meta'>Full text saved under "
                f"translations/{t.language}/</div>"
            )
    parts.append("</body></html>")
    return "".join(parts)


def launch() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(__app_name__)
    app.setStyleSheet(style.STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()
