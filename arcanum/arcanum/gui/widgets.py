"""Reusable GUI widgets: stat cards, chart panels, page comparator."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from . import style


class StatCard(QFrame):
    """Small dashboard card with a big number and a caption."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame {{ background: {style.PANEL2}; border: 1px solid " f"{style.LINE}; border-radius: 10px; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(2)
        self.value_label = QLabel("—")
        self.value_label.setStyleSheet(f"color: {style.GOLD}; font-size: 24px; border: none;")
        caption = QLabel(label.upper())
        caption.setStyleSheet(f"color: {style.MUTED}; font-size: 10px; letter-spacing: 2px; " f"border: none;")
        layout.addWidget(self.value_label)
        layout.addWidget(caption)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class ChartPanel(QFrame):
    """Framed chart image that rescales with the window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame {{ background: {style.PANEL2}; border: 1px solid " f"{style.LINE}; border-radius: 10px; }}"
        )
        self._pixmap: QPixmap | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        self._label = QLabel("")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("border: none;")
        self._label.setMinimumHeight(200)
        layout.addWidget(self._label)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def set_chart(self, path: str | Path) -> None:
        self._pixmap = QPixmap(str(path))
        self._rescale()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if not self._pixmap or self._pixmap.isNull():
            return
        width = max(200, self.width() - 24)
        scaled = self._pixmap.scaledToWidth(width, Qt.SmoothTransformation)
        self._label.setPixmap(scaled)
        self._label.setMinimumHeight(scaled.height())


class PageComparator(QWidget):
    """Page list on the left, original vs. restored preview on the right."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._originals: list[Path] = []
        self._cleaned: list[Path] = []

        splitter = QSplitter(Qt.Horizontal, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.page_list = QListWidget()
        self.page_list.setMaximumWidth(150)
        self.page_list.currentRowChanged.connect(self._show_page)
        splitter.addWidget(self.page_list)

        panel = QWidget()
        panel_layout = QHBoxLayout(panel)
        panel_layout.setContentsMargins(8, 0, 0, 0)
        self._before_label, before_box = self._make_view("ORIGINAL SCAN")
        self._after_label, after_box = self._make_view("RESTORED")
        panel_layout.addWidget(before_box)
        panel_layout.addWidget(after_box)
        splitter.addWidget(panel)
        splitter.setStretchFactor(1, 1)

    def _make_view(self, title: str) -> tuple[QLabel, QWidget]:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        caption = QLabel(title)
        caption.setAlignment(Qt.AlignCenter)
        caption.setStyleSheet(f"color: {style.MUTED}; font-size: 10px; letter-spacing: 2px;")
        image = QLabel()
        image.setAlignment(Qt.AlignCenter)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(image)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: 1px solid {style.LINE}; " f"border-radius: 8px; background: {style.BG}; }}"
        )
        v.addWidget(caption)
        v.addWidget(scroll)
        return image, box

    def load_pages(self, originals: list[Path], cleaned: list[Path]) -> None:
        self._originals = originals
        self._cleaned = cleaned
        self.page_list.clear()
        for i in range(max(len(originals), len(cleaned))):
            self.page_list.addItem(f"Page {i + 1}")
        if self.page_list.count():
            self.page_list.setCurrentRow(0)

    def _show_page(self, row: int) -> None:
        if row < 0:
            return
        for paths, label in ((self._originals, self._before_label), (self._cleaned, self._after_label)):
            if row < len(paths) and Path(paths[row]).exists():
                pix = QPixmap(str(paths[row]))
                label.setPixmap(pix.scaledToWidth(620, Qt.SmoothTransformation))
            else:
                label.setText("not exported")
