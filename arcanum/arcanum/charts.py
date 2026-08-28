"""Publication-quality analytics charts (matplotlib, Agg backend).

All charts share a dark parchment-and-gold visual identity matched to
the GUI and the HTML report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .analytics import BookAnalytics

BG = "#12141c"
PANEL = "#1a1d29"
FG = "#e8e2d4"
MUTED = "#8b8fa3"
GOLD = "#d4af6a"
TEAL = "#5ec8b8"
VIOLET = "#a98fd6"
ROSE = "#d67f8f"
CATEGORY_COLORS = {"esoteric": VIOLET, "theological": GOLD, "scientific": TEAL}

_STYLE = {
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": "#2c3040",
    "axes.labelcolor": FG,
    "axes.titlecolor": FG,
    "text.color": FG,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": "#262a38",
    "grid.linestyle": "-",
    "grid.linewidth": 0.7,
    "font.family": "DejaVu Serif",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "savefig.facecolor": BG,
    "savefig.dpi": 150,
}


def _new_fig(width: float = 8.6, height: float = 4.4):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.grid(True, axis="y", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return fig, ax


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_all_charts(analytics: BookAnalytics, out_dir: str | Path) -> dict[str, Path]:
    """Render every chart; returns {chart key: png path}."""
    out = Path(out_dir)
    charts: dict[str, Path] = {}
    with plt.rc_context(_STYLE):
        charts["ocr_confidence"] = _chart_ocr_confidence(analytics, out)
        charts["category_donut"] = _chart_category_donut(analytics, out)
        charts["top_words"] = _chart_top_words(analytics, out)
        charts["chapter_words"] = _chart_chapter_words(analytics, out)
        charts["density_heatmap"] = _chart_density_heatmap(analytics, out)
        charts["restoration"] = _chart_restoration(analytics, out)
    return {k: v for k, v in charts.items() if v is not None}


def _chart_ocr_confidence(a: BookAnalytics, out: Path) -> Path | None:
    if not a.ocr_confidence_by_page:
        return None
    fig, ax = _new_fig()
    pages = np.arange(1, len(a.ocr_confidence_by_page) + 1)
    conf = np.array(a.ocr_confidence_by_page)
    ax.fill_between(pages, conf, color=TEAL, alpha=0.18, zorder=2)
    ax.plot(pages, conf, color=TEAL, linewidth=2.2, zorder=3, marker="o", markersize=4 if len(pages) <= 60 else 0)
    ax.axhline(a.mean_ocr_confidence, color=GOLD, linewidth=1.2, linestyle="--", zorder=3)
    ax.annotate(
        f"mean {a.mean_ocr_confidence}%",
        xy=(pages[-1], a.mean_ocr_confidence),
        xytext=(-4, 6),
        textcoords="offset points",
        ha="right",
        color=GOLD,
        fontsize=9,
    )
    ax.set_title("OCR Confidence by Page")
    ax.set_xlabel("Page")
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 105)
    return _save(fig, out / "ocr_confidence.png")


def _chart_category_donut(a: BookAnalytics, out: Path) -> Path | None:
    counts = {k: v for k, v in a.category_counts.items() if v > 0}
    if not counts:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    colors = [CATEGORY_COLORS.get(k, MUTED) for k in counts]
    wedges, _texts, autotexts = ax.pie(
        counts.values(),
        labels=[k.title() for k in counts],
        colors=colors,
        autopct="%1.0f%%",
        pctdistance=0.78,
        startangle=100,
        wedgeprops={"width": 0.42, "edgecolor": BG, "linewidth": 2},
        textprops={"color": FG, "fontsize": 11},
    )
    for t in autotexts:
        t.set_color(BG)
        t.set_fontweight("bold")
    total = sum(counts.values())
    ax.text(0, 0.05, f"{total:,}", ha="center", va="center", fontsize=22, fontweight="bold", color=FG)
    ax.text(0, -0.18, "passages", ha="center", va="center", fontsize=10, color=MUTED)
    ax.set_title("Reference Passages by Tradition")
    return _save(fig, out / "category_distribution.png")


def _chart_top_words(a: BookAnalytics, out: Path) -> Path | None:
    if not a.top_words:
        return None
    words, counts = zip(*a.top_words[:18][::-1])
    fig, ax = _new_fig(8.6, 5.6)
    ax.grid(True, axis="x", zorder=0)
    ax.grid(False, axis="y")
    colors = [GOLD if i >= len(words) - 5 else "#9a7f4e" for i in range(len(words))]
    ax.barh(words, counts, color=colors, edgecolor=BG, zorder=3, height=0.72)
    ax.set_title("Most Frequent Content Words")
    ax.set_xlabel("Occurrences")
    return _save(fig, out / "top_words.png")


def _chart_chapter_words(a: BookAnalytics, out: Path) -> Path | None:
    if not a.words_by_chapter:
        return None
    labels = list(a.words_by_chapter)
    values = list(a.words_by_chapter.values())
    fig, ax = _new_fig()
    ax.bar(labels, values, color=TEAL, edgecolor=BG, zorder=3, width=0.66)
    ax.set_title("Words per Chapter")
    ax.set_ylabel("Words")
    if len(labels) > 12:
        ax.tick_params(axis="x", rotation=60, labelsize=8)
    return _save(fig, out / "chapter_lengths.png")


def _chart_density_heatmap(a: BookAnalytics, out: Path) -> Path | None:
    density = a.category_density_by_chapter
    categories = [c for c in ("esoteric", "theological", "scientific") if density.get(c)]
    if not categories:
        return None
    chapter_nums = sorted({n for c in categories for n in density[c]})
    if not chapter_nums:
        return None
    matrix = np.array([[density[c].get(n, 0.0) for n in chapter_nums] for c in categories])
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(chapter_nums)))
    ax.set_xticklabels(
        [f"Ch. {n}" if n else "Front" for n in chapter_nums], fontsize=8, rotation=45 if len(chapter_nums) > 10 else 0
    )
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([c.title() for c in categories], fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("hits / 1k words", fontsize=8, color=MUTED)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title("Reference Density Across Chapters")
    return _save(fig, out / "density_heatmap.png")


def _chart_restoration(a: BookAnalytics, out: Path) -> Path | None:
    if not a.contrast_before:
        return None
    pages = np.arange(1, len(a.contrast_before) + 1)
    fig, ax = _new_fig()
    ax.plot(pages, a.contrast_before, color=ROSE, linewidth=1.8, label="Original scan", zorder=3)
    ax.plot(pages, a.contrast_after, color=TEAL, linewidth=2.2, label="Restored", zorder=4)
    ax.fill_between(
        pages,
        a.contrast_before,
        a.contrast_after,
        where=np.array(a.contrast_after) >= np.array(a.contrast_before),
        color=TEAL,
        alpha=0.12,
        zorder=2,
    )
    ax.set_title("Page Contrast — Before vs. After Restoration")
    ax.set_xlabel("Page")
    ax.set_ylabel("Contrast (std. dev.)")
    ax.legend(facecolor=PANEL, edgecolor="#2c3040", labelcolor=FG, fontsize=9)
    return _save(fig, out / "restoration_quality.png")
