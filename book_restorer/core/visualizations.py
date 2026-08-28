"""Plotly-based visualization builders for the analytics dashboard and
exported reports. All functions return either a ``plotly.graph_objects.Figure``
or a PIL image (word cloud), so they can be reused identically by the
Streamlit GUI and the static HTML/PDF report generator.
"""

from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Optional

import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

_TEMPLATE = "plotly_dark"
_ACCENT_SEQUENCE = ["#8b5cf6", "#22d3ee", "#f472b6", "#facc15", "#34d399", "#fb923c", "#60a5fa"]


def word_frequency_bar(word_freq: List[tuple], title: str = "Most Frequent Words") -> go.Figure:
    words = [w for w, _ in word_freq][::-1]
    counts = [c for _, c in word_freq][::-1]
    fig = go.Figure(go.Bar(x=counts, y=words, orientation="h", marker=dict(color=counts, colorscale="Purples")))
    fig.update_layout(title=title, template=_TEMPLATE, height=max(400, 20 * len(words)), margin=dict(l=10, r=10, t=60, b=10))
    return fig


def sentiment_arc_line(sentiment_by_chapter: List[Dict], title: str = "Sentiment Arc by Chapter") -> go.Figure:
    if not sentiment_by_chapter:
        return go.Figure()
    x = [f"Ch. {s['chapter']}" for s in sentiment_by_chapter]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[s["compound"] for s in sentiment_by_chapter], mode="lines+markers",
                              name="Compound sentiment", line=dict(color="#8b5cf6", width=3), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=x, y=[s["pos"] for s in sentiment_by_chapter], mode="lines", name="Positive",
                              line=dict(color="#34d399", dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=[s["neg"] for s in sentiment_by_chapter], mode="lines", name="Negative",
                              line=dict(color="#f472b6", dash="dot")))
    fig.update_layout(title=title, template=_TEMPLATE, yaxis_title="VADER score", margin=dict(l=10, r=10, t=60, b=10))
    return fig


def reference_category_pie(reference_stats: Dict[str, Dict], title: str = "Reference Mentions by Category") -> go.Figure:
    labels = list(reference_stats.keys())
    values = [reference_stats[k]["total_matches"] for k in labels]
    fig = go.Figure(go.Pie(labels=[l.title() for l in labels], values=values, hole=0.45,
                            marker=dict(colors=_ACCENT_SEQUENCE)))
    fig.update_layout(title=title, template=_TEMPLATE, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def readability_gauge(flesch_score: float, title: str = "Flesch Reading Ease") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=max(0, min(100, flesch_score)),
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#8b5cf6"},
            "steps": [
                {"range": [0, 30], "color": "#3f1d3d"},
                {"range": [30, 60], "color": "#5b3a6e"},
                {"range": [60, 100], "color": "#22577a"},
            ],
        },
    ))
    fig.update_layout(template=_TEMPLATE, margin=dict(l=10, r=10, t=60, b=10), height=320)
    return fig


def chapter_length_bar(chapters, title: str = "Chapter Length (words)") -> go.Figure:
    from core.utils import word_tokenize_basic

    labels = [f"Ch. {c.number}: {c.title[:24]}" for c in chapters]
    lengths = [len(word_tokenize_basic(c.text)) for c in chapters]
    fig = go.Figure(go.Bar(x=labels, y=lengths, marker=dict(color=lengths, colorscale="Tealgrn")))
    fig.update_layout(title=title, template=_TEMPLATE, margin=dict(l=10, r=10, t=60, b=80), xaxis_tickangle=-30)
    return fig


def language_distribution_pie(language_counts: Dict[str, int], title: str = "Detected Language per Page") -> go.Figure:
    fig = go.Figure(go.Pie(labels=list(language_counts.keys()), values=list(language_counts.values()),
                            marker=dict(colors=_ACCENT_SEQUENCE)))
    fig.update_layout(title=title, template=_TEMPLATE, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def entity_frequency_bar(entities: List[tuple], title: str = "Notable Names & Recurring Entities") -> go.Figure:
    labels = [e for e, _ in entities][::-1]
    counts = [c for _, c in entities][::-1]
    fig = go.Figure(go.Bar(x=counts, y=labels, orientation="h", marker=dict(color=counts, colorscale="Sunsetdark")))
    fig.update_layout(title=title, template=_TEMPLATE, height=max(400, 22 * len(labels)), margin=dict(l=10, r=10, t=60, b=10))
    return fig


def noise_reduction_bar(page_metrics: List[Dict], title: str = "Per-page Noise Reduction") -> go.Figure:
    pages = list(range(1, len(page_metrics) + 1))
    reductions = []
    for m in page_metrics:
        before, after = m.get("noise_before", 0.0), m.get("noise_after", 0.0)
        reductions.append(max(0.0, (before - after) / before * 100) if before > 0 else 0.0)
    fig = go.Figure(go.Bar(x=pages, y=reductions, marker=dict(color=reductions, colorscale="Bluyl")))
    fig.update_layout(title=title, template=_TEMPLATE, xaxis_title="Page", yaxis_title="% noise reduced",
                       margin=dict(l=10, r=10, t=60, b=10))
    return fig


def generate_wordcloud_image(word_freq: List[tuple], width: int = 1000, height: int = 500) -> Optional[Image.Image]:
    if not word_freq:
        return None
    from wordcloud import WordCloud

    frequencies = dict(word_freq)
    wc = WordCloud(
        width=width,
        height=height,
        background_color="#0f0f1a",
        colormap="magma",
        prefer_horizontal=0.9,
    ).generate_from_frequencies(frequencies)
    return wc.to_image()


def fig_to_png_bytes(fig: go.Figure, width: int = 1000, height: int = 600) -> Optional[bytes]:
    try:
        return fig.to_image(format="png", width=width, height=height)
    except TypeError:
        # Older kaleido/plotly combinations require an explicit engine kwarg.
        try:
            return fig.to_image(format="png", width=width, height=height, engine="kaleido")
        except Exception:
            return None
    except Exception:
        return None
