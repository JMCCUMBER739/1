"""Plotly visualizations for analytics dashboard."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.pipeline.models import AnalyticsReport, Chapter, JobResult, ReferenceHit

PALETTE = {
    "accent": "#C45C26",
    "ink": "#1A1612",
    "slate": "#2C3E50",
    "sage": "#4A6B5C",
    "gold": "#B08946",
    "paper": "#F7F1E8",
    "esoteric": "#6B3FA0",
    "theological": "#8B4513",
    "scientific": "#1F6F8B",
}

LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(247,241,232,0.35)",
    font=dict(family="Source Sans 3, Segoe UI, sans-serif", color=PALETTE["ink"]),
    margin=dict(l=40, r=20, t=50, b=40),
)


def quality_chart(report: AnalyticsReport) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            y=report.quality_by_page,
            x=list(range(1, len(report.quality_by_page) + 1)),
            mode="lines+markers",
            line=dict(color=PALETTE["accent"], width=2.5),
            marker=dict(size=7, color=PALETTE["slate"]),
            fill="tozeroy",
            fillcolor="rgba(196,92,38,0.12)",
            name="Quality",
        )
    )
    fig.update_layout(
        title="Scan Quality by Page",
        xaxis_title="Page",
        yaxis_title="Quality score",
        yaxis=dict(range=[0, 100]),
        **LAYOUT,
    )
    return fig


def reference_sunburst(references: list[ReferenceHit]) -> go.Figure:
    if not references:
        fig = go.Figure()
        fig.update_layout(title="No references found", **LAYOUT)
        return fig
    labels, parents, values, colors = [], [], [], []
    kind_counts: dict[str, int] = {}
    cat_counts: dict[tuple[str, str], int] = {}
    for r in references:
        kind_counts[r.kind.value] = kind_counts.get(r.kind.value, 0) + 1
        cat_counts[(r.kind.value, r.category)] = cat_counts.get((r.kind.value, r.category), 0) + 1

    labels.append("References")
    parents.append("")
    values.append(len(references))
    colors.append(PALETTE["ink"])

    color_map = {
        "esoteric": PALETTE["esoteric"],
        "theological": PALETTE["theological"],
        "scientific": PALETTE["scientific"],
    }
    for kind, count in kind_counts.items():
        labels.append(kind.title())
        parents.append("References")
        values.append(count)
        colors.append(color_map.get(kind, PALETTE["gold"]))
    for (kind, cat), count in cat_counts.items():
        labels.append(cat)
        parents.append(kind.title())
        values.append(count)
        colors.append(color_map.get(kind, PALETTE["gold"]))

    fig = go.Figure(
        go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors, line=dict(color=PALETTE["paper"], width=1)),
            branchvalues="total",
            hovertemplate="<b>%{label}</b><br>%{value} hits<extra></extra>",
        )
    )
    fig.update_layout(title="Reference Taxonomy", **LAYOUT)
    return fig


def top_terms_chart(report: AnalyticsReport) -> go.Figure:
    terms = report.top_terms[:15][::-1]
    if not terms:
        fig = go.Figure()
        fig.update_layout(title="Top terms unavailable", **LAYOUT)
        return fig
    fig = go.Figure(
        go.Bar(
            x=[c for _, c in terms],
            y=[t for t, _ in terms],
            orientation="h",
            marker=dict(color=PALETTE["sage"]),
        )
    )
    fig.update_layout(title="Top Content Terms", xaxis_title="Frequency", **LAYOUT)
    return fig


def chapter_word_chart(chapters: list[Chapter]) -> go.Figure:
    if not chapters:
        fig = go.Figure()
        fig.update_layout(title="No chapters detected", **LAYOUT)
        return fig
    fig = go.Figure(
        go.Bar(
            x=[c.title[:32] for c in chapters],
            y=[c.word_count for c in chapters],
            marker=dict(color=PALETTE["slate"]),
        )
    )
    fig.update_layout(title="Words per Chapter", **LAYOUT)
    fig.update_xaxes(tickangle=-25)
    return fig


def thematic_radar(report: AnalyticsReport) -> go.Figure:
    weights = report.thematic_weights or {"esoteric": 0, "theological": 0, "scientific": 0}
    cats = ["esoteric", "theological", "scientific"]
    vals = [weights.get(c, 0) for c in cats]
    fig = go.Figure(
        go.Scatterpolar(
            r=vals + [vals[0]],
            theta=[c.title() for c in cats] + [cats[0].title()],
            fill="toself",
            fillcolor="rgba(196,92,38,0.25)",
            line=dict(color=PALETTE["accent"]),
        )
    )
    fig.update_layout(
        title="Thematic Balance",
        polar=dict(radialaxis=dict(visible=True, range=[0, max(vals + [0.1])])),
        **LAYOUT,
    )
    return fig


def language_pie(report: AnalyticsReport) -> go.Figure:
    if not report.language_distribution:
        fig = go.Figure()
        fig.update_layout(title="Language unknown", **LAYOUT)
        return fig
    fig = px.pie(
        names=list(report.language_distribution.keys()),
        values=list(report.language_distribution.values()),
        color_discrete_sequence=[PALETTE["accent"], PALETTE["sage"], PALETTE["slate"], PALETTE["gold"]],
    )
    fig.update_layout(title="Detected Languages", **LAYOUT)
    return fig


def dashboard_figures(result: JobResult) -> dict[str, go.Figure]:
    return {
        "quality": quality_chart(result.analytics),
        "references": reference_sunburst(result.references),
        "terms": top_terms_chart(result.analytics),
        "chapters": chapter_word_chart(result.chapters),
        "radar": thematic_radar(result.analytics),
        "languages": language_pie(result.analytics),
    }


def overview_strip(result: JobResult) -> go.Figure:
    a = result.analytics
    metrics = [
        ("Readability", a.readability_score, 100, PALETTE["accent"]),
        ("Diversity", a.lexical_diversity * 100, 100, PALETTE["sage"]),
        ("Sentiment", (a.sentiment_proxy + 1) * 50, 100, PALETTE["slate"]),
        ("References", float(sum(a.reference_counts.values())), max(sum(a.reference_counts.values()), 10), PALETTE["gold"]),
    ]
    fig = make_subplots(
        rows=1,
        cols=4,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=[m[0] for m in metrics],
    )
    for i, (_label, val, mx, color) in enumerate(metrics, start=1):
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=val,
                gauge=dict(
                    axis=dict(range=[0, mx]),
                    bar=dict(color=color),
                    bgcolor="rgba(255,255,255,0.4)",
                ),
            ),
            row=1,
            col=i,
        )
    fig.update_layout(height=240, **LAYOUT)
    return fig
