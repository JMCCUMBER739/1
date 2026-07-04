"""Visualization suite for backtest results.

Produces, per instrument: price chart with indicators and trade markers,
equity curve with drawdown, R-multiple distribution — plus a portfolio
dashboard combining all instruments.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai_trader.backtest.engine import BacktestResult

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.facecolor": "#0f1419",
    "axes.facecolor": "#0f1419",
    "axes.edgecolor": "#3a4149",
    "axes.labelcolor": "#d5dbe0",
    "text.color": "#d5dbe0",
    "xtick.color": "#8a939c",
    "ytick.color": "#8a939c",
    "grid.color": "#232a31",
    "axes.grid": True,
    "grid.linestyle": ":",
    "font.size": 9,
})

GREEN, RED, BLUE, GOLD, GREY = "#2fbf71", "#ef476f", "#4cc9f0", "#f4a259", "#8a939c"


def plot_instrument(result: BacktestResult, out_dir: str | Path) -> Path:
    """Price + trades, equity curve, drawdown, and R distribution."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ann, eq, trades = result.annotated, result.equity_curve, result.trades

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 13), sharex=False,
        gridspec_kw={"height_ratios": [3, 1.4, 1, 1.2]},
    )
    fig.suptitle(f"{result.instrument.upper()} — backtest", fontsize=14)

    # --- price with indicators and trades -----------------------------
    ax = axes[0]
    ax.plot(ann.index, ann["close"], color=BLUE, lw=0.8, label="close")
    for col, color, label in [
        ("bb_upper", GREY, "BB upper"), ("bb_lower", GREY, "BB lower"),
        ("ema_trend", GOLD, "trend EMA"), ("don_upper", GREY, "Donchian hi"),
        ("don_lower", GREY, "Donchian lo"), ("ema_fast", GOLD, "EMA fast"),
        ("ema_slow", RED, "EMA slow"),
    ]:
        if col in ann.columns:
            ax.plot(ann.index, ann[col], color=color, lw=0.6, alpha=0.7,
                    label=label)
    if len(trades):
        longs = trades[trades["side"] == "long"]
        shorts = trades[trades["side"] == "short"]
        ax.scatter(longs["entry_time"], longs["entry"], marker="^", s=42,
                   color=GREEN, zorder=5, label="long entry")
        ax.scatter(shorts["entry_time"], shorts["entry"], marker="v", s=42,
                   color=RED, zorder=5, label="short entry")
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]
        ax.scatter(wins["exit_time"], wins["exit"], marker="o", s=26,
                   color=GREEN, alpha=0.8, zorder=5, label="exit (win)")
        ax.scatter(losses["exit_time"], losses["exit"], marker="x", s=30,
                   color=RED, zorder=5, label="exit (loss)")
    ax.legend(loc="upper left", ncol=4, fontsize=7, framealpha=0.2)
    ax.set_ylabel("price")

    # --- equity curve ---------------------------------------------------
    ax = axes[1]
    ax.plot(eq.index, eq.values, color=GREEN, lw=1.1)
    ax.set_ylabel("equity ($)")
    ax.axhline(eq.iloc[0], color=GREY, lw=0.6, ls="--")

    # --- drawdown ---------------------------------------------------------
    ax = axes[2]
    dd = 100 * (eq / eq.cummax() - 1)
    ax.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.55)
    ax.set_ylabel("drawdown %")

    # --- R-multiple distribution ------------------------------------------
    ax = axes[3]
    if len(trades):
        r = trades["r_multiple"].clip(-3, 6)
        ax.hist(r, bins=40, color=BLUE, alpha=0.85)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.axvline(r.mean(), color=GOLD, lw=1.2,
                   label=f"expectancy {r.mean():.2f}R")
        ax.legend(fontsize=8)
    ax.set_ylabel("trades")
    ax.set_xlabel("R multiple (profit / initial risk)")

    stats = result.report
    txt = "  ".join(f"{k}={v}" for k, v in stats.items()
                    if k in ("total_return_pct", "sharpe", "max_drawdown_pct",
                             "win_rate_pct", "profit_factor", "num_trades",
                             "expectancy_r"))
    fig.text(0.5, 0.005, txt, ha="center", fontsize=8, color=GOLD)

    path = out_dir / f"{result.instrument}_backtest.png"
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def plot_portfolio(results: dict[str, BacktestResult],
                   out_dir: str | Path) -> Path:
    """Combined dashboard: normalised equity curves + summary table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2.2, 1]})
    fig.suptitle("AI Trader — portfolio dashboard", fontsize=14)

    palette = [GREEN, BLUE, GOLD, RED, "#b388eb"]
    for i, (name, res) in enumerate(results.items()):
        eq = res.equity_curve
        norm = 100 * eq / eq.iloc[0]
        ax1.plot(eq.index, norm, lw=1.1, color=palette[i % len(palette)],
                 label=f"{name} ({res.report.get('total_return_pct', 0):+.1f}%)")
    ax1.axhline(100, color=GREY, lw=0.6, ls="--")
    ax1.set_ylabel("equity (start = 100)")
    ax1.legend(fontsize=9, framealpha=0.2)

    cols = ["total_return_pct", "sharpe", "sortino", "max_drawdown_pct",
            "win_rate_pct", "profit_factor", "expectancy_r", "num_trades"]
    table_rows, labels = [], []
    for name, res in results.items():
        labels.append(name)
        table_rows.append([res.report.get(c, "—") for c in cols])
    ax2.axis("off")
    table = ax2.table(cellText=table_rows, rowLabels=labels,
                      colLabels=[c.replace("_pct", " %").replace("_", " ")
                                 for c in cols],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#3a4149")
        cell.set_facecolor("#151b21")
        cell.get_text().set_color("#d5dbe0")

    path = out_dir / "portfolio_dashboard.png"
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    logger.info("wrote %s", path)
    return path
