"""Risk measurement & feedback — the numbers behind the evening report.

Computes, from an equity curve and a trade list:
Sharpe, Sortino, CAGR, max drawdown, Calmar, 95% VaR, win rate,
profit factor, expectancy (R), average win/loss, and exposure stats.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _annualisation_factor(index: pd.DatetimeIndex) -> float:
    """Estimate periods-per-year from the median bar spacing."""
    if len(index) < 3:
        return 252.0
    spacing = np.median(np.diff(index.values).astype("timedelta64[s]").astype(float))
    seconds_per_year = 365.25 * 24 * 3600
    return max(1.0, seconds_per_year / max(spacing, 1.0))


def performance_report(
    equity_curve: pd.Series, trades: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Full risk/performance dictionary from an equity curve (indexed by time)."""
    eq = equity_curve.dropna()
    if len(eq) < 3:
        return {"error": "not enough data"}

    rets = eq.pct_change().dropna()
    ann = _annualisation_factor(eq.index)

    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    years = max((eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 24 * 3600),
                1e-9)
    cagr = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1.0

    vol = rets.std(ddof=0) * math.sqrt(ann)
    sharpe = (rets.mean() * ann) / vol if vol > 0 else 0.0
    downside = rets[rets < 0].std(ddof=0) * math.sqrt(ann)
    sortino = (rets.mean() * ann) / downside if downside > 0 else 0.0

    peak = eq.cummax()
    dd = eq / peak - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("inf")
    var_95 = np.percentile(rets, 5)

    report: dict[str, Any] = {
        "total_return_pct": round(100 * total_return, 2),
        "cagr_pct": round(100 * cagr, 2),
        "annual_vol_pct": round(100 * vol, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_drawdown_pct": round(100 * max_dd, 2),
        "calmar": round(calmar, 2) if math.isfinite(calmar) else None,
        "var_95_per_bar_pct": round(100 * var_95, 3),
        "bars": len(eq),
    }

    if trades is not None and len(trades):
        pnl = trades["pnl"]
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        gross_win, gross_loss = wins.sum(), abs(losses.sum())
        r_multiples = trades.get("r_multiple")
        report.update(
            {
                "num_trades": len(trades),
                "win_rate_pct": round(100 * len(wins) / len(trades), 1),
                "profit_factor": round(gross_win / gross_loss, 2)
                if gross_loss > 0
                else None,
                "avg_win": round(wins.mean(), 2) if len(wins) else 0.0,
                "avg_loss": round(losses.mean(), 2) if len(losses) else 0.0,
                "expectancy_r": round(r_multiples.mean(), 3)
                if r_multiples is not None
                else None,
                "best_trade": round(pnl.max(), 2),
                "worst_trade": round(pnl.min(), 2),
            }
        )
    return report


def feedback(report: dict[str, Any]) -> list[str]:
    """Plain-English risk feedback derived from the metrics."""
    notes: list[str] = []
    if report.get("error"):
        return ["Not enough data for risk feedback yet."]

    sharpe = report.get("sharpe", 0)
    if sharpe >= 1.5:
        notes.append(f"Sharpe {sharpe}: strong risk-adjusted returns.")
    elif sharpe >= 0.7:
        notes.append(f"Sharpe {sharpe}: acceptable, keep sizing where it is.")
    else:
        notes.append(f"Sharpe {sharpe}: weak edge right now — consider reducing "
                     "risk_per_trade_pct until performance recovers.")

    dd = report.get("max_drawdown_pct", 0)
    if dd < -10:
        notes.append(f"Max drawdown {dd}% is heavy; volatility targeting will "
                     "already be shrinking size — do not override it upward.")
    else:
        notes.append(f"Max drawdown {dd}% is within tolerance.")

    pf = report.get("profit_factor")
    wr = report.get("win_rate_pct")
    if pf is not None and wr is not None:
        notes.append(f"Win rate {wr}% with profit factor {pf}. "
                     + ("Edge is positive — losses are controlled by the 1% hard stop."
                        if pf and pf > 1 else
                        "Profit factor below 1 — the system is currently paying "
                        "for its stops; review the regime filters."))
    return notes
