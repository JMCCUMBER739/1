"""Event-driven backtester.

Realism rules:
  * Signals are computed on bar t; entries fill at bar t+1's OPEN.
  * The hard 1% stop and the profit target are checked INTRABAR against
    the bar's high/low (stop checked first — pessimistic).
  * Trend positions use a chandelier trail (N*ATR off the best close).
  * Position size is volatility-adjusted per the risk engine's rules.
  * Per-side commission+slippage haircut is applied to every fill.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ai_trader.indicators import atr as atr_fn
from ai_trader.risk.engine import position_size
from ai_trader.risk.metrics import performance_report
from ai_trader.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    instrument: str
    equity_curve: pd.Series = field(default_factory=pd.Series)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    report: dict = field(default_factory=dict)
    annotated: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_backtest(
    strategy: Strategy,
    df: pd.DataFrame,
    instrument: str,
    timeframe: str,
    initial_equity: float = 100_000.0,
    hard_stop_pct: float = 1.0,
    risk_per_trade_pct: float = 0.5,
    vol_target_annual_pct: float = 12.0,
    target_r: float = 2.0,
    cost_pct_per_side: float = 0.03,
) -> BacktestResult:
    ann = strategy.annotate(df)
    ann["atr14"] = atr_fn(df, 14)

    o = ann["open"].to_numpy(float)
    h = ann["high"].to_numpy(float)
    l = ann["low"].to_numpy(float)
    c = ann["close"].to_numpy(float)
    atr_v = ann["atr14"].to_numpy(float)
    long_e = ann["long_entry"].fillna(False).to_numpy(bool)
    long_x = ann["long_exit"].fillna(False).to_numpy(bool)
    short_e = ann["short_entry"].fillna(False).to_numpy(bool)
    short_x = ann["short_exit"].fillna(False).to_numpy(bool)
    time_stop = int(ann["time_stop_bars"].iloc[-1]) if "time_stop_bars" in ann else 0
    trail_mult = (
        float(ann["trail_atr_mult"].iloc[-1]) if "trail_atr_mult" in ann else 0.0
    )
    cost = cost_pct_per_side / 100.0

    equity = initial_equity
    eq_curve = np.full(len(ann), np.nan)
    trades: list[dict] = []

    pos = 0            # 0 flat, +1 long, -1 short
    qty = entry = stop = target = 0.0
    best_close = 0.0
    entry_i = 0
    pending = 0        # signal generated on prev bar; fill at this bar's open

    for i in range(1, len(ann)):
        # ---------------- fill pending entry at this bar's open ----------
        if pending != 0 and pos == 0:
            px = o[i] * (1 + cost * pending)          # slippage against us
            stop_dist = min(1.5 * atr_v[i - 1], px * hard_stop_pct / 100.0)
            if stop_dist > 0 and not np.isnan(px):
                q = position_size(equity, px, px - pending * stop_dist,
                                  atr_v[i - 1], timeframe,
                                  risk_per_trade_pct, vol_target_annual_pct)
                if q > 0:
                    pos, qty, entry = pending, q, px
                    stop = px - pending * stop_dist
                    target = px + pending * target_r * stop_dist
                    best_close = c[i]
                    entry_i = i
            pending = 0

        exit_px, exit_reason = None, None

        if pos != 0:
            # ------------- intrabar hard stop (checked first) -------------
            if pos == 1 and l[i] <= stop:
                exit_px, exit_reason = stop, "hard_stop"
            elif pos == -1 and h[i] >= stop:
                exit_px, exit_reason = stop, "hard_stop"
            # ------------- intrabar profit target -------------------------
            elif pos == 1 and h[i] >= target:
                exit_px, exit_reason = target, "target"
            elif pos == -1 and l[i] <= target:
                exit_px, exit_reason = target, "target"
            else:
                # ------------- chandelier trail (trend strategies) --------
                if trail_mult > 0:
                    best_close = max(best_close, c[i]) if pos == 1 else min(
                        best_close, c[i])
                    trail = (best_close - pos * trail_mult * atr_v[i])
                    if (pos == 1 and c[i] < trail) or (pos == -1 and c[i] > trail):
                        exit_px, exit_reason = c[i], "atr_trail"
                # ------------- strategy exit / flip ------------------------
                if exit_px is None:
                    if pos == 1 and (long_x[i] or short_e[i]):
                        exit_px, exit_reason = c[i], (
                            "flip" if short_e[i] else "strategy_exit")
                    elif pos == -1 and (short_x[i] or long_e[i]):
                        exit_px, exit_reason = c[i], (
                            "flip" if long_e[i] else "strategy_exit")
                # ------------- time stop -----------------------------------
                if exit_px is None and time_stop and i - entry_i >= time_stop:
                    exit_px, exit_reason = c[i], "time_stop"

            if exit_px is not None:
                fill = exit_px * (1 - cost * pos)
                pnl = (fill - entry) * qty * pos
                risk_amt = abs(entry - stop) * qty
                equity += pnl
                trades.append({
                    "instrument": instrument,
                    "side": "long" if pos == 1 else "short",
                    "entry_time": ann.index[entry_i],
                    "exit_time": ann.index[i],
                    "entry": entry, "exit": fill, "qty": qty,
                    "pnl": pnl,
                    "r_multiple": pnl / risk_amt if risk_amt > 0 else 0.0,
                    "reason": exit_reason,
                })
                # flip straight into the opposite side if signalled
                if exit_reason == "flip":
                    pending = -pos
                pos, qty = 0, 0.0

        # ---------------- queue new entries -------------------------------
        if pos == 0 and pending == 0:
            if long_e[i]:
                pending = 1
            elif short_e[i]:
                pending = -1

        # mark-to-market equity
        eq_curve[i] = equity + (c[i] - entry) * qty * pos if pos != 0 else equity

    eq = pd.Series(eq_curve, index=ann.index).ffill().fillna(initial_equity)
    trades_df = pd.DataFrame(trades)
    report = performance_report(eq, trades_df if len(trades_df) else None)
    return BacktestResult(instrument, eq, trades_df, report, ann)
