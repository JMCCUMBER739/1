"""Technical indicator library (vectorised, pandas/numpy only).

Implements the building blocks used by the three strategy engines:
Bollinger Bands, Connors RSI-2 style RSI, rolling z-score (mean reversion);
Donchian channels, ADX (momentum breakout); EMA cross, Kaufman Efficiency
Ratio, Wilder ATR (trend following, sizing, stops).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI. Use period=2 for the Connors mean-reversion variant."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    mid = sma(series, period)
    std = series.rolling(period).std(ddof=0)
    return pd.DataFrame(
        {"mid": mid, "upper": mid + num_std * std, "lower": mid - num_std * std}
    )


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    mid = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    return (series - mid) / std.replace(0, np.nan)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder Average True Range; expects columns high/low/close."""
    prev_close = df["close"].shift()
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Donchian channel of the PRIOR `period` bars (shifted to avoid lookahead)."""
    upper = df["high"].rolling(period).max().shift(1)
    lower = df["low"].rolling(period).min().shift(1)
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": (upper + lower) / 2})


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — trend-strength filter."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    tr = atr(df, period)  # already Wilder-smoothed true range
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


def kaufman_efficiency_ratio(series: pd.Series, period: int = 20) -> pd.Series:
    """Kaufman Efficiency Ratio in [0, 1].

    ER = |net move over N bars| / sum of absolute bar-to-bar moves.
    ~1 means a clean directional wave; ~0 means choppy intraday noise.
    Used as the noise filter for the gold/oil trend follower.
    """
    direction = (series - series.shift(period)).abs()
    volatility = series.diff().abs().rolling(period).sum()
    return (direction / volatility.replace(0, np.nan)).fillna(0.0)
