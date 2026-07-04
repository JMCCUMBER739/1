"""Market data acquisition and caching.

Uses Yahoo Finance (free, no key). Important reality about free intraday data:

  * daily bars    : full 10+ year history available
  * 1-hour bars   : ~730 days of history
  * 15-minute bars: ~60 days of history

So the 10-year "perfecting" pass works in two layers:
  1. A 10-year daily backtest of each strategy translated to daily rules,
     to validate the edge across regimes (2016-2026: bull, COVID crash,
     2022 bear, rate-hike chop, recovery).
  2. An intraday backtest on the maximum intraday history available
     (60d of 15m, 730d of 1h/4h) to validate execution-level behaviour.

For deeper intraday history plug a paid feed (Polygon.io, Databento,
Tiingo) into `load_history` — the rest of the engine is source-agnostic.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".data_cache"

# Yahoo limits for free intraday data.
_MAX_PERIOD = {"15m": "60d", "1h": "730d", "1d": "10y"}
_YF_INTERVAL = {"15m": "15m", "1h": "1h", "4h": "1h", "1d": "1d"}


def _cache_path(symbol: str, timeframe: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    safe = symbol.replace("^", "_").replace("=", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}_{timeframe}.parquet"


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV bars to a coarser timeframe (e.g. 1h -> 4h)."""
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df.resample(rule, origin="start_day").agg(agg).dropna(subset=["close"])
    return out


def load_history(
    symbol: str,
    timeframe: str,
    period: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return an OHLCV DataFrame (UTC index, lowercase columns).

    4h bars are built by resampling 1h bars since Yahoo has no native 4h.
    """
    cache = _cache_path(symbol, timeframe)
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
        logger.info("loaded %s %s from cache (%d bars)", symbol, timeframe, len(df))
        return df

    import yfinance as yf

    fetch_tf = _YF_INTERVAL[timeframe]
    fetch_period = period or _MAX_PERIOD["1h" if timeframe == "4h" else timeframe]

    raw = yf.download(
        symbol,
        period=fetch_period,
        interval=fetch_tf,
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"no data returned for {symbol} {timeframe}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if timeframe == "4h":
        df = resample_ohlcv(df, "4h")

    if use_cache:
        df.to_parquet(cache)
    logger.info("downloaded %s %s (%d bars)", symbol, timeframe, len(df))
    return df
