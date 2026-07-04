"""Mean reversion for S&P 500 and NASDAQ 100 on 15-minute candles.

Blend of the two most battle-tested equity-index mean-reversion ideas:

  * Bollinger Band / z-score stretch (Bollinger, 1980s): price closing
    below the lower band with a z-score <= -2 marks a statistically
    stretched dip.
  * Connors RSI-2 (Connors & Alvarez, "Short Term Trading Strategies
    That Work"): a 2-period RSI under ~10 marks panic selling that, in
    index products, mean-reverts with high win rates.
  * Regime filter: dips are only bought while price holds above the
    200-period EMA — the classic "buy pullbacks in an uptrend" rule that
    keeps a reversion system out of 2008/2020/2022-style waterfalls.

Exits: z-score back to 0 (band mid), RSI2 > 70, or a time stop
(~1 trading day of 15m bars). The engine's hard 1% stop applies on top.
"""

from __future__ import annotations

import pandas as pd

from ai_trader.indicators import bollinger_bands, ema, rsi, zscore
from ai_trader.strategies.base import Strategy


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        out = df.copy()

        bb = bollinger_bands(out["close"], p["bb_period"], p["bb_std"])
        out["bb_lower"], out["bb_mid"], out["bb_upper"] = (
            bb["lower"],
            bb["mid"],
            bb["upper"],
        )
        out["rsi_fast"] = rsi(out["close"], p["rsi_period"])
        out["z"] = zscore(out["close"], p["bb_period"])
        out["ema_trend"] = ema(out["close"], p["trend_filter_ema"])

        uptrend = out["close"] > out["ema_trend"]
        stretched = (out["z"] <= p["zscore_entry"]) | (
            out["close"] < out["bb_lower"]
        )
        panic = out["rsi_fast"] <= p["rsi_oversold"]

        out["long_entry"] = uptrend & stretched & panic
        out["long_exit"] = (out["z"] >= p["zscore_exit"]) | (out["rsi_fast"] >= 70)

        # Symmetric short side for rips in a downtrend (kept conservative).
        downtrend = out["close"] < out["ema_trend"]
        stretched_up = (out["z"] >= -p["zscore_entry"]) | (
            out["close"] > out["bb_upper"]
        )
        euphoric = out["rsi_fast"] >= p["rsi_overbought"]
        out["short_entry"] = downtrend & stretched_up & euphoric
        out["short_exit"] = (out["z"] <= p["zscore_exit"]) | (out["rsi_fast"] <= 30)

        out["time_stop_bars"] = p.get("time_stop_bars", 0)
        out["signal_reason"] = (
            "RSI2 panic + z<=-2 stretch above 200EMA (buy the dip)"
        )
        # Deeper stretch -> higher confidence, capped at 0.9.
        out["signal_confidence"] = (out["z"].abs() / 4).clip(0.3, 0.9)
        return out
