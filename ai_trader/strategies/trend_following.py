"""Trend following for Gold and Crude Oil on 4-hour candles —
"ride clean waves, filter intraday noise".

Imported components:

  * EMA 20/50 crossover with price confirmation — the classic
    medium-term trend definition on commodities.
  * ADX >= 22 gate (Wilder) — only trade when a real trend exists.
  * Kaufman Efficiency Ratio noise filter (Kaufman, "Smarter Trading"):
    ER = |net move| / path length over 20 bars. Requiring ER >= 0.30
    means price is travelling in a straight-ish line — a *clean wave* —
    and rejects choppy intraday noise even if the EMAs are crossed.
  * Chandelier trailing exit (LeBeau): exit long when price closes
    3*ATR below the highest close since entry — lets winners run while
    intraday wiggles smaller than 3 ATRs are ignored.

The hard 1% stop still caps the worst case on every position.
"""

from __future__ import annotations

import pandas as pd

from ai_trader.indicators import adx, atr, ema, kaufman_efficiency_ratio
from ai_trader.strategies.base import Strategy


class TrendFollowingStrategy(Strategy):
    name = "trend_following"

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        out = df.copy()

        out["ema_fast"] = ema(out["close"], p["fast_ema"])
        out["ema_slow"] = ema(out["close"], p["slow_ema"])
        out["adx"] = adx(out, p["adx_period"])
        out["er"] = kaufman_efficiency_ratio(
            out["close"], p["efficiency_ratio_period"]
        )
        out["atr"] = atr(out, 14)

        cross_up = (out["ema_fast"] > out["ema_slow"]) & (
            out["ema_fast"].shift(1) <= out["ema_slow"].shift(1)
        )
        cross_dn = (out["ema_fast"] < out["ema_slow"]) & (
            out["ema_fast"].shift(1) >= out["ema_slow"].shift(1)
        )
        bull = out["ema_fast"] > out["ema_slow"]
        bear = out["ema_fast"] < out["ema_slow"]

        trending = out["adx"] >= p["adx_min"]
        clean_wave = out["er"] >= p["efficiency_ratio_min"]  # the noise filter

        # Enter on the cross itself, or on the first bar where a live
        # trend passes the ADX + ER quality gates.
        gates_open = trending & clean_wave
        newly_clean = gates_open & ~gates_open.shift(1, fill_value=False)
        out["long_entry"] = (cross_up & gates_open) | (newly_clean & bull)
        out["short_entry"] = (cross_dn & gates_open) | (newly_clean & bear)

        # Base exits: opposite cross, or the wave gets dirty (ER collapse).
        wave_dead = out["er"] < p["efficiency_ratio_min"] * 0.5
        out["long_exit"] = cross_dn | (bull & wave_dead)
        out["short_exit"] = cross_up | (bear & wave_dead)

        out["trail_atr_mult"] = p.get("trail_atr_mult", 3.0)  # chandelier trail
        out["time_stop_bars"] = 0
        out["signal_reason"] = (
            "EMA20/50 trend + ADX>=22 + Kaufman ER>=0.30 clean-wave filter"
        )
        out["signal_confidence"] = (
            0.4 * (out["adx"] / 50).clip(0, 1) + 0.6 * out["er"]
        ).clip(0.3, 0.9)
        return out
