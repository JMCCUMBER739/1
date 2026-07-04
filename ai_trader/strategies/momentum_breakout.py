"""Momentum breakout for Bitcoin on 1-hour candles — long AND short
("switching" with the momentum, as requested).

Core ideas, imported from the most proven breakout literature:

  * Donchian 55/20 channel breakout — the Turtle Traders System 2
    (Dennis & Eckhardt): enter on a close through the 55-bar extreme,
    exit on the opposite 20-bar extreme.
  * ADX trend-strength gate (Wilder): only take breakouts when ADX >= 20
    so we skip false breaks inside dead ranges.
  * Volume confirmation: breakout bar volume must exceed 1.3x its 20-bar
    average — genuine crypto breakouts come with participation.

The strategy flips: a short signal while long is treated by the engine
as "exit long, enter short" (and vice-versa), so it always rides the
active momentum direction. Hard 1% stop applies on top of channel exits.
"""

from __future__ import annotations

import pandas as pd

from ai_trader.indicators import adx, donchian
from ai_trader.strategies.base import Strategy


class MomentumBreakoutStrategy(Strategy):
    name = "momentum_breakout"

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        out = df.copy()

        entry_ch = donchian(out, p["donchian_period"])
        exit_ch = donchian(out, p["exit_period"])
        out["don_upper"], out["don_lower"] = entry_ch["upper"], entry_ch["lower"]
        out["don_exit_upper"], out["don_exit_lower"] = (
            exit_ch["upper"],
            exit_ch["lower"],
        )
        out["adx"] = adx(out, p["adx_period"])

        vol_avg = out["volume"].rolling(20).mean()
        vol_ok = out["volume"] >= p["volume_mult"] * vol_avg
        trending = out["adx"] >= p["adx_min"]

        out["long_entry"] = (out["close"] > out["don_upper"]) & trending & vol_ok
        out["long_exit"] = out["close"] < out["don_exit_lower"]

        if p.get("allow_short", True):
            out["short_entry"] = (
                (out["close"] < out["don_lower"]) & trending & vol_ok
            )
            out["short_exit"] = out["close"] > out["don_exit_upper"]
        else:
            out["short_entry"] = False
            out["short_exit"] = False

        out["time_stop_bars"] = 0
        out["signal_reason"] = (
            "55-bar Donchian breakout, ADX>=20, volume-confirmed (Turtle S2)"
        )
        out["signal_confidence"] = (out["adx"] / 50).clip(0.3, 0.9)
        return out
