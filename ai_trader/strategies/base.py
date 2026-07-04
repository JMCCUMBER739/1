"""Strategy base class and the Signal contract.

Every strategy consumes an OHLCV DataFrame and produces boolean
entry/exit columns plus its indicator columns (kept for plotting and
for the Claude logic-review payload). All rolling windows are applied
so that a signal on bar *t* only uses information up to and including
bar *t* (entries are filled on bar *t+1*'s open by the backtester —
no lookahead bias).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class Signal:
    """A live, actionable signal with recommended order levels."""

    instrument: str
    side: str                # "long" | "short" | "flat"
    reason: str
    entry: float             # recommended entry price
    stop: float              # recommended stop (never wider than the hard 1%)
    target: float            # recommended profit target
    atr: float               # current ATR (for sizing)
    confidence: float = 0.5  # strategy's own confidence in [0, 1]

    def as_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "side": self.side,
            "reason": self.reason,
            "entry": round(self.entry, 4),
            "stop": round(self.stop, 4),
            "target": round(self.target, 4),
            "atr": round(self.atr, 6),
            "confidence": self.confidence,
        }


class Strategy(ABC):
    """Base strategy. Subclasses implement `annotate`."""

    name: str = "base"

    def __init__(self, params: dict[str, Any]):
        self.params = params

    @abstractmethod
    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with added indicator columns and boolean columns:
        long_entry, long_exit, short_entry, short_exit."""

    def latest_signal(
        self,
        df: pd.DataFrame,
        instrument: str,
        hard_stop_pct: float,
        target_r: float = 2.0,
    ) -> Signal | None:
        """Turn the most recent bar into an actionable Signal, or None."""
        from ai_trader.indicators import atr as atr_fn

        ann = self.annotate(df)
        last = ann.iloc[-1]
        price = float(last["close"])
        cur_atr = float(atr_fn(df, 14).iloc[-1])

        side = None
        if bool(last.get("long_entry")):
            side = "long"
        elif bool(last.get("short_entry")):
            side = "short"
        if side is None:
            return None

        # Recommended stop: 1.5*ATR but NEVER wider than the hard 1% cap.
        stop_dist = min(1.5 * cur_atr, price * hard_stop_pct / 100.0)
        if side == "long":
            stop, target = price - stop_dist, price + target_r * stop_dist
        else:
            stop, target = price + stop_dist, price - target_r * stop_dist

        return Signal(
            instrument=instrument,
            side=side,
            reason=str(last.get("signal_reason", self.name)),
            entry=price,
            stop=stop,
            target=target,
            atr=cur_atr,
            confidence=float(last.get("signal_confidence", 0.5)),
        )
