"""Risk engine — every order must pass through here before execution.

Enforces, in order:
  1. Hard 1% stop: stop distance is clamped to <= 1% of entry price.
  2. Volatility-adjusted position sizing (two constraints, take the min):
       a. fixed-fractional: risk_per_trade% of equity / stop distance
       b. volatility targeting: size so the position's expected annualised
          volatility contribution equals vol_target (ATR-proxied)
  3. Double risk-on guard: when S&P AND NASDAQ are already long, NO new
     risk-on trade (S&P, NASDAQ, BTC longs) is allowed.
  4. Portfolio caps: max open positions, max gross exposure, daily loss limit.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from ai_trader.config import Config
from ai_trader.strategies.base import Signal

logger = logging.getLogger(__name__)

# Bars per year by timeframe, for annualising ATR-based volatility.
_ANNUALISATION = {"15m": 26 * 252, "1h": 24 * 365, "4h": 6 * 252, "1d": 252}


def position_size(
    equity: float,
    entry: float,
    stop: float,
    atr_value: float,
    timeframe: str,
    risk_per_trade_pct: float,
    vol_target_annual_pct: float,
) -> float:
    """Return position size in units (shares / coins), volatility-adjusted.

    Takes the *minimum* of fixed-fractional risk sizing and volatility
    targeting, so size automatically shrinks when the market gets wild
    and grows (up to the risk cap) when it is calm.
    """
    stop_dist = abs(entry - stop)
    if stop_dist <= 0 or entry <= 0:
        return 0.0

    # (a) fixed fractional: lose at most risk_per_trade% if the stop hits
    risk_dollars = equity * risk_per_trade_pct / 100.0
    qty_risk = risk_dollars / stop_dist

    # (b) volatility target: annualised ATR% of the position ~= target
    bars_per_year = _ANNUALISATION.get(timeframe, 252)
    ann_vol_pct = (atr_value / entry) * math.sqrt(bars_per_year) * 100
    if ann_vol_pct > 0:
        target_notional = equity * (vol_target_annual_pct / ann_vol_pct)
        qty_vol = target_notional / entry
    else:
        qty_vol = qty_risk

    return max(0.0, min(qty_risk, qty_vol))


@dataclass
class OpenPosition:
    instrument: str
    side: str          # "long" | "short"
    qty: float
    entry: float
    stop: float
    target: float
    risk_on: bool


@dataclass
class RiskDecision:
    approved: bool
    qty: float
    stop: float
    target: float
    reasons: list[str] = field(default_factory=list)


class RiskEngine:
    def __init__(self, config: Config):
        self.cfg = config
        self.positions: dict[str, OpenPosition] = {}
        self.day_pnl: float = 0.0

    # ---------------------------------------------------------------- state
    def register_fill(self, pos: OpenPosition) -> None:
        self.positions[pos.instrument] = pos

    def register_close(self, instrument: str, pnl: float) -> None:
        self.positions.pop(instrument, None)
        self.day_pnl += pnl

    def reset_day(self) -> None:
        self.day_pnl = 0.0

    # ---------------------------------------------------------------- guards
    def _both_indices_long(self) -> bool:
        sp = self.positions.get("sp500")
        nq = self.positions.get("nasdaq")
        return bool(sp and sp.side == "long" and nq and nq.side == "long")

    def evaluate(self, signal: Signal, equity: float) -> RiskDecision:
        """Vet a signal; returns approved qty and (possibly clamped) stop."""
        r = self.cfg.risk
        inst = self.cfg.instruments[signal.instrument]
        reasons: list[str] = []

        # -- 1. hard 1% stop clamp -------------------------------------
        cap = signal.entry * r.hard_stop_pct / 100.0
        stop = signal.stop
        if abs(signal.entry - stop) > cap:
            stop = (
                signal.entry - cap if signal.side == "long" else signal.entry + cap
            )
            reasons.append(f"stop clamped to hard {r.hard_stop_pct}% cap")

        # -- 2. daily loss circuit breaker ------------------------------
        if self.day_pnl <= -equity * r.daily_loss_limit_pct / 100.0:
            return RiskDecision(False, 0, stop, signal.target,
                                ["daily loss limit hit — trading halted for today"])

        # -- 3. double risk-on exposure guard ----------------------------
        is_risk_on_trade = inst.risk_on and signal.side == "long"
        if (
            r.block_double_risk_on
            and is_risk_on_trade
            and self._both_indices_long()
        ):
            return RiskDecision(False, 0, stop, signal.target,
                                ["S&P and NASDAQ both long — new risk-on trade blocked"])

        # -- 4. position / exposure caps ---------------------------------
        if signal.instrument in self.positions:
            return RiskDecision(False, 0, stop, signal.target,
                                ["position already open for this instrument"])
        if len(self.positions) >= r.max_open_positions:
            return RiskDecision(False, 0, stop, signal.target,
                                ["max open positions reached"])

        # -- 5. volatility-adjusted sizing --------------------------------
        qty = position_size(
            equity=equity,
            entry=signal.entry,
            stop=stop,
            atr_value=signal.atr,
            timeframe=inst.timeframe,
            risk_per_trade_pct=r.risk_per_trade_pct,
            vol_target_annual_pct=r.vol_target_annual_pct,
        )

        gross = sum(p.qty * p.entry for p in self.positions.values())
        max_gross = equity * r.max_gross_exposure_pct / 100.0
        if gross + qty * signal.entry > max_gross:
            allowed = max(0.0, max_gross - gross)
            qty = allowed / signal.entry
            reasons.append("size reduced by gross exposure cap")

        if qty <= 0:
            return RiskDecision(False, 0, stop, signal.target,
                                reasons + ["computed size is zero"])

        reasons.append(
            f"sized {qty:.4f} units (risk {r.risk_per_trade_pct}% / "
            f"vol-target {r.vol_target_annual_pct}%)"
        )
        return RiskDecision(True, qty, stop, signal.target, reasons)
