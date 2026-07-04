"""Configuration loading and validation for AI Trader."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class InstrumentConfig:
    key: str
    symbol: str
    merrill_symbol: str
    strategy: str
    timeframe: str
    risk_on: bool


@dataclass
class RiskConfig:
    hard_stop_pct: float = 1.0
    risk_per_trade_pct: float = 0.5
    vol_target_annual_pct: float = 12.0
    max_open_positions: int = 4
    max_gross_exposure_pct: float = 150.0
    block_double_risk_on: bool = True
    daily_loss_limit_pct: float = 2.0


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)
    equity: float = 100_000.0
    instruments: dict[str, InstrumentConfig] = field(default_factory=dict)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategies: dict[str, dict[str, Any]] = field(default_factory=dict)
    orders: dict[str, Any] = field(default_factory=dict)
    broker: dict[str, Any] = field(default_factory=dict)
    ai: dict[str, Any] = field(default_factory=dict)
    reports: dict[str, Any] = field(default_factory=dict)

    @property
    def anthropic_api_key(self) -> str | None:
        return os.environ.get("ANTHROPIC_API_KEY")

    @property
    def smtp_password(self) -> str | None:
        return os.environ.get("SMTP_PASSWORD")


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate config.yaml into a typed Config object."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.is_absolute() and not path.exists():
        # Relative path not found from the current working directory
        # (common when running inside an IDE like Spyder) — fall back
        # to resolving it against the repository root.
        candidate = DEFAULT_CONFIG_PATH.parent / path
        if candidate.exists():
            path = candidate
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    instruments = {
        key: InstrumentConfig(key=key, **spec)
        for key, spec in raw.get("instruments", {}).items()
    }
    risk = RiskConfig(**raw.get("risk", {}))

    if risk.hard_stop_pct <= 0 or risk.hard_stop_pct > 5:
        raise ValueError("risk.hard_stop_pct must be in (0, 5]")
    if risk.risk_per_trade_pct > risk.hard_stop_pct:
        # Risking more per trade than the stop allows would break the 1% rule.
        raise ValueError("risk.risk_per_trade_pct must be <= hard_stop_pct")

    return Config(
        raw=raw,
        equity=float(raw.get("account", {}).get("equity", 100_000)),
        instruments=instruments,
        risk=risk,
        strategies=raw.get("strategies", {}),
        orders=raw.get("orders", {}),
        broker=raw.get("broker", {}),
        ai=raw.get("ai", {}),
        reports=raw.get("reports", {}),
    )
