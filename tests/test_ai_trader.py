"""Tests for the AI Trader risk engine, indicators, and backtester."""

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from ai_trader.config import Config, InstrumentConfig, RiskConfig
from ai_trader.indicators import atr, donchian, kaufman_efficiency_ratio, rsi
from ai_trader.risk.engine import OpenPosition, RiskEngine, position_size
from ai_trader.strategies.base import Signal


def _config() -> Config:
    cfg = Config()
    cfg.risk = RiskConfig()
    for key, risk_on in [("sp500", True), ("nasdaq", True),
                         ("bitcoin", True), ("gold", False)]:
        cfg.instruments[key] = InstrumentConfig(
            key=key, symbol=key.upper(), merrill_symbol=key.upper(),
            strategy="x", timeframe="1h", risk_on=risk_on)
    return cfg


def _signal(instrument: str, side: str = "long", entry: float = 100.0) -> Signal:
    stop = entry * (0.99 if side == "long" else 1.01)
    tgt = entry * (1.02 if side == "long" else 0.98)
    return Signal(instrument, side, "test", entry, stop, tgt, atr=0.5)


# --------------------------------------------------------------- hard stop
def test_hard_stop_clamped_to_one_percent():
    eng = RiskEngine(_config())
    sig = _signal("gold", entry=100.0)
    sig.stop = 95.0  # 5% away — must be clamped
    decision = eng.evaluate(sig, equity=100_000)
    assert decision.approved
    assert decision.stop == pytest.approx(99.0)  # exactly 1% below entry


def test_risk_per_trade_never_exceeds_budget():
    qty = position_size(
        equity=100_000, entry=100.0, stop=99.0, atr_value=0.5,
        timeframe="1h", risk_per_trade_pct=0.5, vol_target_annual_pct=999,
    )
    worst_loss = qty * (100.0 - 99.0)
    assert worst_loss <= 100_000 * 0.005 + 1e-6


def test_volatility_sizing_shrinks_in_wild_markets():
    calm = position_size(100_000, 100, 99, atr_value=0.2, timeframe="1h",
                         risk_per_trade_pct=0.5, vol_target_annual_pct=12)
    wild = position_size(100_000, 100, 99, atr_value=5.0, timeframe="1h",
                         risk_per_trade_pct=0.5, vol_target_annual_pct=12)
    assert wild < calm


# ------------------------------------------------------- double risk-on guard
def test_double_risk_on_guard_blocks_new_longs():
    eng = RiskEngine(_config())
    for key in ("sp500", "nasdaq"):
        eng.register_fill(OpenPosition(key, "long", 10, 100, 99, 102, True))

    blocked = eng.evaluate(_signal("bitcoin", "long"), 100_000)
    assert not blocked.approved
    assert any("risk-on" in r for r in blocked.reasons)

    # risk-off instrument (gold) is still allowed
    allowed = eng.evaluate(_signal("gold", "long"), 100_000)
    assert allowed.approved

    # shorts are not risk-on and are still allowed
    short = eng.evaluate(_signal("bitcoin", "short"), 100_000)
    assert short.approved


def test_guard_inactive_when_only_one_index_long():
    eng = RiskEngine(_config())
    eng.register_fill(OpenPosition("sp500", "long", 10, 100, 99, 102, True))
    assert eng.evaluate(_signal("bitcoin", "long"), 100_000).approved


def test_daily_loss_circuit_breaker():
    eng = RiskEngine(_config())
    eng.day_pnl = -3000  # beyond the 2% of 100k limit
    decision = eng.evaluate(_signal("gold"), 100_000)
    assert not decision.approved


# ------------------------------------------------------------- indicators
def _ohlcv(n=300, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": close + rng.normal(0, 0.2, n),
        "high": close + abs(rng.normal(0, 0.8, n)),
        "low": close - abs(rng.normal(0, 0.8, n)),
        "close": close,
        "volume": rng.integers(1000, 5000, n).astype(float),
    }, index=idx)


def test_rsi_bounded():
    r = rsi(_ohlcv()["close"], 14)
    assert r.between(0, 100).all()


def test_atr_positive():
    assert (atr(_ohlcv(), 14).dropna() > 0).all()


def test_donchian_no_lookahead():
    df = _ohlcv()
    ch = donchian(df, 20)
    # channel at bar i must ignore bar i itself (shifted by one)
    i = 250
    assert ch["upper"].iloc[i] == df["high"].iloc[i - 20:i].max()


def test_efficiency_ratio_bounds_and_trend_detection():
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    straight = pd.Series(np.linspace(100, 200, n), index=idx)
    er = kaufman_efficiency_ratio(straight, 20).dropna()
    assert er.iloc[-1] == pytest.approx(1.0)  # perfectly clean wave

    rng = np.random.default_rng(0)
    chop = pd.Series(100 + rng.normal(0, 1, n), index=idx)
    assert kaufman_efficiency_ratio(chop, 20).iloc[-1] < 0.5


# ------------------------------------------------------------- backtester
def test_backtest_worst_trade_capped_near_one_percent():
    from ai_trader.backtest.engine import run_backtest
    from ai_trader.strategies import MomentumBreakoutStrategy

    df = _ohlcv(n=1500, seed=3)
    strat = MomentumBreakoutStrategy({
        "donchian_period": 30, "exit_period": 10, "adx_period": 14,
        "adx_min": 5, "volume_mult": 0.5, "allow_short": True,
    })
    res = run_backtest(strat, df, "test", "1h", initial_equity=100_000,
                       hard_stop_pct=1.0, risk_per_trade_pct=0.5)
    assert len(res.trades) > 0
    # every losing trade must be near -1R or better (costs allow slight excess)
    assert res.trades["r_multiple"].min() >= -1.25
