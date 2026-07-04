"""AI Trader — multi-asset, multi-timeframe systematic trading engine.

Strategies:
  * S&P 500 / NASDAQ 100 : mean reversion on 15-minute candles
  * Bitcoin              : momentum breakout on 1-hour candles
  * Gold / Crude Oil     : trend following on 4-hour candles (noise-filtered)

Risk:
  * Hard 1% stop loss on every position
  * ATR volatility-adjusted position sizing
  * Correlation guard: no new risk-on trade while S&P and NASDAQ are both long
  * Full risk measurement & feedback (Sharpe, Sortino, drawdown, VaR, expectancy)

AI:
  * Claude-powered trade logic review and daily morning-strategy /
    evening-analytics email reports.
"""

__version__ = "1.0.0"
