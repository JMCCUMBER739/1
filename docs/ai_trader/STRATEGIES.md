# Strategy reference — theory, parameters, tuning

The three engines implement the most battle-tested public algorithms in
their respective categories. "Sophisticated" in trading rarely means
complicated — it means robust rules with strict regime filters and risk
control. Every parameter below lives in `config.yaml -> strategies`.

---

## 1. Mean reversion — S&P 500 & NASDAQ 100, 15-minute candles

**Thesis.** Equity indices are strongly mean-reverting on intraday
horizons: statistically stretched dips inside an uptrend tend to snap
back to the mean. This is one of the most persistent documented edges
in US index products.

**Imported components**

| Component | Source | Role |
|---|---|---|
| Bollinger Bands / rolling z-score | J. Bollinger | quantifies the stretch: entry needs z ≤ −2 or a close below the lower band |
| RSI-2 | Connors & Alvarez, *Short Term Trading Strategies That Work* | panic trigger: 2-period RSI ≤ 10 marks capitulation |
| 200-EMA regime filter | classic trend filtering | only buy dips **above** the 200-EMA; only fade rips **below** it — keeps the system out of crash regimes |
| Time stop | Connors-style | if reversion hasn't happened in ~1 day of bars, the thesis is wrong; exit |

**Entry (long).** close > EMA200 **and** (z ≤ −2 **or** close < lower
band) **and** RSI2 ≤ 10.
**Exit.** z back to 0 (band mid) or RSI2 ≥ 70 or 26-bar time stop or
the hard 1% stop / 2R target.
Shorts are symmetric below the 200-EMA with RSI2 ≥ 90.

**Tuning.** `rsi_oversold` lower (e.g. 5) = fewer, higher-quality
trades. `zscore_entry` −2.5 = deeper stretch requirement. Widen
`time_stop_bars` on QQQ if exits look premature.

---

## 2. Momentum breakout — Bitcoin, 1-hour candles, long/short switching

**Thesis.** BTC trends hard when it escapes a consolidation range;
momentum persists once a multi-week extreme breaks. The system
"switches" — it holds whichever direction the momentum points, flipping
from long to short (and back) when the opposite extreme breaks.

**Imported components**

| Component | Source | Role |
|---|---|---|
| Donchian 55/20 channel | Turtle Traders System 2 (Dennis & Eckhardt) | entry on a close through the prior 55-bar extreme; exit at the opposite 20-bar extreme |
| ADX ≥ 20 gate | J. Welles Wilder | rejects breakouts inside dead, directionless ranges |
| Volume confirmation ×1.3 | breakout literature | genuine crypto breakouts come with participation; quiet breaks are usually traps |

**Entry (long).** close > 55-bar high **and** ADX ≥ 20 **and** volume ≥
1.3× its 20-bar mean. Short is the mirror through the 55-bar low.
**Exit.** opposite 20-bar extreme, an opposite entry signal (flip), or
the hard 1% stop / 2R target.

**Tuning.** BTC on 1h churns in chop — raising `adx_min` to 25 and/or
`donchian_period` to 70 cuts trade count sharply and usually improves
expectancy at the cost of missing some early moves. `allow_short: false`
turns it into a long-only breakout system.

---

## 3. Trend following — Gold & Oil, 4-hour candles, noise-filtered

**Thesis.** Commodity trends on the 4h chart run for weeks. The
challenge is not finding the trend, it is (a) not being faked out by
intraday noise and (b) not giving back the wave on the first pullback.
Both are solved with explicit filters rather than tighter stops.

**Imported components**

| Component | Source | Role |
|---|---|---|
| EMA 20/50 cross + price confirmation | classic CTA trend definition | direction |
| ADX ≥ 22 gate | Wilder | requires a *real* trend before entering |
| Kaufman Efficiency Ratio ≥ 0.30 | P. Kaufman, *Smarter Trading* | **the noise filter**: ER = \|net move\| ÷ path length over 20 bars. Near 1 = clean directional wave; near 0 = chop. Entries are refused, and exits triggered, when the wave gets dirty |
| Chandelier exit 3×ATR | C. LeBeau | trailing stop off the best close since entry — ignores intraday wiggles smaller than 3 ATRs, so the system *rides the wave* instead of being shaken out |

**Entry (long).** EMA20 crosses above EMA50 (or a live bull trend first
passes the quality gates) **and** ADX ≥ 22 **and** ER ≥ 0.30.
**Exit.** opposite cross, ER collapse below 0.15, chandelier 3×ATR
trail, or the hard 1% stop / 2R target.

**Tuning.** `efficiency_ratio_min` is the noise dial: 0.35–0.40 trades
only the very cleanest waves. `trail_atr_mult` 3.5–4.0 rides longer but
gives back more at the end of the wave.

---

## How the 10-year validation works

`run_backtest.py` runs every strategy twice:

1. on its assigned intraday timeframe over the maximum free intraday
   history (60d of 15m, ~2y of 1h/4h) — validates execution behaviour,
   stop/target mechanics, trade frequency;
2. on **10 years of daily bars** with identical logic — validates that
   the underlying edge survives across regimes: the 2016–2018 bull, the
   2018 vol shock, the 2020 COVID crash, the 2021 melt-up, the 2022
   bear, and the 2023–2026 recovery.

A strategy is only trustworthy if the daily pass shows positive
expectancy across those regimes AND the intraday pass shows the
mechanics behave (stops hit at −1R, not −3R; reasonable trade counts).

## Overfitting warning

Do not tune parameters until the backtest looks perfect — that is
curve-fitting, and it is the fastest way to lose real money. Prefer
parameters that are *plateaus* (small changes barely change results)
over *peaks* (one magic value). The defaults shipped here are the
canonical values from the literature, not optimised ones, precisely so
they are less likely to be overfit.
