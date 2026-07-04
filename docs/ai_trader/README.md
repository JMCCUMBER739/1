# AI Trader — user guide

A multi-asset, multi-timeframe systematic trading engine with an AI
(Claude) logic layer, hard risk controls, daily email reporting, and a
full backtesting + visualization suite.

| Instrument | Data symbol | Traded as (Merrill) | Strategy | Timeframe |
|---|---|---|---|---|
| S&P 500 | SPY | SPY | Mean reversion | 15-minute |
| NASDAQ 100 | QQQ | QQQ | Mean reversion | 15-minute |
| Bitcoin | BTC-USD | IBIT (spot-BTC ETF) | Momentum breakout, long+short switching | 1-hour |
| Gold | GLD | GLD | Trend following, noise-filtered | 4-hour |
| Crude Oil | USO | USO | Trend following, noise-filtered | 4-hour |

## Read this first — honest expectations

Three things in the original request cannot be delivered by anyone,
and pretending otherwise would cost you money:

1. **"I want to only win."** No system only wins. These are
   positive-expectancy systems: individual trades lose (that is what
   the hard 1% stop is for), and the edge shows up across many trades.
   The backtest reports in `reports/backtest/` show the real,
   unflattered numbers, including losing streaks and drawdowns.
2. **Merrill has no trading API.** Merrill Edge / Merrill Lynch does
   not offer a public API, and automating their website violates their
   Terms of Service. This system therefore supports two modes:
   `paper` (fully automatic simulated trading, safe default) and
   `merrill` alert mode (every approved trade is emailed to you as a
   complete order ticket — symbol, side, quantity, limit, stop, target —
   for ~15-second manual entry in the Merrill app). Your true "offline"
   protection is the **stop order you place at the broker**, which the
   ticket includes. For genuine hands-free execution, use a broker with
   an API (Alpaca, Interactive Brokers) — the `Broker` interface in
   `ai_trader/execution/broker.py` is ready for that.
3. **10 years of 15-minute data is not freely available.** Free data
   (Yahoo Finance) provides ~60 days of 15m bars, ~730 days of 1h/4h
   bars, and 10+ years of daily bars. The backtester therefore runs two
   passes: the exact intraday strategies on maximum intraday history,
   plus a 10-year daily pass of the same logic to validate the edge
   across full regimes (2016–2026: bull, COVID crash, 2022 bear,
   recovery). To backtest 10 years of true intraday data, plug a paid
   feed (Polygon.io, Databento, Tiingo) into `ai_trader/data/loader.py`.

## Languages and stack

- **Python 3.10+** — the entire system. Python is the standard for
  quantitative trading research and mid-frequency execution.
- **pandas / numpy** — vectorised indicator and backtest math.
- **matplotlib** — the visualization suite (dark-theme dashboards).
- **yfinance** — free market data (swappable).
- **anthropic** — Claude API for the AI logic review + report writing.
- **PyYAML** — one human-editable `config.yaml` controls everything.
- **schedule** — in-process cron for the trade cycle and daily emails.

No compiled components; runs anywhere Python runs (laptop, VPS, Docker).

## Installation

```bash
git clone <this repo> && cd <repo>
pip install -r requirements-trader.txt
```

Set your secrets as environment variables (never in files):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # enables the Claude AI layer
export SMTP_PASSWORD="your-app-password" # enables real email delivery
```

Then edit `config.yaml`:

- `reports.recipient_email` — **the email address your daily reports go
  to** (morning strategy + evening analytics). This is user-specified,
  as requested.
- `reports.smtp.*` — your SMTP account (for Gmail: create an
  [app password](https://support.google.com/accounts/answer/185833)).
- `account.equity` — your account size, used for position sizing.
- `broker.provider` — `paper` (default) or `merrill` (alert mode).

## Running

### Running from Spyder (or any IDE)

`ai_trader` is not a pip package — it is the source folder inside this
repository, so the whole repo must be on your disk and the scripts must
be able to find it. Checklist if you see `ModuleNotFoundError: No
module named 'ai_trader'`:

1. **Get the full repository, on the right branch.** The trader lives
   on the feature branch until the PR is merged:

   ```bash
   git clone <repo-url>
   cd <repo>
   git checkout cursor/ai-trader-24ab
   ```

   After cloning you should see the `ai_trader/` folder next to
   `run_backtest.py`. If it is missing, you only have `master`.
2. **Open `run_backtest.py` / `run_live.py` from that folder** and run
   them (F5). Both scripts add their own folder to Python's module
   path at startup, so Spyder's `runfile()` works out of the box.
3. **Install the dependencies into the Python that Spyder uses.** In
   Spyder's IPython console run:

   ```python
   %pip install -r requirements-trader.txt
   ```

   (Using `pip` in a different terminal can install into a different
   Python environment than the one Spyder runs — a common trap.)
4. Alternatively, set Spyder's working directory to the repo root:
   Tools → Preferences → Run → "Working directory" → the folder
   containing `config.yaml`.

### 1. Backtest + visualizations (do this first)

```bash
python3 run_backtest.py            # uses cached data if present
python3 run_backtest.py --no-cache # force fresh downloads
```

Outputs to `reports/backtest/`:

- `<instrument>_backtest.png` — price with indicators and every
  entry/exit, equity curve, drawdown, R-multiple distribution.
- `10y/…` — the same charts for the 10-year daily validation pass.
- `portfolio_dashboard.png` — all equity curves + metrics table.
- `summary.json` — machine-readable metrics for both passes.

### 2. Live loop

```bash
python3 run_live.py           # runs forever: trade cycle every 15 min
python3 run_live.py --once    # single evaluation cycle (cron-friendly)
```

What the loop does each cycle:

1. Refreshes data for each instrument on its own timeframe.
2. Manages open positions (hard stop / target / strategy exits).
3. Generates new signals from the three strategy engines.
4. Sends each signal to Claude for a logic review; low-confidence
   signals are vetoed (`ai.veto_threshold`).
5. Passes survivors through the risk engine: hard 1% stop clamp,
   volatility-adjusted sizing, double risk-on guard, exposure caps,
   daily loss circuit breaker.
6. Executes via the configured broker (paper fill or Merrill ticket
   email).

Daily emails (to `reports.recipient_email`):

- **Morning (07:30 default)** — strategy briefing: market posture per
  instrument, pending signals with recommended entry/target/stop,
  exposure-guard status. Narrated by Claude when the API key is set.
- **Evening (17:30 default)** — results & analytics: every trade taken
  and why, day P&L, Sharpe/Sortino/drawdown/VaR/expectancy, and
  plain-English risk feedback.

### Keeping it running while you're away

Run it on an always-on machine. Example systemd unit
(`/etc/systemd/system/ai-trader.service`):

```ini
[Unit]
Description=AI Trader live loop
After=network-online.target

[Service]
WorkingDirectory=/opt/ai-trader
Environment=ANTHROPIC_API_KEY=sk-ant-...
Environment=SMTP_PASSWORD=...
ExecStart=/usr/bin/python3 run_live.py
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now ai-trader
```

Remember: in `merrill` mode the engine cannot click the buy button for
you — the stop order **you place at Merrill from the emailed ticket** is
what protects the position while you are offline.

## Entry / target / stop: recommended vs. manual

`orders.mode` in `config.yaml`:

- `recommended` (default): entry = signal price; stop = min(1.5×ATR,
  hard 1% cap); target = `orders.target_r_multiple` × stop distance
  (default 2R).
- `manual`: your numbers in `orders.manual.<instrument>` override the
  recommendations (the hard 1% stop cap is still enforced — the engine
  will never accept a stop wider than 1%).

## The strategies (what was "imported" and why)

Full detail in [STRATEGIES.md](STRATEGIES.md). In one line each:

- **Mean reversion (S&P/NASDAQ, 15m)** — Bollinger/z-score stretch +
  Connors RSI-2 panic trigger, traded only in the direction of the
  200-EMA regime; exits at the band mean or on a time stop.
- **Momentum breakout (BTC, 1h)** — Turtle-style Donchian 55/20 channel
  breakout with Wilder ADX trend gate and volume confirmation; flips
  long/short with the momentum ("switching").
- **Trend following (gold/oil, 4h)** — EMA 20/50 trend with ADX gate and
  a Kaufman Efficiency Ratio ≥ 0.30 "clean wave" noise filter;
  chandelier 3×ATR trailing exit rides the wave and ignores intraday
  wiggles smaller than 3 ATRs.

## Risk management (non-negotiable layer)

- **Hard 1% stop** on every position — the risk engine clamps every
  stop to at most 1% adverse movement from entry; the backtester
  fills stops intrabar (pessimistically, before targets).
- **Volatility-adjusted sizing** — size is the minimum of
  fixed-fractional risk sizing (`risk.risk_per_trade_pct` of equity per
  trade) and annualised volatility targeting
  (`risk.vol_target_annual_pct`), so positions automatically shrink in
  wild markets and grow (capped) in calm ones.
- **Double risk-on guard** — when S&P **and** NASDAQ are both long, no
  new risk-on trade (index or BTC long) is accepted until one closes.
- **Portfolio caps** — max open positions, max gross exposure, and a
  daily loss circuit breaker (`risk.daily_loss_limit_pct`).
- **Risk measurement & feedback** — Sharpe, Sortino, CAGR, max
  drawdown, Calmar, 95% VaR, win rate, profit factor, expectancy in R;
  delivered every evening with plain-English feedback
  (`ai_trader/risk/metrics.py`).

## Claude integration notes

- `ai.logic_model` / `ai.report_model` are plain config strings — set
  them to any Anthropic model your key can access (e.g. a
  "fable"-family model for logic if it is available on your account).
- "Claude Cowork" has no public send-email API, so daily delivery uses
  standard SMTP email to the address you configure — same cadence
  (morning strategy / evening analytics). If Cowork gains an API, only
  `ai_trader/notify/emailer.py` needs to change.
- Without `ANTHROPIC_API_KEY` the system still runs, pure-quant:
  signals pass through unvetoed and reports contain the raw numbers
  without narrative.

## Repository map

```
config.yaml                     one file controls everything
run_backtest.py                 backtests + charts + summary.json
run_live.py                     live loop + daily reports
ai_trader/
  config.py                     typed config loading & validation
  data/loader.py                data download, caching, 4h resampling
  indicators/core.py            RSI, Bollinger, ATR, ADX, Donchian, ER…
  strategies/                   the three strategy engines + Signal API
  risk/engine.py                sizing, hard stop, guards, circuit breaker
  risk/metrics.py               performance & risk analytics + feedback
  execution/paper.py            simulated broker with CSV blotter
  execution/merrill.py          Merrill alert-mode order tickets
  ai/claude.py                  Claude signal review + report narration
  notify/emailer.py             SMTP with outbox fallback
  notify/reports.py             morning/evening report assembly
  backtest/engine.py            event-driven backtester
  backtest/visualize.py         chart suite
docs/ai_trader/
  README.md                     this file
  STRATEGIES.md                 strategy theory, parameters, tuning
  CONNECTING_YOUR_TOOLS.md      Merrill, Claude, email, deployment
```

## Disclaimer

This software is for research and education. It is not investment
advice. Markets involve substantial risk of loss; past (and backtested)
performance does not guarantee future results. Test in `paper` mode
first, size small, and never risk money you cannot afford to lose.
