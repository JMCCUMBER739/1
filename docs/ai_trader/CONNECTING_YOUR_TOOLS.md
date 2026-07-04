# Connecting your tools — Merrill, Claude, email, deployment

## 1. Merrill (Bank of America)

**The facts, so you can make an informed decision:**

- Merrill Edge / Merrill Lynch **does not offer a public trading API**.
  There is no official way for software to place orders in a Merrill
  account.
- Automating their website (screen-scraping, headless browsers, session
  hijacking) violates their Terms of Service and is a fast path to a
  locked account. Any tool claiming to "automate Merrill" is doing this.

**What this system does instead (`broker.provider: merrill`):**

1. Every risk-approved trade is rendered as a complete order ticket —
   symbol, action, quantity, limit price, stop price, target price —
   matching Merrill's order-entry screen field-for-field.
2. The ticket is emailed to `broker.merrill.alert_email` instantly and
   logged locally. Entering it in the Merrill app takes ~15 seconds.
3. **Crucially: place the stop order at Merrill** exactly as the ticket
   says. A stop resting at the broker protects you while you sleep, fly,
   or lose internet — that is the correct "offline" protection, better
   than any bot that needs your machine to be up.

**Instrument mapping** (`instruments.<name>.merrill_symbol`): Merrill
cannot hold spot Bitcoin, NASDAQ futures, or oil futures directly, so
the config maps signals to liquid ETFs tradeable in a Merrill account:
BTC→IBIT, S&P→SPY, NASDAQ→QQQ, gold→GLD, oil→USO. Change the mapping if
you prefer different vehicles (e.g. FBTC, VOO, IAU, DBO).

**If you want true hands-off automation:** open an account at a broker
with a real API — Alpaca (simplest, commission-free equities/ETFs),
Interactive Brokers (most instruments), tastytrade. Then implement one
class:

```python
from ai_trader.execution.broker import Broker, Order, OrderTicket

class AlpacaBroker(Broker):
    def submit(self, order: Order) -> OrderTicket:
        # POST /v2/orders with bracket (entry + stop + target)
        ...
    def close(self, instrument, price, reason) -> OrderTicket | None:
        ...
```

and set `broker.provider` accordingly. Nothing else in the system
changes.

## 2. Claude (Anthropic)

1. Get an API key at [console.anthropic.com](https://console.anthropic.com).
2. `export ANTHROPIC_API_KEY="sk-ant-..."` in the environment where the
   trader runs (or in the systemd unit / Docker env).
3. Pick models in `config.yaml`:

```yaml
ai:
  logic_model: claude-sonnet-4-20250514    # reviews every signal
  report_model: claude-sonnet-4-20250514   # writes the daily emails
  veto_threshold: 0.35
```

These are plain strings — set them to **any** model name your key can
access (including a "fable"-family model for logic, if your account has
one). Run `python3 -c "import anthropic; print([m.id for m in
anthropic.Anthropic().models.list()])"` to see what you have.

- **Logic role:** Claude receives each proposed trade plus market
  context and returns `{confidence, reasoning}`. Below
  `veto_threshold`, the trade is vetoed. Claude can only veto or
  approve — it can never invent a trade, so the quantitative rules and
  the 1% stop always remain in charge.
- **Reports role:** Claude writes the plain-English morning strategy
  briefing and evening analytics narrative around the raw numbers.

**About "Claude Cowork":** Cowork is a desktop collaboration product
and currently has no public API for sending emails/messages on your
behalf. The daily updates are therefore delivered by standard SMTP
email — same content, same morning/evening cadence, to the address you
choose. If Cowork ships an API later, only
`ai_trader/notify/emailer.py` needs replacing.

## 3. Email (your daily reports + Merrill tickets)

```yaml
reports:
  recipient_email: you@example.com   # <- your email, as requested
  morning_time: "07:30"
  evening_time: "17:30"
  smtp:
    host: smtp.gmail.com
    port: 587
    username: yourbot@gmail.com
```

```bash
export SMTP_PASSWORD="abcd efgh ijkl mnop"   # Gmail app password
```

For Gmail: enable 2-step verification, then create an **app password**
(Google Account → Security → App passwords) — regular passwords will
not work. Outlook/Office365: `smtp.office365.com:587`. Amazon SES also
works and is the cheap option for a VPS.

If SMTP is unconfigured or a send fails, every message is written to
`reports/outbox/` so nothing is ever lost.

## 4. Data

Default feed is Yahoo Finance via `yfinance` — free, no key, with
limits (60d of 15m bars, 730d of 1h). Bars are cached in `.data_cache/`
as parquet. To upgrade to a paid feed with deep intraday history
(Polygon.io, Databento, Tiingo), replace the download call inside
`ai_trader/data/loader.py -> load_history()`; the returned DataFrame
contract (UTC index, open/high/low/close/volume) is all downstream code
depends on.

## 5. Deployment (so it works while you're offline)

Any always-on Linux box (a $5/month VPS is fine):

```bash
sudo apt install python3-pip
git clone <repo> /opt/ai-trader && cd /opt/ai-trader
pip3 install -r requirements-trader.txt
# edit config.yaml, set env vars, then:
python3 run_backtest.py     # sanity-check data + charts first
python3 run_live.py         # or install the systemd unit from README
```

Docker alternative:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements-trader.txt .
RUN pip install -r requirements-trader.txt
COPY . .
CMD ["python3", "run_live.py"]
```

```bash
docker build -t ai-trader . && docker run -d --restart=always \
  -e ANTHROPIC_API_KEY -e SMTP_PASSWORD ai-trader
```

## 6. Recommended go-live sequence

1. `python3 run_backtest.py` — study the charts and `summary.json`.
2. Run in `paper` mode for at least 2–4 weeks; read every evening
   report; verify the trades and the risk numbers make sense to you.
3. Switch to `merrill` alert mode with **small size**
   (`risk.risk_per_trade_pct: 0.25`), entering tickets manually and
   always placing the stop at the broker.
4. Scale up only after the live results track the paper results.
