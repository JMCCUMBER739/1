#!/usr/bin/env python3
"""Live trading loop.

Runs forever (deploy under systemd / Docker / a cloud VM so it keeps
working while you're offline):

  * every 15 minutes: refresh data for each instrument on its own
    timeframe, generate signals, have Claude review them, pass survivors
    through the risk engine (hard 1% stop, volatility sizing, double
    risk-on guard), then hand approved orders to the broker
    (paper = auto-executed simulation; merrill = order ticket emailed
    to you for one-tap entry).
  * every bar: manage open positions (stop / target / strategy exits).
  * at reports.morning_time: email the morning strategy briefing.
  * at reports.evening_time: email the evening results & analytics.

Usage:
  python3 run_live.py [--config config.yaml] [--once]

`--once` runs a single evaluation cycle and exits (useful for cron or
for testing).
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime

import pandas as pd
import schedule

from ai_trader.ai.claude import ClaudeAdvisor
from ai_trader.config import Config, load_config
from ai_trader.execution.broker import Order
from ai_trader.execution.merrill import MerrillAlertBroker
from ai_trader.execution.paper import PaperBroker
from ai_trader.notify.emailer import Emailer
from ai_trader.notify.reports import ReportBuilder
from ai_trader.risk.engine import OpenPosition, RiskEngine
from ai_trader.risk.metrics import performance_report
from ai_trader.data import load_history
from ai_trader.strategies import STRATEGY_REGISTRY

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("live")


class LiveTrader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.risk = RiskEngine(cfg)
        self.equity = cfg.equity
        self.trades_today: list[dict] = []
        self.equity_history: list[tuple[datetime, float]] = []

        smtp = cfg.reports.get("smtp", {})
        self.emailer = Emailer(smtp.get("host"), int(smtp.get("port", 587)),
                               smtp.get("username"), cfg.smtp_password)
        self.advisor = ClaudeAdvisor(
            cfg.anthropic_api_key,
            cfg.ai.get("logic_model", "claude-sonnet-4-20250514"),
            cfg.ai.get("report_model", "claude-sonnet-4-20250514"),
            float(cfg.ai.get("veto_threshold", 0.35)),
        )
        self.reports = ReportBuilder(self.emailer, self.advisor,
                                     cfg.reports.get("recipient_email", ""))

        if cfg.broker.get("provider") == "merrill":
            self.broker = MerrillAlertBroker(
                self.emailer, cfg.broker.get("merrill", {}).get("alert_email", ""))
        else:
            self.broker = PaperBroker()

        self.strategies = {
            key: STRATEGY_REGISTRY[inst.strategy](cfg.strategies[inst.strategy])
            for key, inst in cfg.instruments.items()
        }

    # ------------------------------------------------------------- data
    def _fresh_df(self, inst) -> pd.DataFrame | None:
        try:
            return load_history(inst.symbol, inst.timeframe, use_cache=False)
        except Exception:
            logger.exception("data refresh failed for %s", inst.symbol)
            return None

    # ------------------------------------------------------ trade cycle
    def cycle(self) -> None:
        logger.info("--- evaluation cycle ---")
        market_state: dict[str, dict] = {}

        for key, inst in self.cfg.instruments.items():
            df = self._fresh_df(inst)
            if df is None or len(df) < 250:
                continue
            price = float(df["close"].iloc[-1])
            market_state[key] = {"price": price,
                                 "time": str(df.index[-1])}

            # ---- manage open position -----------------------------------
            open_pos = self.risk.positions.get(key)
            if open_pos:
                hit_stop = (price <= open_pos.stop if open_pos.side == "long"
                            else price >= open_pos.stop)
                hit_target = (price >= open_pos.target if open_pos.side == "long"
                              else price <= open_pos.target)
                ann = self.strategies[key].annotate(df)
                strat_exit = bool(
                    ann.iloc[-1]["long_exit" if open_pos.side == "long"
                                 else "short_exit"])
                if hit_stop or hit_target or strat_exit:
                    reason = ("hard_stop" if hit_stop else
                              "target" if hit_target else "strategy_exit")
                    direction = 1 if open_pos.side == "long" else -1
                    pnl = (price - open_pos.entry) * open_pos.qty * direction
                    self.broker.close(key, price, reason)
                    self.risk.register_close(key, pnl)
                    self.equity += pnl
                    self.trades_today.append({
                        "instrument": key, "side": open_pos.side,
                        "entry": open_pos.entry, "exit": price,
                        "qty": open_pos.qty, "pnl": round(pnl, 2),
                        "reason": reason,
                    })
                continue  # never open a new trade on the same cycle we manage

            # ---- new signals ---------------------------------------------
            sig = self.strategies[key].latest_signal(
                df, key, self.cfg.risk.hard_stop_pct,
                float(self.cfg.orders.get("target_r_multiple", 2.0)))
            if sig is None:
                continue

            # manual entry/target/stop overrides, if configured
            if self.cfg.orders.get("mode") == "manual":
                manual = (self.cfg.orders.get("manual", {}) or {}).get(key, {})
                if manual.get("entry"):
                    sig.entry = float(manual["entry"])
                if manual.get("stop"):
                    sig.stop = float(manual["stop"])
                if manual.get("target"):
                    sig.target = float(manual["target"])

            # ---- Claude logic review --------------------------------------
            review = self.advisor.review_signal(
                sig.as_dict(),
                {"recent_closes": df["close"].tail(30).round(4).tolist(),
                 "instrument": key, "timeframe": inst.timeframe},
            )
            logger.info("Claude review %s: conf=%.2f %s", key,
                        review["confidence"], review["reasoning"])
            if not review["approved"]:
                logger.info("signal VETOED by AI review")
                continue

            # ---- risk engine -----------------------------------------------
            decision = self.risk.evaluate(sig, self.equity)
            logger.info("risk decision %s: approved=%s %s", key,
                        decision.approved, "; ".join(decision.reasons))
            if not decision.approved:
                continue

            order = Order(
                instrument=key, symbol=inst.merrill_symbol,
                side="buy" if sig.side == "long" else "sell",
                qty=decision.qty, entry=sig.entry,
                stop=decision.stop, target=decision.target,
                note=f"{sig.reason} | AI conf {review['confidence']:.2f}",
            )
            ticket = self.broker.submit(order)
            if ticket.status in ("filled", "sent"):
                self.risk.register_fill(OpenPosition(
                    instrument=key, side=sig.side, qty=decision.qty,
                    entry=sig.entry, stop=decision.stop,
                    target=decision.target, risk_on=inst.risk_on,
                ))

        self.equity_history.append((datetime.now(), self.equity))

    # ---------------------------------------------------------- reports
    def morning_report(self) -> None:
        state, planned = {}, []
        for key, inst in self.cfg.instruments.items():
            df = self._fresh_df(inst)
            if df is None:
                continue
            ann = self.strategies[key].annotate(df)
            last = ann.iloc[-1]
            state[key] = {
                "price": round(float(last["close"]), 4),
                "strategy": inst.strategy,
                "timeframe": inst.timeframe,
                "in_position": key in self.risk.positions,
            }
            sig = self.strategies[key].latest_signal(
                df, key, self.cfg.risk.hard_stop_pct)
            if sig:
                planned.append(sig.as_dict())
        guard = ("ACTIVE — S&P and NASDAQ both long, new risk-on trades blocked"
                 if self.risk._both_indices_long() else
                 "clear — new trades allowed")
        self.reports.send_morning(state, planned, guard)

    def evening_report(self) -> None:
        if len(self.equity_history) >= 3:
            eq = pd.Series(
                [e for _, e in self.equity_history],
                index=pd.DatetimeIndex([t for t, _ in self.equity_history]),
            )
            risk_rep = performance_report(eq)
        else:
            risk_rep = {"note": "not enough live history yet"}
        day_pnl = self.risk.day_pnl
        self.reports.send_evening(self.trades_today, day_pnl, self.equity, risk_rep)
        self.trades_today = []
        self.risk.reset_day()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--once", action="store_true",
                    help="run one evaluation cycle and exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    trader = LiveTrader(cfg)

    if args.once:
        trader.cycle()
        return

    schedule.every(15).minutes.do(trader.cycle)
    schedule.every().day.at(cfg.reports.get("morning_time", "07:30")).do(
        trader.morning_report)
    schedule.every().day.at(cfg.reports.get("evening_time", "17:30")).do(
        trader.evening_report)

    logger.info("live loop started (broker=%s, AI=%s, reports -> %s)",
                cfg.broker.get("provider"),
                "on" if trader.advisor.enabled else "off (no API key)",
                cfg.reports.get("recipient_email"))
    trader.cycle()
    while True:
        schedule.run_pending()
        time.sleep(20)


if __name__ == "__main__":
    main()
