#!/usr/bin/env python3
"""Run the full backtest suite and generate visualizations.

Two passes per instrument:

  1. INTRADAY pass — each strategy on its assigned timeframe
     (15m for S&P/NASDAQ, 1h for BTC, 4h for gold/oil) over the maximum
     intraday history the free data source provides.
  2. 10-YEAR pass — the same strategy logic on daily bars over 10 years,
     validating the edge across full market regimes (2016-2026).

Outputs:
  reports/backtest/<instrument>_backtest.png        (intraday pass)
  reports/backtest/10y/<instrument>_backtest.png    (10-year pass)
  reports/backtest/portfolio_dashboard.png
  reports/backtest/summary.json

Usage:
  python3 run_backtest.py [--config config.yaml] [--no-cache]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make the repo root importable no matter how this script is launched
# (Spyder's runfile(), IDEs, cron, double-click) so `import ai_trader`
# always resolves to the sibling ai_trader/ folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai_trader.backtest.engine import run_backtest
from ai_trader.backtest.visualize import plot_instrument, plot_portfolio
from ai_trader.config import load_config
from ai_trader.data import load_history
from ai_trader.strategies import STRATEGY_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backtest")


def run_pass(cfg, timeframe_override: str | None, out_dir: Path,
             use_cache: bool) -> dict:
    results = {}
    for key, inst in cfg.instruments.items():
        tf = timeframe_override or inst.timeframe
        try:
            df = load_history(inst.symbol, tf, use_cache=use_cache)
        except Exception:
            logger.exception("data load failed for %s %s — skipping", inst.symbol, tf)
            continue
        strat = STRATEGY_REGISTRY[inst.strategy](cfg.strategies[inst.strategy])
        res = run_backtest(
            strategy=strat,
            df=df,
            instrument=key,
            timeframe=tf,
            initial_equity=cfg.equity,
            hard_stop_pct=cfg.risk.hard_stop_pct,
            risk_per_trade_pct=cfg.risk.risk_per_trade_pct,
            vol_target_annual_pct=cfg.risk.vol_target_annual_pct,
            target_r=float(cfg.orders.get("target_r_multiple", 2.0)),
        )
        results[key] = res
        plot_instrument(res, out_dir)
        logger.info("%s [%s] -> %s", key, tf, res.report)
    if results:
        plot_portfolio(results, out_dir)
    return {k: r.report for k, r in results.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path("reports/backtest")
    use_cache = not args.no_cache

    logger.info("=== PASS 1: intraday timeframes (max free history) ===")
    intraday = run_pass(cfg, None, out, use_cache)

    logger.info("=== PASS 2: 10-year daily regime validation ===")
    ten_year = run_pass(cfg, "1d", out / "10y", use_cache)

    summary = {"intraday_pass": intraday, "ten_year_daily_pass": ten_year}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("summary written to %s", out / "summary.json")

    print("\n================ BACKTEST SUMMARY ================")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
