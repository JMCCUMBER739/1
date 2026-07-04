"""Daily report builder — morning strategy + evening results & analytics.

Assembles the quantitative payload, optionally has Claude write the
narrative, and emails it to the configured recipient.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from ai_trader.ai.claude import ClaudeAdvisor
from ai_trader.notify.emailer import Emailer
from ai_trader.risk.metrics import feedback

logger = logging.getLogger(__name__)


def _fmt_dict(d: dict[str, Any], indent: str = "  ") -> str:
    return "\n".join(f"{indent}{k}: {v}" for k, v in d.items())


class ReportBuilder:
    def __init__(self, emailer: Emailer, advisor: ClaudeAdvisor, recipient: str):
        self.emailer = emailer
        self.advisor = advisor
        self.recipient = recipient

    # ---------------------------------------------------------- morning
    def send_morning(self, market_state: dict[str, Any],
                     planned: list[dict[str, Any]],
                     guard_status: str) -> None:
        today = datetime.now().strftime("%A, %B %d %Y")
        payload = {"date": today, "market_state": market_state,
                   "planned_signals": planned, "exposure_guard": guard_status}
        narrative = self.advisor.narrate("morning", payload)

        lines = [f"AI TRADER — MORNING STRATEGY — {today}", "=" * 50, ""]
        if narrative:
            lines += [narrative, "", "-" * 50, "RAW DATA", ""]
        lines.append("Market state:")
        for inst, state in market_state.items():
            lines.append(f"  {inst}:")
            lines.append(_fmt_dict(state, "    "))
        lines.append("")
        if planned:
            lines.append("Actionable signals at the open:")
            for sig in planned:
                lines.append(_fmt_dict(sig, "  "))
                lines.append("")
        else:
            lines.append("No actionable signals right now — standing by.")
        lines += ["", f"Exposure guard: {guard_status}"]

        self.emailer.send(self.recipient,
                          f"[AI Trader] Morning strategy — {today}",
                          "\n".join(lines))

    # ---------------------------------------------------------- evening
    def send_evening(self, trades_today: list[dict[str, Any]],
                     day_pnl: float, equity: float,
                     risk_report: dict[str, Any]) -> None:
        today = datetime.now().strftime("%A, %B %d %Y")
        payload = {"date": today, "trades": trades_today, "day_pnl": day_pnl,
                   "equity": equity, "risk_metrics": risk_report}
        narrative = self.advisor.narrate("evening", payload)

        lines = [f"AI TRADER — EVENING RESULTS & ANALYTICS — {today}", "=" * 50, ""]
        if narrative:
            lines += [narrative, "", "-" * 50, "RAW DATA", ""]
        lines.append(f"Day P&L: {day_pnl:+,.2f}   |   Equity: {equity:,.2f}")
        lines.append("")
        if trades_today:
            lines.append("Trades:")
            for t in trades_today:
                lines.append(_fmt_dict(t, "  "))
                lines.append("")
        else:
            lines.append("No trades executed today.")
        lines += ["", "Risk metrics:", _fmt_dict(risk_report), "",
                  "Risk feedback:"]
        lines += [f"  - {n}" for n in feedback(risk_report)]

        self.emailer.send(self.recipient,
                          f"[AI Trader] Evening analytics — {today}",
                          "\n".join(lines))
