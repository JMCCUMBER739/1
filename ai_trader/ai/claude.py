"""Claude integration — the "AI" in AI Trader.

Two jobs, two configurable models (config.yaml -> ai.*):

  1. LOGIC (ai.logic_model): every strategy signal is sent to Claude with
     full market context. Claude returns a confidence score and reasoning;
     signals scoring below ai.veto_threshold are vetoed. This is a sanity
     layer on top of the quantitative rules, never a replacement for them
     — Claude can only veto or approve, it cannot invent trades.

  2. REPORTS (ai.report_model): Claude writes the plain-English morning
     strategy briefing and the evening results/analytics narrative that
     get emailed to you.

Note on model names: set ai.logic_model / ai.report_model to whatever
Anthropic model your API key can access (e.g. a "fable"-family model if
one is available on your account). "Claude Cowork" is a desktop
collaboration product without a public send-email API, so delivery here
is standard SMTP email — same daily cadence, addressed to the email you
configure. If Cowork later exposes an API, only Emailer needs changing.

Everything degrades gracefully: with no ANTHROPIC_API_KEY the system
runs pure-quant (signals pass through unvetoed, reports are generated
from the raw numbers without narrative).
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_LOGIC_SYSTEM = """You are a rigorous quantitative trading risk reviewer.
You receive one proposed trade from a rules-based system plus market context.
Respond ONLY with JSON: {"confidence": <0..1>, "reasoning": "<max 60 words>"}.
Confidence reflects whether current context (trend regime, volatility,
recent behaviour) supports THIS specific signal. Be skeptical: fighting a
strong higher-timeframe trend, entering right before dead liquidity hours,
or chasing an exhausted move should score low."""

_REPORT_SYSTEM = """You are a professional trading assistant writing a daily
email to the account owner. Be concrete, numerate and honest — never hide
losses and never promise profits. Keep it under 400 words, plain text."""


class ClaudeAdvisor:
    def __init__(self, api_key: str | None, logic_model: str, report_model: str,
                 veto_threshold: float = 0.35):
        self.logic_model = logic_model
        self.report_model = report_model
        self.veto_threshold = veto_threshold
        self._client = None
        if api_key:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=api_key)
            except Exception:
                logger.exception("anthropic client init failed; running pure-quant")

    @property
    def enabled(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------ logic
    def review_signal(self, signal: dict[str, Any],
                      context: dict[str, Any]) -> dict[str, Any]:
        """Return {"approved": bool, "confidence": float, "reasoning": str}."""
        if not self.enabled:
            return {"approved": True, "confidence": signal.get("confidence", 0.5),
                    "reasoning": "AI review disabled (no API key) — quant rules only"}
        try:
            msg = self._client.messages.create(
                model=self.logic_model,
                max_tokens=300,
                system=_LOGIC_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": ("Proposed trade:\n"
                                + json.dumps(signal, default=str)
                                + "\n\nMarket context:\n"
                                + json.dumps(context, default=str)),
                }],
            )
            text = msg.content[0].text.strip()
            start, end = text.find("{"), text.rfind("}") + 1
            parsed = json.loads(text[start:end])
            conf = float(parsed.get("confidence", 0.5))
            return {
                "approved": conf >= self.veto_threshold,
                "confidence": conf,
                "reasoning": str(parsed.get("reasoning", "")),
            }
        except Exception:
            logger.exception("Claude review failed — passing signal through")
            return {"approved": True, "confidence": signal.get("confidence", 0.5),
                    "reasoning": "AI review unavailable; quant rules only"}

    # ------------------------------------------------------------ reports
    def narrate(self, kind: str, payload: dict[str, Any]) -> str:
        """kind: 'morning' | 'evening'. Returns the narrative body text."""
        if not self.enabled:
            return ""
        prompt = {
            "morning": ("Write the MORNING STRATEGY briefing from this data. "
                        "Cover: market posture per instrument, pending/possible "
                        "signals, planned risk per trade, and the exposure-guard "
                        "status."),
            "evening": ("Write the EVENING RESULTS & ANALYTICS report from this "
                        "data. Cover: trades taken and why, P&L, risk metrics, "
                        "what worked / what did not, and any risk feedback."),
        }[kind]
        try:
            msg = self._client.messages.create(
                model=self.report_model,
                max_tokens=800,
                system=_REPORT_SYSTEM,
                messages=[{"role": "user",
                           "content": prompt + "\n\nDATA:\n"
                           + json.dumps(payload, default=str)}],
            )
            return msg.content[0].text.strip()
        except Exception:
            logger.exception("Claude narration failed")
            return ""
