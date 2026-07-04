"""Merrill (Bank of America) integration — ALERT MODE.

THE HONEST TRUTH: Merrill Edge / Merrill Lynch does NOT offer a public
trading API, and screen-scraping or automating their website violates
their Terms of Service and will get the account locked. No third-party
library can legitimately place trades in a Merrill account.

So this adapter does the best legitimate thing possible:

  * every risk-approved order is rendered as a complete, ready-to-enter
    ORDER TICKET — symbol, side, quantity, limit price, stop price and
    target price — exactly matching the fields on Merrill's order entry
    screen, and
  * it is emailed/logged to you immediately, so entering it in the
    Merrill app takes ~15 seconds, and
  * the stop is a hard number you place AT THE BROKER as a stop order,
    which is what actually protects you while you are offline.

If you want genuine hands-off automated execution, open a brokerage
account with an API (Alpaca, Interactive Brokers, tastytrade) and add a
Broker subclass for it; the rest of this system is already compatible.
"""

from __future__ import annotations

import logging

from ai_trader.execution.broker import Broker, Order, OrderTicket
from ai_trader.notify.emailer import Emailer

logger = logging.getLogger(__name__)

_TICKET_TEMPLATE = """\
=========================================
 MERRILL ORDER TICKET  ({action})
=========================================
 Symbol        : {symbol}
 Action        : {side_h}
 Quantity      : {qty}
 Order type    : Limit
 Limit price   : {entry}
 --- protective bracket (enter as OTO/OCO if available) ---
 Stop order    : {stop}   (hard 1% risk cap)
 Target (limit): {target}
 Time in force : Day
 Signal        : {note}
=========================================
"""


class MerrillAlertBroker(Broker):
    def __init__(self, emailer: Emailer | None, alert_email: str):
        self.emailer = emailer
        self.alert_email = alert_email
        self.open_orders: dict[str, Order] = {}

    def _render(self, order: Order, action: str) -> str:
        side_h = {"buy": "Buy", "sell": "Sell"}.get(order.side, order.side)
        qty = int(order.qty) if order.qty >= 1 else round(order.qty, 6)
        return _TICKET_TEMPLATE.format(
            action=action, symbol=order.symbol, side_h=side_h, qty=qty,
            entry=f"{order.entry:.2f}", stop=f"{order.stop:.2f}",
            target=f"{order.target:.2f}", note=order.note,
        )

    def submit(self, order: Order) -> OrderTicket:
        ticket = self._render(order, "NEW POSITION")
        logger.info("\n%s", ticket)
        if self.emailer:
            self.emailer.send(
                to=self.alert_email,
                subject=f"[AI Trader] {order.side.upper()} {order.symbol} — order ticket",
                body=ticket,
            )
        self.open_orders[order.instrument] = order
        return OrderTicket(order, "sent", None,
                           "ticket emailed for manual entry at Merrill")

    def close(self, instrument: str, price: float, reason: str) -> OrderTicket | None:
        order = self.open_orders.pop(instrument, None)
        if order is None:
            return None
        close_side = "sell" if order.side == "buy" else "buy"
        msg = (f"CLOSE {order.symbol}: {close_side.upper()} {order.qty:.4f} "
               f"near {price:.2f} — reason: {reason}")
        logger.info(msg)
        if self.emailer:
            self.emailer.send(
                to=self.alert_email,
                subject=f"[AI Trader] CLOSE {order.symbol}",
                body=msg,
            )
        return OrderTicket(order, "sent", None, msg)
