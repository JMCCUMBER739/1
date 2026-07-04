"""Paper broker — simulated execution with a persistent trade blotter.

This is the default and the safe way to run the system 24/7 ("offline"
from your point of view): it trades automatically in simulation, and the
daily reports tell you exactly what it did and why.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from ai_trader.execution.broker import Broker, Order, OrderTicket

logger = logging.getLogger(__name__)


class PaperBroker(Broker):
    def __init__(self, blotter_path: str | Path = "reports/paper_blotter.csv"):
        self.blotter_path = Path(blotter_path)
        self.open_orders: dict[str, Order] = {}
        self.blotter_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.blotter_path.exists():
            with open(self.blotter_path, "w", newline="") as fh:
                csv.writer(fh).writerow(
                    ["time", "instrument", "symbol", "side", "qty",
                     "price", "stop", "target", "event", "note"]
                )

    def _log(self, order: Order, event: str, price: float, note: str = "") -> None:
        with open(self.blotter_path, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [order.created_at.isoformat(), order.instrument, order.symbol,
                 order.side, f"{order.qty:.6f}", f"{price:.4f}",
                 f"{order.stop:.4f}", f"{order.target:.4f}", event, note]
            )

    def submit(self, order: Order) -> OrderTicket:
        self.open_orders[order.instrument] = order
        self._log(order, "fill", order.entry, order.note)
        logger.info("PAPER FILL %s %s %.4f @ %.4f (stop %.4f, target %.4f)",
                    order.side, order.symbol, order.qty, order.entry,
                    order.stop, order.target)
        return OrderTicket(order, "filled", order.entry, "paper fill")

    def close(self, instrument: str, price: float, reason: str) -> OrderTicket | None:
        order = self.open_orders.pop(instrument, None)
        if order is None:
            return None
        self._log(order, "close", price, reason)
        logger.info("PAPER CLOSE %s @ %.4f (%s)", order.symbol, price, reason)
        return OrderTicket(order, "filled", price, reason)
