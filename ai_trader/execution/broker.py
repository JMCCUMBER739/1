"""Broker abstraction.

All execution goes through the Broker interface so the engine is
broker-agnostic: PaperBroker simulates fills; MerrillAlertBroker turns
approved orders into ready-to-tap order tickets (because Merrill offers
no public trading API — see docs/ai_trader/CONNECTING_YOUR_TOOLS.md).
A future AlpacaBroker/IBKRBroker can be added by implementing `submit`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Order:
    instrument: str
    symbol: str            # broker-tradeable ticker (merrill_symbol)
    side: str              # "buy" | "sell" (sell == open short / close long)
    qty: float
    entry: float
    stop: float
    target: float
    note: str = ""
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class OrderTicket:
    """Result of submitting an order."""

    order: Order
    status: str            # "filled" | "sent" | "rejected"
    fill_price: float | None = None
    message: str = ""


class Broker(ABC):
    @abstractmethod
    def submit(self, order: Order) -> OrderTicket:
        ...

    @abstractmethod
    def close(self, instrument: str, price: float, reason: str) -> OrderTicket | None:
        ...
