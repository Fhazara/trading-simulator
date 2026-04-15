"""
strategies/base.py
Abstract base class for all trading strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from order_book import LimitOrderBook, Side, OrderType, Trade


# A lightweight spec that a strategy returns to the simulation engine
class OrderSpec:
    """Describes an order the strategy wants to submit."""
    __slots__ = ("side", "order_type", "quantity", "price")

    def __init__(
        self,
        side: "Side",
        order_type: "OrderType",
        quantity: float,
        price: Optional[float] = None,
    ) -> None:
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price


class BaseStrategy(ABC):
    """
    Base class all strategies inherit from.

    Responsibilities
    ────────────────
    • Maintain cash, position, and portfolio value history.
    • Provide generate_orders() for the simulation engine to call.
    • Record mark-to-market PnL at every bar.
    """

    def __init__(
        self,
        trader_id: str,
        initial_capital: float = 100_000.0,
        max_position: int = 1_000,
        warmup: int = 30,
    ) -> None:
        self.trader_id = trader_id
        self.initial_capital = initial_capital
        self.max_position = max_position
        self.warmup = warmup

        self.cash: float = initial_capital
        self.position: float = 0.0      # shares (positive = long, negative = short)

        self.pnl_history: List[float] = []       # PnL at each bar
        self.value_history: List[float] = [initial_capital]
        self.trade_count: int = 0
        self.realized_pnl: float = 0.0

    # ── to be implemented by subclasses ──────────────────────────────

    @abstractmethod
    def generate_orders(
        self,
        prices: pd.DataFrame,
        order_book: "LimitOrderBook",
        step: int,
    ) -> List[OrderSpec]:
        """
        Analyse the market and return a list of OrderSpec objects
        (may be empty) for the simulation engine to execute.

        Parameters
        ----------
        prices   : DataFrame with columns open/high/low/close/volume up to *step*
        order_book: current state of the shared order book
        step     : current bar index (0-based)
        """

    # ── fill accounting ───────────────────────────────────────────────

    def apply_fill(
        self,
        trade: "Trade",
        impact_per_share: float = 0.0,
    ) -> None:
        """
        Update cash and position from a trade, including market impact.
        impact_per_share is always positive; direction is applied here.
        """
        if trade.buyer_id == self.trader_id:
            effective_price = trade.price + impact_per_share
            self.cash -= effective_price * trade.quantity
            self.position += trade.quantity
        elif trade.seller_id == self.trader_id:
            effective_price = trade.price - impact_per_share
            self.cash += effective_price * trade.quantity
            self.position -= trade.quantity
        else:
            return
        self.trade_count += 1

    def mark_to_market(self, current_price: float) -> None:
        """Record portfolio value and PnL for the current bar."""
        val = self.portfolio_value(current_price)
        self.value_history.append(val)
        self.pnl_history.append(val - self.initial_capital)

    def portfolio_value(self, current_price: float) -> float:
        return self.cash + self.position * current_price

    # ── convenience helpers ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.trader_id

    def returns_series(self) -> np.ndarray:
        """Period-over-period returns from value_history."""
        vals = np.array(self.value_history, dtype=float)
        if len(vals) < 2:
            return np.array([])
        return np.diff(vals) / vals[:-1]
