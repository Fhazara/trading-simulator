"""
order_book.py
Price-time priority limit order book engine supporting:
  - Limit orders  (rest in book if not immediately marketable)
  - Market orders (sweep through available liquidity)
  - Cancellations
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple


# ─────────────────────────── enumerations ────────────────────────────

class Side(Enum):
    BID = "BID"
    ASK = "ASK"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(Enum):
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


# ─────────────────────────── data classes ────────────────────────────

@dataclass
class Order:
    order_id: int
    trader_id: str
    side: Side
    order_type: OrderType
    quantity: float
    timestamp: int
    price: Optional[float] = None   # None → market order
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    status: OrderStatus = OrderStatus.OPEN

    @property
    def remaining_qty(self) -> float:
        return self.quantity - self.filled_qty

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)


@dataclass
class Trade:
    trade_id: int
    timestamp: int
    buyer_id: str
    seller_id: str
    price: float
    quantity: float
    bid_order_id: int
    ask_order_id: int


# ─────────────────────────── order book ──────────────────────────────

class LimitOrderBook:
    """
    Central limit order book using price-time priority (FIFO within level).

    Bids  → keyed by price descending (best bid = highest price)
    Asks  → keyed by price ascending  (best ask = lowest price)
    """

    def __init__(self) -> None:
        # price → FIFO queue of resting orders
        self.bids: Dict[float, Deque[Order]] = {}
        self.asks: Dict[float, Deque[Order]] = {}

        self.orders: Dict[int, Order] = {}   # all orders ever submitted
        self.trades: List[Trade] = []        # all executed trades

        self._oid = itertools.count(1)
        self._tid = itertools.count(1)
        self.timestamp: int = 0

    # ── public API ───────────────────────────────────────────────────

    def submit_order(
        self,
        trader_id: str,
        side: Side,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
    ) -> Tuple[Order, List[Trade]]:
        """
        Submit a new order. Returns (order, list_of_trades).
        Limit orders that are not fully filled rest in the book.
        Market orders that exhaust liquidity are partially filled / cancelled.
        """
        order = Order(
            order_id=next(self._oid),
            trader_id=trader_id,
            side=side,
            order_type=order_type,
            quantity=quantity,
            timestamp=self.timestamp,
            price=price,
        )
        self.orders[order.order_id] = order

        trades = self._match(order)

        # Resting logic
        if order.is_active and order.remaining_qty > 1e-9:
            if order_type == OrderType.LIMIT and price is not None:
                book = self.bids if side == Side.BID else self.asks
                book.setdefault(price, deque()).append(order)
            else:
                # Market order with no counterpart – cancel remainder
                order.status = (
                    OrderStatus.PARTIALLY_FILLED if order.filled_qty > 0
                    else OrderStatus.CANCELLED
                )

        return order, trades

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a resting order. Returns True if successful."""
        order = self.orders.get(order_id)
        if order is None or not order.is_active:
            return False

        book = self.bids if order.side == Side.BID else self.asks
        level = book.get(order.price)
        if level is not None:
            try:
                level.remove(order)
            except ValueError:
                pass
            if not level:
                del book[order.price]

        order.status = OrderStatus.CANCELLED
        return True

    def cancel_all_orders(self, trader_id: str) -> List[int]:
        """Cancel every active order belonging to *trader_id*."""
        cancelled: List[int] = []
        for order in list(self.orders.values()):
            if order.trader_id == trader_id and order.is_active:
                if self.cancel_order(order.order_id):
                    cancelled.append(order.order_id)
        return cancelled

    # ── book queries ─────────────────────────────────────────────────

    def best_bid(self) -> Optional[float]:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> Optional[float]:
        return min(self.asks) if self.asks else None

    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        return (bb + ba) / 2 if bb is not None and ba is not None else None

    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        return (ba - bb) if bb is not None and ba is not None else None

    def depth(self, levels: int = 10) -> Dict[str, List[Tuple[float, float]]]:
        """Return aggregated bid / ask depth (price, cumulative_qty)."""
        def _agg(book: dict, descending: bool) -> List[Tuple[float, float]]:
            rows: List[Tuple[float, float]] = []
            prices = sorted(book.keys(), reverse=descending)[:levels]
            cum = 0.0
            for p in prices:
                qty = sum(o.remaining_qty for o in book[p] if o.is_active)
                if qty > 0:
                    cum += qty
                    rows.append((p, cum))
            return rows

        return {
            "bids": _agg(self.bids, descending=True),
            "asks": _agg(self.asks, descending=False),
        }

    def total_bid_volume(self) -> float:
        return sum(o.remaining_qty
                   for q in self.bids.values() for o in q if o.is_active)

    def total_ask_volume(self) -> float:
        return sum(o.remaining_qty
                   for q in self.asks.values() for o in q if o.is_active)

    def active_orders(self, trader_id: str) -> List[Order]:
        return [o for o in self.orders.values()
                if o.trader_id == trader_id and o.is_active]

    def purge_inactive(self) -> int:
        """
        Remove filled / cancelled orders from self.orders to prevent
        unbounded dict growth.  Safe to call at any time because the
        bid/ask price-level structures hold live Order object references;
        deleted entries are already inactive and not reachable via those
        structures.
        """
        stale = [oid for oid, o in self.orders.items() if not o.is_active]
        for oid in stale:
            del self.orders[oid]
        return len(stale)

    # ── internal matching ─────────────────────────────────────────────

    def _match(self, incoming: Order) -> List[Trade]:
        trades: List[Trade] = []

        if incoming.side == Side.BID:
            opposite = self.asks
            sorted_prices = sorted(opposite.keys())
            def crosses(level_price: float) -> bool:
                if incoming.order_type == OrderType.MARKET:
                    return True
                return incoming.price is not None and level_price <= incoming.price
        else:
            opposite = self.bids
            sorted_prices = sorted(opposite.keys(), reverse=True)
            def crosses(level_price: float) -> bool:
                if incoming.order_type == OrderType.MARKET:
                    return True
                return incoming.price is not None and level_price >= incoming.price

        for lvl_price in sorted_prices:
            if incoming.remaining_qty <= 1e-9:
                break
            if not crosses(lvl_price):
                break

            queue = opposite[lvl_price]
            while queue and incoming.remaining_qty > 1e-9:
                resting = queue[0]
                if not resting.is_active:
                    queue.popleft()
                    continue

                fill_qty = min(incoming.remaining_qty, resting.remaining_qty)
                fill_price = lvl_price  # price-improvement: fill at resting price

                # Update both orders
                self._apply_fill(incoming, fill_qty, fill_price)
                self._apply_fill(resting, fill_qty, fill_price)

                # Build trade record
                if incoming.side == Side.BID:
                    buyer_id, seller_id = incoming.trader_id, resting.trader_id
                    bid_oid, ask_oid = incoming.order_id, resting.order_id
                else:
                    buyer_id, seller_id = resting.trader_id, incoming.trader_id
                    bid_oid, ask_oid = resting.order_id, incoming.order_id

                trade = Trade(
                    trade_id=next(self._tid),
                    timestamp=self.timestamp,
                    buyer_id=buyer_id,
                    seller_id=seller_id,
                    price=fill_price,
                    quantity=fill_qty,
                    bid_order_id=bid_oid,
                    ask_order_id=ask_oid,
                )
                trades.append(trade)
                self.trades.append(trade)

                if resting.remaining_qty <= 1e-9:
                    queue.popleft()

            if not queue:
                del opposite[lvl_price]

        # Final status for incoming
        if incoming.filled_qty >= incoming.quantity - 1e-9:
            incoming.status = OrderStatus.FILLED
        elif incoming.filled_qty > 0:
            incoming.status = OrderStatus.PARTIALLY_FILLED

        return trades

    @staticmethod
    def _apply_fill(order: Order, qty: float, price: float) -> None:
        prev_filled = order.filled_qty
        order.filled_qty += qty
        # Running average fill price
        if order.filled_qty > 0:
            order.avg_fill_price = (
                order.avg_fill_price * prev_filled + price * qty
            ) / order.filled_qty
        if order.filled_qty >= order.quantity - 1e-9:
            order.status = OrderStatus.FILLED
        elif order.filled_qty > 0:
            order.status = OrderStatus.PARTIALLY_FILLED
