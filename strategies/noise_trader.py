"""
strategies/noise_trader.py
Random noise trader – uninformed, liquidity-consuming order flow.

Logic
─────
• At each bar, trade with probability `trade_prob`.
• Buy or sell with equal probability.
• Random order size in [min_qty, max_qty].
• 65 % of orders are marketable limits (1 tick aggressive);
  35 % are market orders.
• Models uninformed retail / algorithmic noise that provides a
  constant stream of order flow for other strategies to trade against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
import pandas as pd

from .base import BaseStrategy, OrderSpec
from order_book import Side, OrderType

if TYPE_CHECKING:
    from order_book import LimitOrderBook


class NoiseTraderStrategy(BaseStrategy):
    """
    Uniformly random buy/sell noise trader.

    Parameters
    ----------
    trade_prob : float
        Probability of submitting an order at any given bar.
    min_qty, max_qty : int
        Range of random order sizes.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        trader_id: str = "NoiseTrader",
        initial_capital: float = 100_000.0,
        max_position: int = 300,
        warmup: int = 5,
        trade_prob: float = 0.25,
        min_qty: int = 20,
        max_qty: int = 150,
        seed: int = 99,
    ) -> None:
        super().__init__(trader_id, initial_capital, max_position, warmup)
        self.trade_prob = trade_prob
        self.min_qty = min_qty
        self.max_qty = max_qty
        self._rng = np.random.default_rng(seed)

    def generate_orders(
        self,
        prices: pd.DataFrame,
        order_book: "LimitOrderBook",
        step: int,
    ) -> List[OrderSpec]:
        # Cancel-and-refresh: resting aggressive limits from the previous bar
        # can fill at stale prices if not cancelled, causing position to
        # accumulate beyond max_position over many bars.
        order_book.cancel_all_orders(self.trader_id)

        if step < self.warmup:
            return []

        # Random decision: trade this bar?
        if self._rng.random() > self.trade_prob:
            return []

        mid = prices["close"].iloc[step]
        qty = int(self._rng.integers(self.min_qty, self.max_qty + 1))

        # Buy or sell randomly
        side = Side.BID if self._rng.random() < 0.5 else Side.ASK

        # Position limits
        if side == Side.BID and self.position + qty > self.max_position:
            qty = max(0, self.max_position - int(self.position))
        elif side == Side.ASK and self.position - qty < -self.max_position:
            qty = max(0, self.max_position + int(self.position))
        if qty == 0:
            return []

        # Order type
        use_market = self._rng.random() < 0.35
        if use_market:
            return [OrderSpec(side, OrderType.MARKET, qty)]
        else:
            # Slightly aggressive limit (inside mid ± 1-2 ticks)
            tick = round(self._rng.uniform(0.01, 0.05), 2)
            if side == Side.BID:
                price = round(mid + tick, 2)
            else:
                price = round(mid - tick, 2)
            return [OrderSpec(side, OrderType.LIMIT, qty, price)]
