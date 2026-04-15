"""
strategies/market_maker.py
Market-making strategy: continuously quotes both sides of the book.

Logic
─────
• At every bar, cancel all existing quotes and post fresh ones.
• The fair-value mid is the AAPL close price.
• Quoted spread = max(min_spread, 2 × realised_vol × mid_price).
• Inventory skew: when long, shift both quotes down to attract sellers;
  when short, shift them up to attract buyers.
• Quote size shrinks as inventory approaches the position limit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
import pandas as pd

from .base import BaseStrategy, OrderSpec
from order_book import Side, OrderType

if TYPE_CHECKING:
    from order_book import LimitOrderBook


class MarketMakerStrategy(BaseStrategy):
    """
    Passive liquidity provider quoting both bid and ask.

    Parameters
    ----------
    base_spread_bps : float
        Minimum half-spread in basis points (default 5 bps each side).
    vol_window : int
        Look-back for realised volatility (bars).
    quote_size : int
        Nominal shares per quote (reduced near position limit).
    inventory_skew_factor : float
        How aggressively to skew quotes toward reducing inventory.
    """

    def __init__(
        self,
        trader_id: str = "MarketMaker",
        initial_capital: float = 100_000.0,
        max_position: int = 500,
        warmup: int = 20,
        base_spread_bps: float = 5.0,
        vol_window: int = 20,
        quote_size: int = 100,
        inventory_skew_factor: float = 0.3,
    ) -> None:
        super().__init__(trader_id, initial_capital, max_position, warmup)
        self.base_spread_bps = base_spread_bps
        self.vol_window = vol_window
        self.quote_size = quote_size
        self.inventory_skew_factor = inventory_skew_factor
        self._active_bid_ids: List[int] = []
        self._active_ask_ids: List[int] = []

    def generate_orders(
        self,
        prices: pd.DataFrame,
        order_book: "LimitOrderBook",
        step: int,
    ) -> List[OrderSpec]:
        if step < self.warmup:
            return []

        # ── cancel all resting quotes ──────────────────────────────
        order_book.cancel_all_orders(self.trader_id)

        mid = prices["close"].iloc[step]

        # ── estimate realised volatility ───────────────────────────
        lookback = min(self.vol_window, step)
        recent_rets = prices["log_returns"].iloc[step - lookback + 1: step + 1].dropna()
        rv = recent_rets.std() if len(recent_rets) > 2 else 0.002

        # ── compute half-spread ────────────────────────────────────
        # Multiplier 0.5 keeps the MM tighter than background depth (30 bps)
        # so the MM gets filled before background orders do.
        min_half = self.base_spread_bps / 10_000 * mid
        vol_half = 0.5 * rv * mid
        half_spread = max(min_half, vol_half)
        half_spread = round(half_spread, 2)

        # ── inventory skew (shift both quotes to unwind inventory) ──
        inventory_ratio = self.position / self.max_position  # in [-1, 1]
        skew = self.inventory_skew_factor * inventory_ratio * half_spread

        bid_price = round(mid - half_spread - skew, 2)
        ask_price = round(mid + half_spread - skew, 2)

        if bid_price <= 0 or ask_price <= bid_price:
            return []

        # ── quote size – shrink near limit ─────────────────────────
        inv_util = abs(self.position) / self.max_position
        size_scale = max(0.1, 1.0 - inv_util ** 2)
        qty = max(10, int(self.quote_size * size_scale))

        # Avoid crossing the limit
        bid_qty = min(qty, max(0, self.max_position - int(self.position)))
        ask_qty = min(qty, max(0, self.max_position + int(self.position)))

        specs: List[OrderSpec] = []
        if bid_qty > 0:
            specs.append(OrderSpec(Side.BID, OrderType.LIMIT, bid_qty, bid_price))
        if ask_qty > 0:
            specs.append(OrderSpec(Side.ASK, OrderType.LIMIT, ask_qty, ask_price))
        return specs
