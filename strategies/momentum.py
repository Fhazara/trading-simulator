"""
strategies/momentum.py
Trend-following momentum strategy based on EMA crossover.

Logic
─────
• Compute a fast EMA (short window) and a slow EMA (long window).
• Long signal  : fast_ema > slow_ema × (1 + threshold)
• Short signal : fast_ema < slow_ema × (1 − threshold)
• Neutral      : otherwise (close out position)
• Entries via marketable limit order (2 cents inside mid) to avoid
  being swept at unfavourable prices.
• Hard stop-loss at stop_loss_pct adverse excursion from average entry.

All resting orders are cancelled-and-refreshed every bar to prevent
stale limits from accumulating on the same signal across multiple bars.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy, OrderSpec
from order_book import Side, OrderType

if TYPE_CHECKING:
    from order_book import LimitOrderBook


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


class MomentumStrategy(BaseStrategy):
    """
    EMA-crossover trend-follower.

    Parameters
    ----------
    fast_span, slow_span : int
        EMA windows.
    threshold : float
        Crossover significance threshold (fractional, e.g. 0.0005).
    trade_size : int
        Shares per entry signal.
    stop_loss_pct : float
        Close position if unrealised loss exceeds this fraction.
    """

    def __init__(
        self,
        trader_id: str = "Momentum",
        initial_capital: float = 100_000.0,
        max_position: int = 500,
        warmup: int = 35,
        fast_span: int = 10,
        slow_span: int = 30,
        threshold: float = 0.0003,
        trade_size: int = 150,
        stop_loss_pct: float = 0.025,
    ) -> None:
        super().__init__(trader_id, initial_capital, max_position, warmup)
        self.fast_span     = fast_span
        self.slow_span     = slow_span
        self.threshold     = threshold
        self.trade_size    = trade_size
        self.stop_loss_pct = stop_loss_pct
        self._entry_price: Optional[float] = None

    def generate_orders(
        self,
        prices: pd.DataFrame,
        order_book: "LimitOrderBook",
        step: int,
    ) -> List[OrderSpec]:
        # Always cancel-and-refresh: prevents stale limit accumulation on a
        # persistent signal across multiple bars.
        order_book.cancel_all_orders(self.trader_id)

        if step < max(self.warmup, self.slow_span + 5):
            return []

        close = prices["close"].iloc[: step + 1]
        mid   = float(close.iloc[-1])

        fast = float(_ema(close, self.fast_span).iloc[-1])
        slow = float(_ema(close, self.slow_span).iloc[-1])

        long_signal  = fast > slow * (1.0 + self.threshold)
        short_signal = fast < slow * (1.0 - self.threshold)

        pos = int(round(self.position))

        # ── stop-loss ────────────────────────────────────────────────
        if self._entry_price is not None and pos != 0:
            if pos > 0:
                loss_pct = (self._entry_price - mid) / self._entry_price
            else:
                loss_pct = (mid - self._entry_price) / self._entry_price
            if loss_pct > self.stop_loss_pct:
                self._entry_price = None
                qty = abs(pos)
                if pos > 0:
                    return [OrderSpec(Side.ASK, OrderType.MARKET, qty)]
                else:
                    return [OrderSpec(Side.BID, OrderType.MARKET, qty)]

        specs: List[OrderSpec] = []

        if long_signal:
            if pos < 0:
                # Flip short → long: close shorts first, then enter
                specs.append(OrderSpec(Side.BID, OrderType.MARKET, abs(pos)))
                self._entry_price = None
            if pos <= 0:
                target = min(self.trade_size, self.max_position)
                if target > 0:
                    limit_px = round(mid + 0.02, 2)
                    specs.append(OrderSpec(Side.BID, OrderType.LIMIT, target, limit_px))
                    if self._entry_price is None:
                        self._entry_price = mid

        elif short_signal:
            if pos > 0:
                # Flip long → short: close longs first, then enter
                specs.append(OrderSpec(Side.ASK, OrderType.MARKET, pos))
                self._entry_price = None
            if pos >= 0:
                target = min(self.trade_size, self.max_position)
                if target > 0:
                    limit_px = round(mid - 0.02, 2)
                    specs.append(OrderSpec(Side.ASK, OrderType.LIMIT, target, limit_px))
                    if self._entry_price is None:
                        self._entry_price = mid

        else:
            # Neutral signal → close any open position with a market order
            if pos != 0:
                self._entry_price = None
                qty = abs(pos)
                if pos > 0:
                    specs.append(OrderSpec(Side.ASK, OrderType.MARKET, qty))
                else:
                    specs.append(OrderSpec(Side.BID, OrderType.MARKET, qty))

        return specs
