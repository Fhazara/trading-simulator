"""
strategies/trend_follower.py
Price-channel breakout trend-follower (Donchian-style) with ATR-based stops.

Logic
─────
• Compute the N-bar highest-high and lowest-low (the "channel").
• LONG  entry : close breaks above the channel high → buy.
• SHORT entry : close breaks below the channel low  → sell.
• ATR-based stop:  stop = entry_price ± atr_stop_mult × ATR(atr_period)
• ATR-based target: target = entry_price ± atr_tp_mult × ATR   (1.5 : 1 RR)
• Position sizing: risk a fixed fraction of capital per trade
  (size = risk_pct × capital / (atr_stop_mult × ATR))
  capped at max_position.

Differences from MomentumStrategy (EMA crossover)
──────────────────────────────────────────────────
• Signal is price/channel breakout (absolute price level), not smoothed EMAs.
• ATR-anchored stops and targets instead of a fixed stop_loss_pct.
• Dynamic position sizing based on current volatility.
• No regime filter needed: breakouts ARE the regime confirmation.

Performance expectation on downtrends
──────────────────────────────────────
A new 20-bar low is formed quickly in sustained downtrends, giving an early
short entry.  ATR-sized stops keep losses bounded if the breakout is false.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseStrategy, OrderSpec
from order_book import Side, OrderType

if TYPE_CHECKING:
    from order_book import LimitOrderBook


# ── ATR helper ────────────────────────────────────────────────────────────────

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> float:
    """
    Wilder-smoothed ATR for the last `period` bars.
    Returns scalar float; NaN-safe (returns small fallback on bad input).
    """
    if len(close) < period + 1:
        return float(close.iloc[-1]) * 0.002  # fallback: 0.2 % of price

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_val = tr.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
    return float(atr_val) if np.isfinite(atr_val) else float(close.iloc[-1]) * 0.002


class TrendFollowerStrategy(BaseStrategy):
    """
    Donchian-channel breakout strategy with ATR-scaled stops and targets.

    Parameters
    ----------
    channel_period : int
        Look-back bars for the high/low channel (default 20).
    atr_period : int
        Look-back for ATR calculation (default 14).
    atr_stop_mult : float
        Stop distance = atr_stop_mult × ATR (default 2.0).
    atr_tp_mult : float
        Take-profit distance = atr_tp_mult × ATR (default 3.0 → 1.5 : 1 RR).
    risk_pct : float
        Fraction of current capital to risk per trade (default 0.01 = 1 %).
    warmup : int
        Bars to skip before trading (needs channel_period + atr_period history).
    """

    def __init__(
        self,
        trader_id: str = "TrendFollower",
        initial_capital: float = 100_000.0,
        max_position: int = 600,
        warmup: int = 25,
        channel_period: int = 20,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        risk_pct: float = 0.01,
    ) -> None:
        super().__init__(trader_id, initial_capital, max_position, warmup)
        self.channel_period = channel_period
        self.atr_period     = atr_period
        self.atr_stop_mult  = atr_stop_mult
        self.atr_tp_mult    = atr_tp_mult
        self.risk_pct       = risk_pct

        # State tracking for an open position
        self._entry_price: Optional[float] = None
        self._stop_price:  Optional[float] = None
        self._tp_price:    Optional[float] = None
        self._direction:   int             = 0  # +1 long, -1 short, 0 flat

    # ── internal helpers ──────────────────────────────────────────────────────

    def _compute_size(self, capital: float, atr: float) -> int:
        """Position size that risks risk_pct of capital for atr_stop_mult × ATR."""
        stop_dist = self.atr_stop_mult * atr
        if stop_dist <= 0:
            return 10
        size = int((capital * self.risk_pct) / stop_dist)
        return max(10, min(size, self.max_position))

    def _close_position(self, pos: int) -> List[OrderSpec]:
        """Emit a market order to flatten the current position."""
        self._entry_price = None
        self._stop_price  = None
        self._tp_price    = None
        self._direction   = 0
        qty = abs(pos)
        if pos > 0:
            return [OrderSpec(Side.ASK, OrderType.MARKET, qty)]
        elif pos < 0:
            return [OrderSpec(Side.BID, OrderType.MARKET, qty)]
        return []

    # ── main logic ────────────────────────────────────────────────────────────

    def generate_orders(
        self,
        prices: pd.DataFrame,
        order_book: "LimitOrderBook",
        step: int,
    ) -> List[OrderSpec]:
        # Cancel-and-refresh all resting orders every bar
        order_book.cancel_all_orders(self.trader_id)

        min_warmup = max(self.warmup, self.channel_period + self.atr_period + 2)
        if step < min_warmup:
            return []

        # ── gather price data ──────────────────────────────────────────
        end   = step + 1
        close = prices["close"].iloc[:end]
        high  = prices["high"].iloc[:end]
        low   = prices["low"].iloc[:end]
        mid   = float(close.iloc[-1])

        # Previous bar's channel (exclude current bar for breakout logic)
        prev_high_n = float(high.iloc[-(self.channel_period + 1): -1].max())
        prev_low_n  = float(low.iloc[-(self.channel_period + 1): -1].min())

        atr = _atr(high, low, close, self.atr_period)

        pos = int(round(self.position))

        # ── check exits on open position ───────────────────────────────
        if pos != 0 and self._entry_price is not None:
            if pos > 0:
                # Long: hit stop or take-profit?
                if (self._stop_price is not None and mid <= self._stop_price) or \
                   (self._tp_price   is not None and mid >= self._tp_price):
                    return self._close_position(pos)
            else:
                # Short: hit stop or take-profit?
                if (self._stop_price is not None and mid >= self._stop_price) or \
                   (self._tp_price   is not None and mid <= self._tp_price):
                    return self._close_position(pos)

        # ── no new entry if already positioned ───────────────────────
        if pos != 0:
            return []

        # ── breakout signals ───────────────────────────────────────────
        # Long breakout: close rises above the previous N-bar high
        long_breakout  = mid > prev_high_n
        # Short breakout: close falls below the previous N-bar low
        short_breakout = mid < prev_low_n

        capital = self.portfolio_value(mid)
        size    = self._compute_size(capital, atr)

        if long_breakout:
            stop  = round(mid - self.atr_stop_mult * atr, 2)
            tp    = round(mid + self.atr_tp_mult   * atr, 2)
            entry_limit = round(mid + 0.02, 2)  # slightly aggressive
            self._entry_price = mid
            self._stop_price  = stop
            self._tp_price    = tp
            self._direction   = 1
            return [OrderSpec(Side.BID, OrderType.LIMIT, size, entry_limit)]

        elif short_breakout:
            stop  = round(mid + self.atr_stop_mult * atr, 2)
            tp    = round(mid - self.atr_tp_mult   * atr, 2)
            entry_limit = round(mid - 0.02, 2)  # slightly aggressive
            self._entry_price = mid
            self._stop_price  = stop
            self._tp_price    = tp
            self._direction   = -1
            return [OrderSpec(Side.ASK, OrderType.LIMIT, size, entry_limit)]

        return []
