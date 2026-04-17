"""
strategies/momentum.py
Trend-following momentum strategy based on EMA crossover.

Logic
─────
• Compute a fast EMA (short window) and a slow EMA (long window).
• Long signal  : fast_ema > slow_ema × (1 + threshold)
• Short signal : fast_ema < slow_ema × (1 − threshold)
• Neutral      : otherwise (close out position)

Regime gate (NEW)
─────────────────
• In TRENDING_UP   – only long entries are permitted.
• In TRENDING_DOWN – only short entries are permitted.
• In RANGING       – no new entries (EMA crossovers in choppy markets produce
                     whipsaws); existing position is closed on neutral signal.
• In UNKNOWN       – original behaviour (warming up).

This prevents the strategy from going long into a sustained downtrend (the
$280 → $244 AAPL move that caused −$10k losses in the original version).

All resting orders are cancelled-and-refreshed every bar to prevent stale
limits from accumulating on the same signal across multiple bars.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy, OrderSpec
from order_book import Side, OrderType
from regime import RegimeDetector, Regime

if TYPE_CHECKING:
    from order_book import LimitOrderBook


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


class MomentumStrategy(BaseStrategy):
    """
    EMA-crossover trend-follower with ADX regime gate.

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
    adx_period : int
        ADX period for the internal regime detector.
    trend_thresh : float
        ADX level above which a trend is confirmed (default 25).
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
        adx_period: int = 14,
        trend_thresh: float = 25.0,
    ) -> None:
        super().__init__(trader_id, initial_capital, max_position, warmup)
        self.fast_span     = fast_span
        self.slow_span     = slow_span
        self.threshold     = threshold
        self.trade_size    = trade_size
        self.stop_loss_pct = stop_loss_pct
        self._entry_price: Optional[float] = None
        self._regime_detector = RegimeDetector(
            adx_period=adx_period,
            trend_thresh=trend_thresh,
        )

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

        # ── regime gate ──────────────────────────────────────────────
        regime = self._regime_detector.detect(prices, step)

        # In RANGING regime, suppress new entries (crossovers whipsaw).
        # In TRENDING regimes, filter to the confirmed direction only.
        if regime == Regime.RANGING:
            long_signal  = False
            short_signal = False
        elif regime == Regime.TRENDING_UP:
            short_signal = False   # don't short into an uptrend
        elif regime == Regime.TRENDING_DOWN:
            long_signal  = False   # don't long into a downtrend
        # UNKNOWN: preserve both signals (warming up, use raw EMA)

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
