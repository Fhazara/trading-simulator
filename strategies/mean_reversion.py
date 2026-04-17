"""
strategies/mean_reversion.py
Statistical mean-reversion strategy using rolling Z-score.

Logic
─────
• Compute rolling mean and std over `lookback` bars.
• Z-score = (price − mean) / std
• FLAT → LONG   when z ≤ −entry_z  (price unusually cheap → buy)
• FLAT → SHORT  when z ≥ +entry_z  (price unusually expensive → sell)
• LONG  → exit  when |z| ≤ exit_z  OR  z ≥ entry_z (overshoot / stop)
• SHORT → exit  when |z| ≤ exit_z  OR  z ≤ −entry_z (overshoot / stop)

Regime gate (NEW)
─────────────────
• In RANGING       – normal operation (mean reversion is expected to work).
• In TRENDING_UP   – skip new entries; close existing positions on neutral Z.
                     Dips in uptrends continue lower; shorts are lethal.
• In TRENDING_DOWN – skip new entries; close existing positions on neutral Z.
                     Rallies in downtrends continue lower; longs are lethal.
• In UNKNOWN       – original behaviour (warming up).

This prevents the strategy from "buying dips that didn't revert" during the
$280 → $244 AAPL downtrend.

State machine uses the actual position (not a flag) so it cannot diverge from
the order book's accounting even when limit orders fill across multiple bars
or market orders partially execute.

Orders are cancelled-and-refreshed every bar so stale quotes are never left
to fill at out-of-date prices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
import pandas as pd

from .base import BaseStrategy, OrderSpec
from order_book import Side, OrderType
from regime import RegimeDetector, Regime

if TYPE_CHECKING:
    from order_book import LimitOrderBook


class MeanReversionStrategy(BaseStrategy):
    """
    Bollinger-band mean-reversion trader with ADX regime gate.

    Parameters
    ----------
    lookback : int
        Rolling window for mean / std calculation.
    entry_z : float
        Z-score threshold to enter a position.
    exit_z : float
        Z-score threshold to exit (reversion signal).
    trade_size : int
        Shares per signal.
    adx_period : int
        ADX period for the internal regime detector.
    trend_thresh : float
        ADX level above which a trend is confirmed (default 25).
    """

    def __init__(
        self,
        trader_id: str = "MeanReversion",
        initial_capital: float = 100_000.0,
        max_position: int = 400,
        warmup: int = 35,
        lookback: int = 30,
        entry_z: float = 1.5,
        exit_z: float = 0.4,
        trade_size: int = 120,
        adx_period: int = 14,
        trend_thresh: float = 25.0,
    ) -> None:
        super().__init__(trader_id, initial_capital, max_position, warmup)
        self.lookback    = lookback
        self.entry_z     = entry_z
        self.exit_z      = exit_z
        self.trade_size  = trade_size
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
        # Always cancel-and-refresh to prevent stale resting orders.
        order_book.cancel_all_orders(self.trader_id)

        if step < max(self.warmup, self.lookback + 2):
            return []

        close  = prices["close"].iloc[: step + 1]
        mid    = float(close.iloc[-1])
        window = close.iloc[-self.lookback :]
        mu     = float(window.mean())
        sigma  = float(window.std())

        if sigma < 1e-8:
            return []

        z = (mid - mu) / sigma

        # ── regime gate ──────────────────────────────────────────────
        regime = self._regime_detector.detect(prices, step)

        # In a trending regime, new mean-reversion entries are suppressed:
        # dips keep going lower in downtrends; rallies keep going higher in
        # uptrends.  We still honour exits so we can get out of any position
        # that was entered before the regime flipped.
        allow_new_entry = regime in (Regime.RANGING, Regime.UNKNOWN)

        specs: List[OrderSpec] = []
        pos = int(round(self.position))

        # ── exit logic (checked before entry to avoid same-bar flip) ──
        if pos > 0:
            # Holding long: close when reverted toward mean or overshot
            if abs(z) <= self.exit_z or z >= self.entry_z:
                specs.append(OrderSpec(Side.ASK, OrderType.MARKET, pos))
            # In a trending regime with an open long, also close if we're
            # not near entry z (i.e. we entered before regime flipped).
            elif not allow_new_entry:
                # Force close: don't hold longs in a downtrend
                specs.append(OrderSpec(Side.ASK, OrderType.MARKET, pos))

        elif pos < 0:
            # Holding short: close when reverted toward mean or undershot
            if abs(z) <= self.exit_z or z <= -self.entry_z:
                specs.append(OrderSpec(Side.BID, OrderType.MARKET, abs(pos)))
            elif not allow_new_entry:
                # Force close: don't hold shorts in an uptrend
                specs.append(OrderSpec(Side.BID, OrderType.MARKET, abs(pos)))

        elif allow_new_entry:
            # Flat + regime allows entry: look for fresh signal
            if z <= -self.entry_z:
                qty = min(self.trade_size, self.max_position)
                if qty > 0:
                    # Slightly aggressive buy limit
                    limit_px = round(mid + 0.02, 2)
                    specs.append(OrderSpec(Side.BID, OrderType.LIMIT, qty, limit_px))

            elif z >= self.entry_z:
                qty = min(self.trade_size, self.max_position)
                if qty > 0:
                    # Slightly aggressive sell limit
                    limit_px = round(mid - 0.02, 2)
                    specs.append(OrderSpec(Side.ASK, OrderType.LIMIT, qty, limit_px))

        return specs
