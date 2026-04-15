"""
market_impact.py
Square-root market impact model (Almgren–Chriss inspired).

The impact on fill price is:
    impact_per_share = eta * sigma * sqrt(Q / ADV) * S0

where
    eta   – dimensionless impact coefficient (default 0.1)
    sigma – intraday return volatility (from recent bars)
    Q     – order size in shares
    ADV   – average daily volume (shares)
    S0    – current price

For a BUY  the effective fill price is raised by impact_per_share.
For a SELL the effective fill price is lowered by impact_per_share.
"""

from __future__ import annotations

import numpy as np


class MarketImpactModel:
    """
    Square-root permanent market impact model.

    Parameters
    ----------
    eta : float
        Dimensionless market impact coefficient.  Higher → more slippage.
    adv_shares : int
        Assumed average daily volume in shares (used when live ADV is
        unavailable).  Typical AAPL ADV ≈ 55 million shares.
    """

    def __init__(self, eta: float = 0.08, adv_shares: int = 55_000_000) -> None:
        self.eta = eta
        self.adv_shares = adv_shares
        # We accumulate recent bar volumes for a live ADV estimate
        self._volume_window: list[float] = []
        self._window_size: int = 78  # 78 × 5-min bars ≈ 1 trading day

    def update(self, bar_volume: float) -> None:
        """Call once per bar to maintain a rolling ADV estimate."""
        self._volume_window.append(bar_volume)
        if len(self._volume_window) > self._window_size:
            self._volume_window.pop(0)

    def adv(self) -> float:
        """Current ADV estimate (shares per day)."""
        if not self._volume_window:
            return self.adv_shares
        # Scale bar average to full trading day (78 bars of 5 min each)
        bars_per_day = 78
        return np.mean(self._volume_window) * bars_per_day

    def impact_per_share(
        self,
        quantity: float,
        current_price: float,
        sigma: float = 0.002,   # default 5-min bar vol (~20 % ann.)
    ) -> float:
        """
        Returns the price impact per share (always non-negative).
        The caller applies it in the correct direction (+/−).
        """
        if quantity <= 0 or current_price <= 0:
            return 0.0
        adv = self.adv()
        participation = min(quantity / adv, 1.0)   # clip at 100 % of ADV
        impact = self.eta * sigma * np.sqrt(participation) * current_price
        return max(impact, 0.0)

    def effective_buy_price(
        self, quoted_price: float, quantity: float, current_price: float,
        sigma: float = 0.002
    ) -> float:
        return quoted_price + self.impact_per_share(quantity, current_price, sigma)

    def effective_sell_price(
        self, quoted_price: float, quantity: float, current_price: float,
        sigma: float = 0.002
    ) -> float:
        return quoted_price - self.impact_per_share(quantity, current_price, sigma)
