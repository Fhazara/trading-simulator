"""
regime.py
Intraday market regime detection using ADX (Average Directional Index).

Regimes
───────
  TRENDING_UP   – sustained upward trend  (ADX > threshold, +DI > -DI)
  TRENDING_DOWN – sustained downward trend (ADX > threshold, -DI > +DI)
  RANGING       – low-trend-strength sideways market (ADX < threshold)
  UNKNOWN       – insufficient history for reliable classification

Method
──────
1. Compute True Range and Wilder-smoothed Directional Movements (+DM/-DM).
2. Derive +DI and -DI as percentages of smoothed True Range.
3. Compute DX and ADX via Wilder's smoothing.
4. Classify:  ADX >= trend_thresh → trending (direction from +DI vs -DI)
              ADX <  trend_thresh → ranging

Results are cached per (step, period) to avoid redundant computation when
multiple strategies call detect() at the same bar.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


class Regime(str, Enum):
    TRENDING_UP   = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING       = "RANGING"
    UNKNOWN       = "UNKNOWN"


# ── ADX computation ───────────────────────────────────────────────────────────

def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing: equivalent to EWM with alpha = 1/period."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """
    Compute ADX, +DI, and -DI for a price series.

    Parameters
    ----------
    high, low, close : pd.Series  – OHLCV columns (same index)
    period           : int        – Wilder smoothing period (default 14)

    Returns
    -------
    DataFrame with columns: adx, plus_di, minus_di
    """
    prev_close = close.shift(1)

    # True Range
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Raw directional movements
    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm_raw  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm_raw = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s  = pd.Series(plus_dm_raw,  index=close.index, dtype=float)
    minus_dm_s = pd.Series(minus_dm_raw, index=close.index, dtype=float)

    # Wilder-smoothed components
    tr_smooth       = _wilder_smooth(tr,        period)
    plus_dm_smooth  = _wilder_smooth(plus_dm_s, period)
    minus_dm_smooth = _wilder_smooth(minus_dm_s, period)

    # Directional indicators (percent)
    safe_tr = tr_smooth.replace(0.0, np.nan)
    plus_di  = 100.0 * plus_dm_smooth  / safe_tr
    minus_di = 100.0 * minus_dm_smooth / safe_tr

    # DX → ADX
    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx     = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx    = _wilder_smooth(dx, period)

    return pd.DataFrame(
        {"adx": adx, "plus_di": plus_di, "minus_di": minus_di},
        index=close.index,
    )


# ── Regime detector ───────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Classifies the current bar into a market regime using ADX.

    Parameters
    ----------
    adx_period    : int   – Wilder smoothing period for ADX (default 14)
    trend_thresh  : float – ADX level above which the market is "trending" (default 25)
    lookback_mult : int   – total lookback = adx_period × lookback_mult (default 5)

    Usage
    -----
    detector = RegimeDetector()
    regime   = detector.detect(prices_df, step)
    """

    def __init__(
        self,
        adx_period:   int   = 14,
        trend_thresh: float = 25.0,
        lookback_mult: int  = 5,
    ) -> None:
        self.adx_period    = adx_period
        self.trend_thresh  = trend_thresh
        self.min_bars      = adx_period * lookback_mult
        # Cache: step → Regime  (avoids recomputing for multiple strategies)
        self._cache: Dict[int, Regime] = {}

    def detect(self, prices: pd.DataFrame, step: int) -> Regime:
        """
        Detect the regime at *step* using price history up to and including
        that bar.  Result is cached so repeated calls at the same step are free.

        Parameters
        ----------
        prices : DataFrame with columns high, low, close (plus others)
        step   : current bar index (0-based)

        Returns
        -------
        Regime enum value
        """
        if step in self._cache:
            return self._cache[step]

        if step < self.min_bars:
            self._cache[step] = Regime.UNKNOWN
            return Regime.UNKNOWN

        end   = step + 1
        start = max(0, end - self.min_bars - self.adx_period * 2)

        high  = prices["high"].iloc[start:end]
        low   = prices["low"].iloc[start:end]
        close = prices["close"].iloc[start:end]

        adx_df = compute_adx(high, low, close, self.adx_period)

        adx_val      = float(adx_df["adx"].iloc[-1])
        plus_di_val  = float(adx_df["plus_di"].iloc[-1])
        minus_di_val = float(adx_df["minus_di"].iloc[-1])

        if not np.isfinite(adx_val):
            result = Regime.UNKNOWN
        elif adx_val < self.trend_thresh:
            result = Regime.RANGING
        elif plus_di_val >= minus_di_val:
            result = Regime.TRENDING_UP
        else:
            result = Regime.TRENDING_DOWN

        self._cache[step] = result
        return result

    def detect_series(self, prices: pd.DataFrame) -> pd.Series:
        """
        Compute regime for every bar in *prices* (useful for visualisation).
        Returns a pd.Series of Regime strings indexed like prices.
        """
        results = []
        for i in range(len(prices)):
            results.append(self.detect(prices, i))
        return pd.Series(results, index=prices.index, dtype=str)
