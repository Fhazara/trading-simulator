"""
metrics.py
Full performance metric suite for each strategy.

Metrics computed
────────────────
• Total PnL ($)
• Total Return (%)
• Annualised Return (%)
• Sharpe Ratio (annualised)
• Sortino Ratio (annualised)
• Maximum Drawdown (%)
• Calmar Ratio
• VaR 95 % (one-bar, dollar)
• CVaR 95 % (Expected Shortfall, dollar)
• Win Rate (%)
• Number of Trades
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy

# 5-min bars per trading year  (252 days × 78 bars/day)
BARS_PER_YEAR = 252 * 78


def _drawdown_series(values: np.ndarray) -> np.ndarray:
    """Return the drawdown at each bar (negative or zero)."""
    peak = np.maximum.accumulate(values)
    return (values - peak) / peak


def compute_metrics(strategy: BaseStrategy, bars_per_year: int = BARS_PER_YEAR) -> Dict:
    """
    Compute the full metric set for one strategy.

    Returns a dictionary suitable for display / export.
    """
    values = np.array(strategy.value_history, dtype=float)
    if len(values) < 3:
        return _empty_metrics(strategy)

    # ── returns ─────────────────────────────────────────────────────
    rets = np.diff(values) / values[:-1]
    rets = rets[np.isfinite(rets)]

    if len(rets) == 0:
        return _empty_metrics(strategy)

    # ── basic PnL ────────────────────────────────────────────────────
    total_pnl = values[-1] - values[0]
    total_return_pct = (values[-1] / values[0] - 1) * 100

    # ── annualised return ────────────────────────────────────────────
    n_bars = len(rets)
    ann_factor = bars_per_year / max(n_bars, 1)
    ann_return_pct = ((1 + total_return_pct / 100) ** ann_factor - 1) * 100

    # ── Sharpe ───────────────────────────────────────────────────────
    mean_r = np.mean(rets)
    std_r = np.std(rets, ddof=1)
    sharpe = (mean_r / std_r * np.sqrt(bars_per_year)) if std_r > 1e-12 else 0.0

    # ── Sortino ──────────────────────────────────────────────────────
    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else std_r
    sortino = (mean_r / downside_std * np.sqrt(bars_per_year)) if downside_std > 1e-12 else 0.0

    # ── Drawdown ─────────────────────────────────────────────────────
    dd_series = _drawdown_series(values)
    max_dd_pct = float(np.min(dd_series) * 100)

    # ── Calmar ───────────────────────────────────────────────────────
    calmar = (ann_return_pct / abs(max_dd_pct)) if max_dd_pct < -1e-8 else 0.0

    # ── VaR & CVaR (95 %, one-bar, dollar) ───────────────────────────
    dollar_rets = np.diff(values)
    dollar_rets = dollar_rets[np.isfinite(dollar_rets)]
    var_95 = float(np.percentile(dollar_rets, 5))   # negative = loss
    cvar_95_arr = dollar_rets[dollar_rets <= var_95]
    cvar_95 = float(np.mean(cvar_95_arr)) if len(cvar_95_arr) > 0 else var_95

    # ── Win rate ─────────────────────────────────────────────────────
    win_rate = float(np.mean(rets > 0) * 100)

    return {
        "Strategy":            strategy.trader_id,
        "Total PnL ($)":       round(total_pnl, 2),
        "Total Return (%)":    round(total_return_pct, 3),
        "Ann. Return (%)":     round(ann_return_pct, 3),
        "Sharpe Ratio":        round(sharpe, 3),
        "Sortino Ratio":       round(sortino, 3),
        "Max Drawdown (%)":    round(max_dd_pct, 3),
        "Calmar Ratio":        round(calmar, 3),
        "VaR 95% ($)":         round(var_95, 2),
        "CVaR 95% ($)":        round(cvar_95, 2),
        "Win Rate (%)":        round(win_rate, 2),
        "Num Trades":          strategy.trade_count,
    }


def _empty_metrics(strategy: BaseStrategy) -> Dict:
    return {
        "Strategy":            strategy.trader_id,
        "Total PnL ($)":       0.0,
        "Total Return (%)":    0.0,
        "Ann. Return (%)":     0.0,
        "Sharpe Ratio":        0.0,
        "Sortino Ratio":       0.0,
        "Max Drawdown (%)":    0.0,
        "Calmar Ratio":        0.0,
        "VaR 95% ($)":         0.0,
        "CVaR 95% ($)":        0.0,
        "Win Rate (%)":        0.0,
        "Num Trades":          0,
    }


def all_metrics(strategies: List[BaseStrategy]) -> pd.DataFrame:
    """Return a DataFrame with one row per strategy."""
    rows = [compute_metrics(s) for s in strategies]
    return pd.DataFrame(rows).set_index("Strategy")


def drawdown_series_dict(strategies: List[BaseStrategy]) -> Dict[str, np.ndarray]:
    """Return {strategy_name: drawdown_array} for plotting."""
    result = {}
    for s in strategies:
        vals = np.array(s.value_history, dtype=float)
        if len(vals) > 1:
            result[s.trader_id] = _drawdown_series(vals) * 100
        else:
            result[s.trader_id] = np.array([0.0])
    return result
