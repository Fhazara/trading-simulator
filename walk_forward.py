"""
walk_forward.py
Walk-forward out-of-sample backtesting for the AAPL trading simulator.

Methodology
───────────
A 30-trading-day in-sample (IS) window slides forward in 5-trading-day OOS
steps across the full dataset.  For each window:

  1. Fresh strategy instances are created (identical parameters to simulate.py).
  2. TradingSimulation runs over the combined IS+OOS slice.
  3. Performance metrics are extracted from the OOS portion of each strategy's
     value_history only — the IS period purely warms up indicators; no metric
     numbers from it leak into the evaluation.

This eliminates lookahead bias: at the moment metrics are measured, strategies
have only ever "seen" bars up to and including the OOS window's own last bar.

Outputs
───────
• Prints a per-strategy summary table (mean OOS PnL, std, % profitable, worst).
• Saves three PNGs to output/:
    walk_forward_pnl.png          — stacked-bar OOS PnL + cumulative line
    walk_forward_sharpe.png       — rolling OOS Sharpe per strategy
    walk_forward_consistency.png  — green/red win-loss heatmap
• Returns structured data for embedding in the PDF report.

Usage
─────
    python3 walk_forward.py              # standalone
    python3 simulate.py --walk-forward   # integrated with full run
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from typing import Dict, List, Optional, Tuple

# ── Path setup (supports both standalone and imported use) ────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from data_fetcher import fetch_aapl_data
from simulation import TradingSimulation
from strategies import (
    MarketMakerStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    NoiseTraderStrategy,
    TrendFollowerStrategy,
)

# ── Constants ─────────────────────────────────────────────────────────────────
BARS_PER_DAY    = 78       # 5-min bars in one trading day
IS_DAYS         = 30       # in-sample window (trading days)
OOS_DAYS        = 5        # out-of-sample step (trading days)
IS_BARS         = IS_DAYS  * BARS_PER_DAY    # 2 340
OOS_BARS        = OOS_DAYS * BARS_PER_DAY    # 390
BARS_PER_YEAR   = 252 * BARS_PER_DAY         # annualisation factor

INITIAL_CAPITAL = 100_000.0
IMPACT_ETA      = 0.08

# Ordered list – determines stacking / chart order
STRATEGY_NAMES = [
    "MarketMaker",
    "Momentum",
    "MeanReversion",
    "NoiseTrader",
    "TrendFollower",
]

PALETTE: Dict[str, str] = {
    "MarketMaker":   "#4FC3F7",
    "Momentum":      "#81C784",
    "MeanReversion": "#FFB74D",
    "NoiseTrader":   "#E57373",
    "TrendFollower": "#CE93D8",
}

OUTPUT_DIR = os.path.join(ROOT, "output")


# ── Strategy factory ──────────────────────────────────────────────────────────

def _make_strategies() -> List:
    """
    Return fresh strategy instances using the same parameters as simulate.py.
    Called once per walk-forward window so each window starts with clean state.
    """
    return [
        MarketMakerStrategy(
            trader_id="MarketMaker",
            initial_capital=INITIAL_CAPITAL,
            max_position=500,
            warmup=20,
            base_spread_bps=5.0,
            vol_window=20,
            quote_size=100,
        ),
        MomentumStrategy(
            trader_id="Momentum",
            initial_capital=INITIAL_CAPITAL,
            max_position=500,
            warmup=35,
            fast_span=10,
            slow_span=30,
            threshold=0.0003,
            trade_size=150,
        ),
        MeanReversionStrategy(
            trader_id="MeanReversion",
            initial_capital=INITIAL_CAPITAL,
            max_position=400,
            warmup=35,
            lookback=30,
            entry_z=1.5,
            exit_z=0.4,
            trade_size=120,
        ),
        NoiseTraderStrategy(
            trader_id="NoiseTrader",
            initial_capital=INITIAL_CAPITAL,
            max_position=300,
            warmup=5,
            trade_prob=0.25,
            min_qty=20,
            max_qty=150,
        ),
        TrendFollowerStrategy(
            trader_id="TrendFollower",
            initial_capital=INITIAL_CAPITAL,
            max_position=600,
            warmup=25,
            channel_period=20,
            atr_period=14,
            atr_stop_mult=2.0,
            atr_tp_mult=3.0,
            risk_pct=0.01,
        ),
    ]


# ── OOS metric extraction ─────────────────────────────────────────────────────

def _compute_oos_metrics(strategy, oos_bars: int) -> Dict:
    """
    Extract performance metrics from the OOS portion of a strategy's history.

    After running IS_BARS + OOS_BARS total bars, value_history contains:
        [V₀, V₁, …, V_{IS}, V_{IS+1}, …, V_{IS+OOS}]
         ^initial        ^end-of-IS         ^end-of-OOS

    The OOS slice is value_history[IS_BARS:] which has oos_bars+1 entries.
    No IS-period values are included in any metric calculation.

    Returns a dict with keys: pnl, sharpe, max_drawdown, win_rate.
    """
    all_vals = np.array(strategy.value_history, dtype=float)

    # Slice to OOS portion (last oos_bars+1 entries)
    oos_vals = all_vals[-oos_bars - 1:]

    if len(oos_vals) < 3:
        return {"pnl": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}

    # PnL: end-of-OOS value minus start-of-OOS value
    pnl = float(oos_vals[-1] - oos_vals[0])

    # Bar-level returns
    rets = np.diff(oos_vals) / oos_vals[:-1]
    rets = rets[np.isfinite(rets)]

    if len(rets) < 2:
        return {"pnl": round(pnl, 2), "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}

    # Annualised Sharpe
    mean_r = np.mean(rets)
    std_r  = np.std(rets, ddof=1)
    sharpe = float(mean_r / std_r * np.sqrt(BARS_PER_YEAR)) if std_r > 1e-12 else 0.0

    # Maximum drawdown (%)
    peak   = np.maximum.accumulate(oos_vals)
    dd     = (oos_vals - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = float(np.min(dd) * 100)

    # Win rate: fraction of OOS bars with a positive return
    win_rate = float(np.mean(rets > 0) * 100)

    return {
        "pnl":          round(pnl, 2),
        "sharpe":       round(sharpe, 3),
        "max_drawdown": round(max_dd, 3),
        "win_rate":     round(win_rate, 2),
    }


# ── Walk-forward engine ───────────────────────────────────────────────────────

def run_walk_forward(
    data: pd.DataFrame,
    is_days:  int = IS_DAYS,
    oos_days: int = OOS_DAYS,
    verbose:  bool = True,
) -> List[Dict]:
    """
    Slide a fixed IS window forward in OOS-sized steps.

    For each window:
      • A fresh set of strategies is instantiated.
      • TradingSimulation runs over the IS+OOS slice (strategies warm up in IS,
        no metrics are captured there).
      • OOS metrics are read from the trailing oos_bars of value_history.

    Parameters
    ----------
    data     : Full OHLCV DataFrame from fetch_aapl_data().
    is_days  : In-sample window size in trading days (default 30).
    oos_days : Out-of-sample step in trading days (default 5).
    verbose  : If True, print progress; if False, suppress sub-simulation output.

    Returns
    -------
    List of window-result dicts, each with keys:
        window_idx, oos_label, oos_start, oos_end, results
    where results = {strategy_id: {pnl, sharpe, max_drawdown, win_rate}}.
    """
    n_bars   = len(data)
    is_bars  = is_days  * BARS_PER_DAY
    oos_bars = oos_days * BARS_PER_DAY

    # Adaptive scaling: if real data is shorter than expected, scale windows
    # proportionally so we always get at least 3 usable windows.
    min_windows = 3
    if n_bars < is_bars + oos_bars:
        is_bars  = int(n_bars * 0.60)
        oos_bars = int(n_bars * 0.10)
        print(f"[WF] Warning: only {n_bars} bars available. "
              f"Scaling to IS={is_bars} bars, OOS={oos_bars} bars per window.")

    windows   : List[Dict] = []
    window_idx = 0
    offset     = 0   # bar index where the IS window starts

    print(f"\n[WF] Walk-forward: IS={is_bars} bars ({is_days}d), "
          f"OOS={oos_bars} bars ({oos_days}d)")

    while offset + is_bars + oos_bars <= n_bars:
        is_end  = offset + is_bars
        oos_end = is_end + oos_bars

        # Build the combined IS+OOS slice; reset index so step numbering
        # starts at 0 (strategies and simulation rely on 0-based steps).
        window_data = data.iloc[offset:oos_end].copy()
        window_data = window_data.reset_index(drop=True)

        # Human-readable label for the OOS date range
        try:
            oos_idx = data.index[is_end:oos_end]
            label = (f"W{window_idx + 1}: "
                     f"{oos_idx[0].strftime('%m-%d')}"
                     f"→{oos_idx[-1].strftime('%m-%d')}")
        except Exception:
            label = f"W{window_idx + 1}"

        if verbose:
            print(f"\n[WF] Window {window_idx + 1}: "
                  f"IS=[{offset}:{is_end}), OOS=[{is_end}:{oos_end})  {label}")

        # Fresh strategies – identical parameters to simulate.py
        strategies = _make_strategies()

        # Run the simulation; suppress noisy per-bar output when not verbose
        sim = TradingSimulation(window_data, strategies, impact_eta=IMPACT_ETA)
        if verbose:
            sim.run()
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                sim.run()

        # Extract OOS-only metrics for each strategy
        window_results: Dict[str, Dict] = {}
        for strat in strategies:
            window_results[strat.trader_id] = _compute_oos_metrics(strat, oos_bars)

        windows.append({
            "window_idx": window_idx,
            "oos_label":  label,
            "oos_start":  is_end,
            "oos_end":    oos_end,
            "results":    window_results,
        })

        offset     += oos_bars
        window_idx += 1

    print(f"\n[WF] Completed {len(windows)} walk-forward windows.")
    return windows


# ── Summary table ─────────────────────────────────────────────────────────────

def print_wf_summary(windows: List[Dict]) -> pd.DataFrame:
    """
    Aggregate per-window OOS results and print a strategy-level summary.

    Columns
    -------
    Mean OOS PnL ($)       – average PnL across all OOS windows
    Std OOS PnL ($)        – standard deviation of window PnLs
    % Windows Profitable   – fraction of windows where PnL > 0
    Worst Window ($)       – lowest single-window PnL

    Returns the summary as a DataFrame (for embedding in the PDF report).
    """
    rows = []
    for name in STRATEGY_NAMES:
        pnls = [w["results"][name]["pnl"] for w in windows]

        mean_pnl  = float(np.mean(pnls))
        std_pnl   = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0
        pct_win   = float(np.mean([p > 0 for p in pnls]) * 100)
        worst_win = float(min(pnls))

        rows.append({
            "Strategy":              name,
            "Mean OOS PnL ($)":      round(mean_pnl,  2),
            "Std OOS PnL ($)":       round(std_pnl,   2),
            "% Windows Profitable":  round(pct_win,   1),
            "Worst Window ($)":      round(worst_win, 2),
        })

    summary_df = pd.DataFrame(rows).set_index("Strategy")

    # ── console output ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  WALK-FORWARD ANALYSIS – OUT-OF-SAMPLE SUMMARY")
    print("=" * 70)
    print(f"  IS window : {IS_DAYS} trading days  ({IS_BARS} bars)")
    print(f"  OOS step  : {OOS_DAYS} trading days  ({OOS_BARS} bars)")
    print(f"  Windows   : {len(windows)}")
    print("=" * 70)

    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.width", 100)
    print(summary_df.to_string())
    print()

    return summary_df


# ── Visualisations ────────────────────────────────────────────────────────────

def generate_wf_charts(windows: List[Dict]) -> Dict[str, str]:
    """
    Generate and save the three walk-forward visualisations.

    Returns
    -------
    Dict mapping chart keys to absolute file paths:
        walk_forward_pnl, walk_forward_sharpe, walk_forward_consistency
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    labels    = [w["oos_label"] for w in windows]
    n_windows = len(windows)
    x         = np.arange(n_windows)
    bar_width = 0.65

    paths: Dict[str, str] = {}

    # ─────────────────────────────────────────────────────────────────
    # 1.  walk_forward_pnl.png
    #     Stacked bar chart: each strategy's OOS PnL per window.
    #     Positive contributions stack upward from zero; negative downward.
    #     Overlaid dashed line = cumulative total PnL across all windows.
    # ─────────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(max(12, n_windows * 2), 6))
    fig.patch.set_facecolor("#0d1117")
    ax1.set_facecolor("#0d1117")

    bottoms_pos = np.zeros(n_windows)
    bottoms_neg = np.zeros(n_windows)

    legend_handles = []
    for name in STRATEGY_NAMES:
        pnls     = np.array([w["results"][name]["pnl"] for w in windows])
        pos_vals = np.where(pnls >= 0, pnls, 0.0)
        neg_vals = np.where(pnls <  0, pnls, 0.0)

        bar_pos = ax1.bar(
            x, pos_vals, bar_width,
            bottom=bottoms_pos,
            color=PALETTE[name], alpha=0.88, label=name,
        )
        ax1.bar(
            x, neg_vals, bar_width,
            bottom=bottoms_neg,
            color=PALETTE[name], alpha=0.88,
        )
        legend_handles.append(bar_pos)
        bottoms_pos = bottoms_pos + pos_vals
        bottoms_neg = bottoms_neg + neg_vals

    # Cumulative total PnL line on secondary y-axis
    cum_pnl = np.cumsum([
        sum(w["results"][n]["pnl"] for n in STRATEGY_NAMES)
        for w in windows
    ])
    ax2 = ax1.twinx()
    ax2.set_facecolor("#0d1117")
    (cum_line,) = ax2.plot(
        x, cum_pnl,
        color="white", linewidth=2.0, linestyle="--",
        marker="o", markersize=5, zorder=5,
        label="Cumulative Total PnL",
    )
    ax2.tick_params(colors="white", labelsize=8)
    ax2.set_ylabel("Cumulative Total PnL ($)", color="white", fontsize=10)
    ax2.spines["right"].set_color("#555")

    ax1.axhline(0, color="#666", linewidth=0.8, linestyle=":")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8, color="white")
    ax1.tick_params(colors="white", labelsize=8)
    ax1.set_ylabel("OOS PnL per Strategy ($)", color="white", fontsize=11)
    ax1.set_xlabel("Out-of-Sample Window", color="white", fontsize=10)
    ax1.set_title(
        "Walk-Forward PnL — Out-of-Sample Windows",
        color="white", fontsize=13, fontweight="bold", pad=14,
    )
    for spine in ax1.spines.values():
        spine.set_color("#555")

    leg = ax1.legend(
        legend_handles + [cum_line],
        STRATEGY_NAMES + ["Cumulative Total PnL"],
        loc="upper left", fontsize=8, framealpha=0.35,
        labelcolor="white", facecolor="#1a1a2e", edgecolor="#555",
    )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "walk_forward_pnl.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    paths["walk_forward_pnl"] = path
    print(f"[WF] Saved {path}")

    # ─────────────────────────────────────────────────────────────────
    # 2.  walk_forward_sharpe.png
    #     Line chart: annualised OOS Sharpe ratio per strategy per window.
    # ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(11, n_windows * 2), 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    for name in STRATEGY_NAMES:
        sharpes = [w["results"][name]["sharpe"] for w in windows]
        ax.plot(
            x, sharpes,
            color=PALETTE[name], linewidth=2.0,
            marker="o", markersize=6, label=name,
        )

    ax.axhline(0, color="#666", linewidth=0.8, linestyle=":")
    ax.axhline(1, color="#888", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8, color="white")
    ax.tick_params(colors="white", labelsize=8)
    ax.set_ylabel("Annualised OOS Sharpe Ratio", color="white", fontsize=11)
    ax.set_xlabel("Out-of-Sample Window", color="white", fontsize=10)
    ax.set_title(
        "Walk-Forward Rolling Out-of-Sample Sharpe Ratio",
        color="white", fontsize=13, fontweight="bold", pad=14,
    )
    for spine in ax.spines.values():
        spine.set_color("#555")

    ax.legend(
        loc="upper right", fontsize=9, framealpha=0.35,
        labelcolor="white", facecolor="#1a1a2e", edgecolor="#555",
    )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "walk_forward_sharpe.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    paths["walk_forward_sharpe"] = path
    print(f"[WF] Saved {path}")

    # ─────────────────────────────────────────────────────────────────
    # 3.  walk_forward_consistency.png
    #     Heatmap: strategies × windows; green = profitable, red = loss.
    #     Cells annotated with the actual OOS PnL dollar value.
    # ─────────────────────────────────────────────────────────────────
    n_strats = len(STRATEGY_NAMES)
    matrix   = np.zeros((n_strats, n_windows), dtype=float)
    for j, w in enumerate(windows):
        for i, name in enumerate(STRATEGY_NAMES):
            matrix[i, j] = 1.0 if w["results"][name]["pnl"] > 0 else -1.0

    fig_w = max(9, n_windows * 1.6)
    fig_h = max(3.5, n_strats * 0.7 + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "wf_consistency", ["#C62828", "#2E7D32"], N=256
    )
    im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    # Annotate every cell with sign + dollar PnL
    for i, name in enumerate(STRATEGY_NAMES):
        for j, w in enumerate(windows):
            pnl  = w["results"][name]["pnl"]
            sign = "+" if pnl >= 0 else ""
            ax.text(
                j, i, f"{sign}{pnl:,.0f}",
                ha="center", va="center",
                fontsize=7, color="white", fontweight="bold",
            )

    ax.set_xticks(range(n_windows))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8, color="white")
    ax.set_yticks(range(n_strats))
    ax.set_yticklabels(STRATEGY_NAMES, fontsize=9, color="white")
    ax.tick_params(colors="white")
    ax.set_xlabel("Out-of-Sample Window", color="white", fontsize=10)
    ax.set_title(
        "Walk-Forward Consistency — Win (green) / Loss (red) per Window",
        color="white", fontsize=12, fontweight="bold", pad=10,
    )
    for spine in ax.spines.values():
        spine.set_color("#555")

    cbar = fig.colorbar(im, ax=ax, ticks=[-1, 1], fraction=0.025, pad=0.02)
    cbar.ax.set_yticklabels(["Loss", "Win"], color="white", fontsize=9)
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("#888")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "walk_forward_consistency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    paths["walk_forward_consistency"] = path
    print(f"[WF] Saved {path}")

    return paths


# ── Public entry point ────────────────────────────────────────────────────────

def run(
    data: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> Tuple[List[Dict], pd.DataFrame, Dict[str, str]]:
    """
    Run the full walk-forward pipeline.

    Parameters
    ----------
    data    : Pre-fetched OHLCV DataFrame.  If None, fetch_aapl_data() is called.
    verbose : Passed through to run_walk_forward().

    Returns
    -------
    (windows, summary_df, chart_paths)
    """
    if data is None:
        data = fetch_aapl_data(period="60d", interval="5m")

    windows    = run_walk_forward(data, verbose=verbose)
    summary_df = print_wf_summary(windows)
    chart_paths = generate_wf_charts(windows)

    return windows, summary_df, chart_paths


# ── Standalone CLI ────────────────────────────────────────────────────────────

def main() -> None:
    run(verbose=True)


if __name__ == "__main__":
    main()
