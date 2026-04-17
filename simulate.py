#!/usr/bin/env python3
"""
simulate.py  –  Single entry-point for the AAPL algorithmic trading simulator.

Run with:
    python3 simulate.py

What it does
────────────
1. Installs all required Python packages automatically.
2. Fetches real AAPL 5-minute bars via yfinance (falls back to synthetic data).
3. Instantiates four competing strategies in a shared limit order book.
4. Runs the full agent-based simulation bar-by-bar.
5. Computes full performance metrics for every strategy.
6. Saves four PNG visualisations to ./output/.
7. Generates a research-style PDF report to ./output/simulation_report.pdf.
"""

import argparse
import os
import subprocess
import sys

# ── Auto-install dependencies ─────────────────────────────────────────────────

_REQUIRED = ["yfinance", "pandas", "numpy", "matplotlib", "scipy", "reportlab", "plotly"]


def _ensure_packages() -> None:
    missing = []
    for pkg in _REQUIRED:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[Setup] Installing missing packages: {', '.join(missing)} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--break-system-packages",
             "-q", "--", *missing],
            stderr=subprocess.DEVNULL,
        )


_ensure_packages()

# ── Project root on path ──────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Standard imports (after dependency check) ─────────────────────────────────

import time

import numpy as np
import pandas as pd

from data_fetcher import fetch_aapl_data
from simulation import TradingSimulation
from metrics import all_metrics
from visualization import generate_all
from report import generate_report
from dashboard import generate_dashboard
from strategies import (
    MarketMakerStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    NoiseTraderStrategy,
    TrendFollowerStrategy,
)

# ── Configuration ─────────────────────────────────────────────────────────────

INITIAL_CAPITAL = 100_000.0   # $ per strategy
IMPACT_ETA      = 0.08        # market impact coefficient η


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_banner() -> None:
    print("=" * 65)
    print("  AAPL Algorithmic Trading Simulator")
    print("  Multi-Agent Order Book  ·  5 Competing Strategies")
    print("=" * 65)


def print_metrics(metrics_df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("  PERFORMANCE METRICS")
    print("=" * 65)

    display_cols = [
        "Total PnL ($)", "Ann. Return (%)", "Sharpe Ratio",
        "Sortino Ratio", "Max Drawdown (%)", "Calmar Ratio",
        "VaR 95% ($)", "CVaR 95% ($)", "Win Rate (%)", "Num Trades",
    ]
    available = [c for c in display_cols if c in metrics_df.columns]

    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 130)
    print(metrics_df[available].to_string())
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AAPL Algorithmic Trading Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        default=False,
        help=(
            "Run walk-forward out-of-sample backtesting after the main simulation. "
            "Slides a 30-day IS window in 5-day OOS steps, evaluates all 5 strategies "
            "on each OOS window, and embeds a Walk-Forward Analysis section into the PDF."
        ),
    )
    args = parser.parse_args()

    t_start = time.time()
    print_banner()

    # ── 1. Fetch data ─────────────────────────────────────────────────────
    data = fetch_aapl_data(period="60d", interval="5m")
    n_bars = len(data)

    try:
        date_range = (
            f"{data.index[0].strftime('%Y-%m-%d')} "
            f"→ {data.index[-1].strftime('%Y-%m-%d')}"
        )
    except Exception:
        date_range = f"{n_bars} bars"

    data_info = {
        "n_bars":     n_bars,
        "date_range": date_range,
        "ticker":     "AAPL",
        "interval":   "5m",
    }

    # ── 2. Create strategies ──────────────────────────────────────────────
    print(f"\n[Setup] Initialising strategies with ${INITIAL_CAPITAL:,.0f} each...\n")

    strategies = [
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

    for s in strategies:
        print(f"   {s.trader_id:<20} capital=${s.initial_capital:,.0f}  "
              f"max_pos=±{s.max_position}")

    # ── 3. Run simulation ─────────────────────────────────────────────────
    sim = TradingSimulation(data, strategies, impact_eta=IMPACT_ETA)
    sim.run()

    # ── 4. Print metrics ──────────────────────────────────────────────────
    metrics_df = all_metrics(strategies)
    print_metrics(metrics_df)

    summary = sim.strategy_summary()
    print("Strategy Summary:")
    print(summary.to_string(index=False))
    print()

    # ── 5. Generate visualisations ────────────────────────────────────────
    chart_paths = generate_all(
        strategies=strategies,
        price_history=sim.price_history,
        depth_snapshot=sim.depth_snapshot,
    )

    # ── 6. Walk-forward analysis (optional) ───────────────────────────────────
    wf_data = None
    if args.walk_forward:
        from walk_forward import run as wf_run
        print("\n[WF] Starting walk-forward backtesting...")
        wf_windows, wf_summary, wf_chart_paths = wf_run(data=data, verbose=True)
        wf_data = {
            "windows":     wf_windows,
            "summary_df":  wf_summary,
            "chart_paths": wf_chart_paths,
        }

    # ── 7. Generate PDF report ─────────────────────────────────────────────
    print("\n[Report] Building PDF report...")
    report_path = generate_report(
        strategies=strategies,
        chart_paths=chart_paths,
        price_history=sim.price_history,
        data_info=data_info,
        wf_data=wf_data,
    )

    # ── 8. Generate interactive HTML dashboard ─────────────────────────────
    print("\n[Dashboard] Building interactive HTML dashboard...")
    dashboard_path = generate_dashboard(
        strategies=strategies,
        price_history=sim.price_history,
        data=data,
    )

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print(f"  Simulation complete in {elapsed:.1f}s")
    print(f"  Charts    → {os.path.join(ROOT, 'output')}/")
    print(f"  Report    → {report_path}")
    if dashboard_path:
        print(f"  Dashboard → {dashboard_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
