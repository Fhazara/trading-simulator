"""
visualization.py
Generates all simulation charts and saves them as PNGs in ./output/.

Charts produced
───────────────
1. order_book_depth.png   – Bid/ask depth snapshot
2. cumulative_pnl.png     – Each strategy's cumulative PnL over time
3. drawdown.png           – Per-strategy drawdown curves
4. metrics_table.png      – Formatted performance metrics table
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from metrics import all_metrics, drawdown_series_dict

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colour palette – one colour per strategy
PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

# ─────────────────────────────────────────────────────────────────────────────


def _savefig(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Viz] Saved {path}")
    return path


# ── 1. Order-book depth chart ─────────────────────────────────────────────────

def plot_order_book_depth(
    depth: Dict,
    mid_price: Optional[float] = None,
    filename: str = "order_book_depth.png",
) -> str:
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    bids = depth.get("bids", [])
    asks = depth.get("asks", [])

    if bids:
        bid_prices = [b[0] for b in bids]
        bid_cum    = [b[1] for b in bids]
        ax.fill_betweenx(bid_prices, 0, bid_cum, alpha=0.6, color="#4CAF50", label="Bids")
        ax.step(bid_cum, bid_prices, color="#4CAF50", linewidth=1.5, where="post")

    if asks:
        ask_prices = [a[0] for a in asks]
        ask_cum    = [a[1] for a in asks]
        ax.fill_betweenx(ask_prices, 0, ask_cum, alpha=0.6, color="#F44336", label="Asks")
        ax.step(ask_cum, ask_prices, color="#F44336", linewidth=1.5, where="post")

    if mid_price:
        ax.axhline(mid_price, color="white", linewidth=0.8, linestyle="--",
                   alpha=0.7, label=f"Mid ${mid_price:.2f}")

    ax.set_xlabel("Cumulative Volume (shares)", color="white")
    ax.set_ylabel("Price ($)", color="white")
    ax.set_title("Order Book Depth Snapshot (Mid-Simulation)", color="white",
                 fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
    ax.grid(axis="x", color="#333", linewidth=0.5)

    return _savefig(fig, filename)


# ── 2. Cumulative PnL curves ──────────────────────────────────────────────────

def plot_cumulative_pnl(
    strategies: List[BaseStrategy],
    price_history: List[float],
    filename: str = "cumulative_pnl.png",
) -> str:
    fig, (ax_pnl, ax_price) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor="#0d1117",
    )

    for ax in (ax_pnl, ax_price):
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    for i, strat in enumerate(strategies):
        pnl = np.array(strat.pnl_history, dtype=float)
        if len(pnl) == 0:
            continue
        color = PALETTE[i % len(PALETTE)]
        ax_pnl.plot(pnl, color=color, linewidth=1.4, label=strat.trader_id, alpha=0.9)

    ax_pnl.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax_pnl.set_ylabel("PnL ($)", color="white")
    ax_pnl.set_title("Cumulative PnL – All Strategies", color="white",
                      fontsize=14, fontweight="bold")
    ax_pnl.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white",
                   loc="upper left")
    ax_pnl.grid(color="#222", linewidth=0.5)

    # AAPL reference price on lower panel
    if price_history:
        ax_price.plot(price_history, color="#90CAF9", linewidth=1.0, alpha=0.9)
        ax_price.set_ylabel("AAPL ($)", color="white")
        ax_price.set_xlabel("Bar (5-min)", color="white")
        ax_price.grid(color="#222", linewidth=0.5)

    fig.tight_layout()
    return _savefig(fig, filename)


# ── 3. Drawdown chart ─────────────────────────────────────────────────────────

def plot_drawdown(
    strategies: List[BaseStrategy],
    filename: str = "drawdown.png",
) -> str:
    dd_dict = drawdown_series_dict(strategies)

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    for i, strat in enumerate(strategies):
        dd = dd_dict.get(strat.trader_id, np.array([0.0]))
        color = PALETTE[i % len(PALETTE)]
        ax.fill_between(range(len(dd)), dd, 0, alpha=0.3, color=color)
        ax.plot(dd, color=color, linewidth=1.2, label=strat.trader_id)

    ax.axhline(0, color="#555", linewidth=0.5)
    ax.set_ylabel("Drawdown (%)", color="white")
    ax.set_xlabel("Bar (5-min)", color="white")
    ax.set_title("Strategy Drawdown Over Time", color="white",
                 fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
    ax.grid(color="#222", linewidth=0.5)

    return _savefig(fig, filename)


# ── 4. Metrics comparison table ───────────────────────────────────────────────

def plot_metrics_table(
    strategies: List[BaseStrategy],
    filename: str = "metrics_table.png",
) -> str:
    df = all_metrics(strategies).T  # metrics as rows, strategies as columns
    df = df.reset_index()
    df.columns = ["Metric"] + list(df.columns[1:])

    n_rows = len(df)
    n_cols = len(df.columns)
    fig_w = max(10, n_cols * 2.2)
    fig_h = max(5, n_rows * 0.55 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.axis("off")

    # Colour cells by strategy
    cell_colors: List[List] = []
    header_bg = "#1a1a2e"
    alt_row_bg = "#111827"
    row_bg = "#0d1117"

    header_row = ["#1a1a2e"] * n_cols
    cell_colors.append(header_row)

    for r in range(n_rows):
        bg = alt_row_bg if r % 2 == 1 else row_bg
        row_colors = [bg] * n_cols
        # Highlight strategy-specific columns
        for c_idx in range(1, n_cols):
            # red/green tint for PnL row
            if "PnL" in str(df.iloc[r, 0]) or "Return" in str(df.iloc[r, 0]):
                try:
                    val = float(str(df.iloc[r, c_idx]).replace(",", ""))
                    row_colors[c_idx] = "#1b3a1b" if val >= 0 else "#3a1b1b"
                except Exception:
                    pass
        cell_colors.append(row_colors)

    col_labels = list(df.columns)
    cell_text = [col_labels] + df.values.tolist()

    table = ax.table(
        cellText=cell_text[1:],
        colLabels=cell_text[0],
        cellLoc="center",
        loc="center",
        cellColours=cell_colors[1:],
        colColours=cell_colors[0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style all cells
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#333")
        if row == 0:
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 0:
            cell.set_text_props(color="#90CAF9", fontweight="bold")
        else:
            cell.set_text_props(color="white")

    ax.set_title("Strategy Performance Metrics", color="white",
                 fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    return _savefig(fig, filename)


# ── master call ───────────────────────────────────────────────────────────────

def generate_all(
    strategies: List[BaseStrategy],
    price_history: List[float],
    depth_snapshot: Optional[Dict],
) -> Dict[str, str]:
    """
    Generate all charts. Returns dict of {chart_name: file_path}.
    """
    paths: Dict[str, str] = {}

    print("\n[Viz] Generating charts...")

    # 1. Order book depth
    if depth_snapshot:
        mid = (price_history[len(price_history) // 2]
               if price_history else None)
        paths["order_book_depth"] = plot_order_book_depth(
            depth_snapshot, mid_price=mid
        )

    # 2. Cumulative PnL
    paths["cumulative_pnl"] = plot_cumulative_pnl(strategies, price_history)

    # 3. Drawdown
    paths["drawdown"] = plot_drawdown(strategies)

    # 4. Metrics table
    paths["metrics_table"] = plot_metrics_table(strategies)

    return paths
