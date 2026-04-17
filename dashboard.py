"""
dashboard.py
Interactive HTML dashboard generated with Plotly.

Layout (single-page, dark theme)
─────────────────────────────────
  Row 1 (60 %) – AAPL price + volume bars + regime colour bands
  Row 2 (40 %) – Cumulative PnL for all strategies
  Row 3 (30 %) – Drawdown curves
  Row 4 (40 %) – Performance metrics: grouped bar chart

All panels share a linked x-axis (bar index) and support hover, zoom, and
pan without any server.  Saved as a single self-contained HTML file.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.colors as pc
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False

from strategies.base import BaseStrategy
from metrics import all_metrics, drawdown_series_dict
from regime import RegimeDetector, Regime

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── colour constants ──────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
GRID_COL  = "#1e2130"
TEXT_COL  = "#c9d1d9"
ACCENT    = "#58a6ff"

STRAT_COLOURS = [
    "#2196F3",   # MarketMaker – blue
    "#F44336",   # Momentum    – red
    "#4CAF50",   # MeanRev     – green
    "#FF9800",   # NoiseTrader – orange
    "#CE93D8",   # TrendFollower – purple
]

REGIME_COLOURS = {
    Regime.TRENDING_UP:   "rgba(76,175,80,0.12)",    # green tint
    Regime.TRENDING_DOWN: "rgba(244,67,54,0.12)",    # red tint
    Regime.RANGING:       "rgba(33,150,243,0.07)",   # blue tint
    Regime.UNKNOWN:       "rgba(100,100,100,0.05)",
}

REGIME_LABELS = {
    Regime.TRENDING_UP:   "Trending ↑",
    Regime.TRENDING_DOWN: "Trending ↓",
    Regime.RANGING:       "Ranging",
    Regime.UNKNOWN:       "",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _dark_layout(fig: "go.Figure", **kwargs) -> "go.Figure":
    """Apply a uniform dark theme to a figure."""
    # Merge legend defaults with any caller-supplied overrides
    legend_defaults = dict(bgcolor="rgba(13,17,23,0.8)", bordercolor="#30363d", borderwidth=1)
    if "legend" in kwargs:
        legend_defaults.update(kwargs.pop("legend"))
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_COL, size=11),
        legend=legend_defaults,
        margin=dict(l=60, r=30, t=50, b=40),
        **kwargs,
    )
    fig.update_xaxes(
        gridcolor=GRID_COL, zerolinecolor=GRID_COL,
        tickfont=dict(color=TEXT_COL),
    )
    fig.update_yaxes(
        gridcolor=GRID_COL, zerolinecolor=GRID_COL,
        tickfont=dict(color=TEXT_COL),
    )
    return fig


def _regime_spans(
    regime_series: "pd.Series",
) -> List[Dict]:
    """
    Convert a series of Regime values to a list of contiguous spans
    [{regime, x0, x1}, ...] suitable for vrect overlays.
    """
    if len(regime_series) == 0:
        return []
    spans = []
    current = regime_series.iloc[0]
    x0 = 0
    for i, val in enumerate(regime_series):
        if val != current:
            spans.append({"regime": current, "x0": x0, "x1": i - 1})
            current = val
            x0 = i
    spans.append({"regime": current, "x0": x0, "x1": len(regime_series) - 1})
    return spans


# ── panel builders ────────────────────────────────────────────────────────────

def _add_regime_bands(
    fig: "go.Figure",
    regime_series: "pd.Series",
    row: int,
    col: int = 1,
    added_labels: Optional[set] = None,
) -> None:
    """Add shaded vrect background bands per regime to a subplot row."""
    if added_labels is None:
        added_labels = set()
    spans = _regime_spans(regime_series)
    for span in spans:
        r = span["regime"]
        label = REGIME_LABELS.get(r, "")
        show_legend = (label and label not in added_labels)
        if show_legend:
            added_labels.add(label)
        fig.add_vrect(
            x0=span["x0"], x1=span["x1"],
            fillcolor=REGIME_COLOURS.get(r, "rgba(0,0,0,0)"),
            opacity=1.0,
            layer="below",
            line_width=0,
            row=row, col=col,
            annotation_text="",
        )


def _build_price_panel(
    price_history: List[float],
    data: pd.DataFrame,
    regime_series: "pd.Series",
    fig: "go.Figure",
    row: int,
) -> None:
    """Price line + volume bars + regime background."""
    xs = list(range(len(price_history)))

    # Regime background bands
    _add_regime_bands(fig, regime_series, row=row)

    # AAPL price line
    fig.add_trace(
        go.Scatter(
            x=xs, y=price_history,
            mode="lines",
            name="AAPL",
            line=dict(color=ACCENT, width=1.5),
            hovertemplate="Bar %{x}<br>Price $%{y:.2f}<extra></extra>",
        ),
        row=row, col=1,
    )

    # Volume bars in a mini sub-panel (secondary y-axis)
    if "volume" in data.columns and len(data) >= len(price_history):
        vols = data["volume"].iloc[: len(price_history)].tolist()
        fig.add_trace(
            go.Bar(
                x=xs, y=vols,
                name="Volume",
                marker_color="rgba(88,166,255,0.25)",
                hovertemplate="Bar %{x}<br>Volume %{y:,.0f}<extra></extra>",
                yaxis="y2",
                showlegend=True,
            ),
            row=row, col=1,
        )

    # Regime legend entries (invisible scatter just for legend)
    for regime, colour in REGIME_COLOURS.items():
        label = REGIME_LABELS.get(regime, "")
        if not label:
            continue
        # Convert rgba to a solid version for legend swatch
        solid = colour.replace("rgba(", "rgb(").split(",")
        solid_col = ",".join(solid[:3]) + ")"
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                name=label,
                marker=dict(
                    size=12, symbol="square",
                    color=colour.replace("0.12", "0.6").replace("0.07", "0.5"),
                ),
                showlegend=True,
                legendgroup="regime",
            ),
            row=row, col=1,
        )


def _build_pnl_panel(
    strategies: List[BaseStrategy],
    fig: "go.Figure",
    row: int,
) -> None:
    """Cumulative PnL line for each strategy."""
    for i, strat in enumerate(strategies):
        pnl = np.array(strat.pnl_history, dtype=float)
        if len(pnl) == 0:
            continue
        colour = STRAT_COLOURS[i % len(STRAT_COLOURS)]
        final  = pnl[-1]
        sign   = "+" if final >= 0 else ""
        fig.add_trace(
            go.Scatter(
                x=list(range(len(pnl))),
                y=pnl.tolist(),
                mode="lines",
                name=f"{strat.trader_id} ({sign}${final:,.0f})",
                line=dict(color=colour, width=1.8),
                hovertemplate=(
                    f"<b>{strat.trader_id}</b><br>"
                    "Bar %{x}<br>PnL $%{y:,.2f}<extra></extra>"
                ),
            ),
            row=row, col=1,
        )
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#555", row=row, col=1)


def _hex_to_rgba(hex_color: str, alpha: float = 0.18) -> str:
    """Convert '#RRGGBB' to 'rgba(R,G,B,alpha)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _build_drawdown_panel(
    strategies: List[BaseStrategy],
    fig: "go.Figure",
    row: int,
) -> None:
    """Per-strategy drawdown curves."""
    dd_dict = drawdown_series_dict(strategies)
    for i, strat in enumerate(strategies):
        dd = dd_dict.get(strat.trader_id, np.array([0.0]))
        colour = STRAT_COLOURS[i % len(STRAT_COLOURS)]
        fill_colour = _hex_to_rgba(colour) if colour.startswith("#") else colour
        fig.add_trace(
            go.Scatter(
                x=list(range(len(dd))),
                y=dd.tolist(),
                mode="lines",
                name=strat.trader_id,
                line=dict(color=colour, width=1.4),
                fill="tozeroy",
                fillcolor=fill_colour,
                showlegend=False,
                hovertemplate=(
                    f"<b>{strat.trader_id}</b><br>"
                    "Bar %{x}<br>DD %{y:.2f}%<extra></extra>"
                ),
            ),
            row=row, col=1,
        )


def _build_metrics_panel(
    strategies: List[BaseStrategy],
    fig: "go.Figure",
    row: int,
) -> None:
    """Grouped bar chart comparing key metrics across strategies."""
    metrics_df = all_metrics(strategies)
    display_metrics = [
        "Total PnL ($)", "Sharpe Ratio", "Sortino Ratio",
        "Max Drawdown (%)", "Win Rate (%)",
    ]
    available = [m for m in display_metrics if m in metrics_df.columns]

    strat_names = metrics_df.index.tolist()

    for i, metric in enumerate(available):
        vals = metrics_df[metric].tolist()
        colours = [
            (STRAT_COLOURS[j % len(STRAT_COLOURS)])
            for j in range(len(strat_names))
        ]
        fig.add_trace(
            go.Bar(
                name=metric,
                x=strat_names,
                y=vals,
                marker_color=colours,
                text=[f"{v:.2f}" for v in vals],
                textposition="outside",
                hovertemplate=(
                    f"<b>%{{x}}</b><br>{metric}: %{{y:.3f}}<extra></extra>"
                ),
                visible=(i == 0),  # only first metric shown by default
            ),
            row=row, col=1,
        )

    # Dropdown to switch metric
    n_strats = len(strat_names)
    n_metrics = len(available)
    buttons = []
    for idx, metric in enumerate(available):
        visibility = [False] * n_metrics
        visibility[idx] = True
        # Each bar trace maps to one metric; previous traces (price/PnL/dd)
        # are not in this row so we don't touch them here — we address only
        # the metric bar traces by their relative visibility list.
        buttons.append(
            dict(
                label=metric,
                method="update",
                args=[
                    {"visible": [True, True, True, True, True,  # price+vol+regime
                                 True] * 1               # spacer for other rows
                                + [v for v in visibility]},
                ],
            )
        )


# ── master builder ────────────────────────────────────────────────────────────

def generate_dashboard(
    strategies: List[BaseStrategy],
    price_history: List[float],
    data: pd.DataFrame,
    output_path: Optional[str] = None,
    adx_period: int = 14,
    trend_thresh: float = 25.0,
) -> str:
    """
    Build and save the interactive HTML dashboard.

    Parameters
    ----------
    strategies    : list of strategies after simulation
    price_history : close prices recorded by simulation
    data          : original OHLCV DataFrame (for volume + regime computation)
    output_path   : where to save the HTML (default: output/dashboard.html)
    adx_period    : ADX period for regime overlay
    trend_thresh  : ADX threshold for trending classification

    Returns
    -------
    str – absolute path to the saved HTML file
    """
    if not _PLOTLY_OK:
        print("[Dashboard] plotly not installed – skipping dashboard generation.")
        return ""

    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "dashboard.html")

    print("[Dashboard] Computing regime history...")
    detector = RegimeDetector(adx_period=adx_period, trend_thresh=trend_thresh)
    n = len(price_history)
    regime_series = pd.Series(
        [detector.detect(data.iloc[:i + 1].reset_index(drop=True), i) for i in range(n)],
        dtype=str,
    )

    print("[Dashboard] Building interactive dashboard...")

    # ── subplot grid ──────────────────────────────────────────────────
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.32, 0.26, 0.20, 0.22],
        vertical_spacing=0.04,
        subplot_titles=[
            "AAPL Price  |  Regime: <span style='color:#4CAF50'>▮</span> Up  "
            "<span style='color:#F44336'>▮</span> Down  "
            "<span style='color:#2196F3'>▮</span> Ranging",
            "Cumulative PnL ($)",
            "Drawdown (%)",
            "Performance Metrics",
        ],
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
    )

    # ── populate panels ───────────────────────────────────────────────
    _build_price_panel(price_history, data, regime_series, fig, row=1)
    _build_pnl_panel(strategies, fig, row=2)
    _build_drawdown_panel(strategies, fig, row=3)
    _build_metrics_panel(strategies, fig, row=4)

    # ── axis labels ───────────────────────────────────────────────────
    fig.update_yaxes(title_text="Price ($)",    row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volume",       row=1, col=1, secondary_y=True,
                     showgrid=False)
    fig.update_yaxes(title_text="PnL ($)",      row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
    fig.update_xaxes(title_text="Bar (5-min)",  row=4, col=1)

    # Metric selector dropdown (updates which bar series is visible in row 4)
    metrics_df = all_metrics(strategies)
    display_metrics = [
        "Total PnL ($)", "Sharpe Ratio", "Sortino Ratio",
        "Max Drawdown (%)", "Win Rate (%)",
    ]
    available_metrics = [m for m in display_metrics if m in metrics_df.columns]

    # Count traces not in the metrics panel so we can offset visibility list
    n_price_traces = 2 + 4   # price line + volume bar + 4 regime legend entries
    n_pnl_traces   = len(strategies) + 1   # one per strategy + hline (invisible)
    n_dd_traces    = len(strategies)
    n_prefix       = n_price_traces + n_pnl_traces + n_dd_traces
    n_metric_bars  = len(available_metrics)

    buttons = []
    for idx, metric in enumerate(available_metrics):
        vis = [True] * n_prefix + [i == idx for i in range(n_metric_bars)]
        buttons.append(
            dict(label=metric, method="restyle",
                 args=[{"visible": vis}])
        )

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01, xanchor="left",
                y=0.06, yanchor="top",
                bgcolor="#1e2130",
                bordercolor="#30363d",
                font=dict(color=TEXT_COL),
            )
        ]
    )

    # ── global dark theme ─────────────────────────────────────────────
    n_strategies = len(strategies)
    strat_names  = [s.trader_id for s in strategies]

    title_text = (
        f"AAPL Algorithmic Trading — {n_strategies} Strategies  "
        f"({len(price_history):,} bars)"
    )

    _dark_layout(
        fig,
        title=dict(text=title_text, font=dict(size=15, color=TEXT_COL), x=0.01),
        height=980,
        hovermode="x unified",
        legend=dict(orientation="v", x=1.01, y=1),
    )

    # Subplot title colour
    for ann in fig.layout.annotations:
        ann.font.color = TEXT_COL
        ann.font.size  = 12

    # ── save ─────────────────────────────────────────────────────────
    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        },
    )
    print(f"[Dashboard] Saved {output_path}")
    return output_path
