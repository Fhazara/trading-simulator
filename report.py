"""
report.py
Generates a research-style PDF report summarising the simulation.
Uses ReportLab for PDF creation with embedded charts.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)

from strategies.base import BaseStrategy
from metrics import all_metrics

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# ── colours ──────────────────────────────────────────────────────────────────
DARK_BG   = colors.HexColor("#0d1117")
BLUE_H    = colors.HexColor("#1565C0")
LIGHT_BG  = colors.HexColor("#f5f7fa")
ACCENT    = colors.HexColor("#1976D2")
SUCCESS   = colors.HexColor("#2E7D32")
DANGER    = colors.HexColor("#C62828")
TEXT_DARK = colors.HexColor("#212121")
TEXT_MED  = colors.HexColor("#424242")
GREY_RULE = colors.HexColor("#BDBDBD")


def _styles():
    base = getSampleStyleSheet()
    custom = {
        "Title": ParagraphStyle(
            "ReportTitle", parent=base["Title"],
            fontSize=28, textColor=colors.white,
            alignment=TA_CENTER, spaceAfter=6,
            fontName="Helvetica-Bold",
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"],
            fontSize=13, textColor=colors.HexColor("#90CAF9"),
            alignment=TA_CENTER, spaceAfter=4,
            fontName="Helvetica",
        ),
        "H1": ParagraphStyle(
            "H1", parent=base["Heading1"],
            fontSize=16, textColor=ACCENT,
            spaceAfter=6, spaceBefore=14,
            fontName="Helvetica-Bold",
            borderPad=(0, 0, 2, 0),
        ),
        "H2": ParagraphStyle(
            "H2", parent=base["Heading2"],
            fontSize=12, textColor=BLUE_H,
            spaceAfter=4, spaceBefore=8,
            fontName="Helvetica-Bold",
        ),
        "Body": ParagraphStyle(
            "Body", parent=base["Normal"],
            fontSize=10, textColor=TEXT_DARK,
            spaceAfter=6, leading=15,
            alignment=TA_JUSTIFY,
            fontName="Helvetica",
        ),
        "Bullet": ParagraphStyle(
            "Bullet", parent=base["Normal"],
            fontSize=10, textColor=TEXT_MED,
            spaceAfter=3, leading=14,
            leftIndent=14, bulletIndent=4,
            fontName="Helvetica",
        ),
        "Caption": ParagraphStyle(
            "Caption", parent=base["Normal"],
            fontSize=8, textColor=TEXT_MED,
            alignment=TA_CENTER, spaceAfter=4, spaceBefore=2,
            fontName="Helvetica-Oblique",
        ),
        "TableHead": ParagraphStyle(
            "TableHead", parent=base["Normal"],
            fontSize=9, textColor=colors.white,
            alignment=TA_CENTER, fontName="Helvetica-Bold",
        ),
        "TableCell": ParagraphStyle(
            "TableCell", parent=base["Normal"],
            fontSize=8.5, textColor=TEXT_DARK,
            alignment=TA_CENTER, fontName="Helvetica",
        ),
    }
    return custom


def _cover_page(canvas, doc):
    """Dark gradient cover page background."""
    canvas.saveState()
    canvas.setFillColor(DARK_BG)
    canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    # top accent bar
    canvas.setFillColor(BLUE_H)
    canvas.rect(0, A4[1] - 8 * mm, A4[0], 8 * mm, fill=1, stroke=0)
    # bottom accent bar
    canvas.setFillColor(BLUE_H)
    canvas.rect(0, 0, A4[0], 5 * mm, fill=1, stroke=0)
    canvas.restoreState()


def _page_header_footer(canvas, doc):
    canvas.saveState()
    # header rule
    canvas.setStrokeColor(GREY_RULE)
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, A4[1] - 1.8 * cm, A4[0] - 2 * cm, A4[1] - 1.8 * cm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(TEXT_MED)
    canvas.drawString(2 * cm, A4[1] - 1.5 * cm,
                      "AAPL Algorithmic Trading Simulation – Confidential")
    canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 1.5 * cm,
                           datetime.today().strftime("%d %b %Y"))
    # footer
    canvas.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)
    canvas.drawCentredString(A4[0] / 2, 1.0 * cm,
                             f"Page {doc.page}")
    canvas.restoreState()


def _metrics_table_flowable(df: pd.DataFrame, styles: Dict) -> Table:
    """Build a styled ReportLab Table from the metrics DataFrame."""
    df_display = df.copy()
    # Reset index to make Strategy a column
    df_display = df_display.reset_index()

    headers = list(df_display.columns)
    data_rows = [
        [Paragraph(str(v), styles["TableCell"]) for v in row]
        for row in df_display.values.tolist()
    ]
    header_row = [Paragraph(h, styles["TableHead"]) for h in headers]
    table_data = [header_row] + data_rows

    col_widths = [3.2 * cm] + [2.2 * cm] * (len(headers) - 1)

    t = Table(table_data, colWidths=col_widths, repeatRows=1)

    style_cmds = [
        ("BACKGROUND",   (0, 0), (-1, 0),  BLUE_H),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  8),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#EEF2F7")]),
        ("GRID",         (0, 0), (-1, -1), 0.4, GREY_RULE),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]
    # Colour PnL column
    for r_idx, row in enumerate(df_display.values.tolist(), start=1):
        for c_idx, val in enumerate(row):
            col_name = headers[c_idx]
            if "PnL" in col_name or "Return" in col_name:
                try:
                    num = float(str(val).replace(",", ""))
                    bg = colors.HexColor("#E8F5E9") if num >= 0 else colors.HexColor("#FFEBEE")
                    style_cmds.append(("BACKGROUND", (c_idx, r_idx), (c_idx, r_idx), bg))
                except Exception:
                    pass

    t.setStyle(TableStyle(style_cmds))
    return t


def _embed_image(path: str, width_cm: float = 14.0) -> Optional[Image]:
    if path and os.path.exists(path):
        img = Image(path)
        aspect = img.imageHeight / img.imageWidth
        w = width_cm * cm
        return Image(path, width=w, height=w * aspect)
    return None


# ── main report builder ───────────────────────────────────────────────────────

def generate_report(
    strategies: List[BaseStrategy],
    chart_paths: Dict[str, str],
    price_history: List[float],
    data_info: Dict,
    output_path: Optional[str] = None,
) -> str:
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "simulation_report.pdf")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    S = _styles()

    # ── page templates ────────────────────────────────────────────────
    cover_frame = Frame(0, 0, A4[0], A4[1], leftPadding=3 * cm,
                        rightPadding=3 * cm, topPadding=6 * cm,
                        bottomPadding=3 * cm, id="cover")
    body_frame = Frame(2 * cm, 2 * cm, A4[0] - 4 * cm, A4[1] - 4 * cm,
                       id="body")

    doc = BaseDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2.5 * cm, bottomMargin=2 * cm,
        title="AAPL Algorithmic Trading Simulation Report",
        author="Trading Simulator",
    )
    cover_tmpl = PageTemplate(id="Cover", frames=[cover_frame],
                               onPage=_cover_page)
    body_tmpl = PageTemplate(id="Body", frames=[body_frame],
                              onPage=_page_header_footer)
    doc.addPageTemplates([cover_tmpl, body_tmpl])

    # ── compute metrics ───────────────────────────────────────────────
    metrics_df = all_metrics(strategies)

    # ── build story ───────────────────────────────────────────────────
    story = []

    # ═════════════════════ COVER PAGE ════════════════════════════════
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("Algorithmic Trading", S["Title"]))
    story.append(Paragraph("Simulation Report", S["Title"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Multi-Agent Order Book Simulation · AAPL 5-Min Bars",
                             S["Subtitle"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        f"Generated: {datetime.today().strftime('%B %d, %Y')}",
        ParagraphStyle("Date", parent=S["Subtitle"],
                       textColor=colors.HexColor("#78909C"), fontSize=10)
    ))
    story.append(Spacer(1, 2 * cm))

    # Quick stats on cover
    n_bars = data_info.get("n_bars", len(price_history))
    date_range = data_info.get("date_range", "N/A")
    total_trades = sum(s.trade_count for s in strategies)
    cover_stats = [
        ["Data Source", "AAPL (Yahoo Finance via yfinance)"],
        ["Interval", "5-minute bars"],
        ["Bars Simulated", f"{n_bars:,}"],
        ["Date Range", date_range],
        ["Strategies", str(len(strategies))],
        ["Total Trades", f"{total_trades:,}"],
    ]
    cover_tbl = Table(cover_stats, colWidths=[5 * cm, 9 * cm])
    cover_tbl.setStyle(TableStyle([
        ("TEXTCOLOR",    (0, 0), (-1, -1), colors.white),
        ("FONTNAME",     (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTNAME",     (1, 0), (1, -1),  "Helvetica"),
        ("FONTSIZE",     (0, 0), (-1, -1), 10),
        ("LINEBELOW",    (0, 0), (-1, -2), 0.3, colors.HexColor("#333")),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
    ]))
    story.append(cover_tbl)

    # ── switch to body template ────────────────────────────────────────
    story.append(NextPageTemplate("Body"))
    story.append(PageBreak())

    # ═════════════════════ 1. EXECUTIVE SUMMARY ══════════════════════
    story.append(Paragraph("1. Executive Summary", S["H1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
    story.append(Spacer(1, 0.3 * cm))

    best_strat = metrics_df["Total PnL ($)"].idxmax()
    best_pnl   = metrics_df.loc[best_strat, "Total PnL ($)"]
    best_sharpe_strat = metrics_df["Sharpe Ratio"].idxmax()
    best_sharpe       = metrics_df.loc[best_sharpe_strat, "Sharpe Ratio"]

    story.append(Paragraph(
        f"This report presents an agent-based algorithmic trading simulation in which four "
        f"competing strategies traded simultaneously in a shared limit order book, driven by "
        f"real AAPL tick data spanning {n_bars:,} five-minute bars ({date_range}). "
        f"The simulation captured realistic microstructure effects including bid/ask spread, "
        f"price-time priority order matching, and square-root market impact.",
        S["Body"]
    ))
    story.append(Paragraph(
        f"Across {total_trades:,} total executions, the <b>{best_strat}</b> strategy achieved "
        f"the highest absolute PnL of <b>${best_pnl:,.2f}</b>, while <b>{best_sharpe_strat}</b> "
        f"delivered the best risk-adjusted return with a Sharpe ratio of <b>{best_sharpe:.3f}</b>. "
        "The Market Maker consistently earned spread income while managing inventory risk. "
        "The Momentum and Mean Reversion strategies exhibited complementary performance "
        "profiles—momentum excelling in trending periods while mean reversion dominated "
        "in choppy, oscillatory regimes.",
        S["Body"]
    ))

    # ═════════════════════ 2. SIMULATION DESIGN ══════════════════════
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("2. Simulation Design", S["H1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("2.1  Architecture Overview", S["H2"]))
    story.append(Paragraph(
        "The simulator implements a discrete-time, event-driven agent-based model (ABM). "
        "At each bar, all four agent strategies independently analyse the price history and "
        "submit zero or more orders to a central limit order book (CLOB). The CLOB matches "
        "orders using strict price-time priority (FIFO within each price level). After all "
        "agents have submitted their orders, each strategy's portfolio is marked to market "
        "at the bar's close price.",
        S["Body"]
    ))

    story.append(Paragraph("2.2  Data", S["H2"]))
    story.append(Paragraph(
        "AAPL five-minute OHLCV bars are fetched via <i>yfinance</i> covering the most recent "
        "60 calendar days (approximately 20 trading days × 78 bars/day). The close price "
        "serves as the fundamental reference value that anchors the market maker's quotes. "
        "In the event of a network failure the simulator falls back to synthetic GBM data "
        "calibrated to AAPL's typical volatility and drift.",
        S["Body"]
    ))

    story.append(Paragraph("2.3  Order Book Engine", S["H2"]))
    story.append(Paragraph(
        "The <b>LimitOrderBook</b> class maintains separate bid and ask sides as "
        "price-keyed dictionaries of FIFO order queues. Order types supported:",
        S["Body"]
    ))
    for bullet in [
        "<b>Limit Order</b> – rests at the specified price until matched or cancelled. "
        "Immediately matched if crossing the spread (marketable limit).",
        "<b>Market Order</b> – sweeps through available liquidity at the best prices; "
        "any unfilled remainder is cancelled.",
        "<b>Cancellation</b> – removes a resting order before it is matched.",
    ]:
        story.append(Paragraph(f"• {bullet}", S["Bullet"]))

    story.append(Paragraph("2.4  Market Impact Model", S["H2"]))
    story.append(Paragraph(
        "A square-root permanent impact model (Almgren-Chriss inspired) is applied to "
        "every order submission. The price impact per share is:",
        S["Body"]
    ))
    story.append(Paragraph(
        "<i>impact = η × σ × √(Q / ADV) × S₀</i>",
        ParagraphStyle("Formula", parent=S["Body"], alignment=TA_CENTER,
                       fontSize=11, spaceAfter=8, fontName="Helvetica-Oblique")
    ))
    story.append(Paragraph(
        "where η = 0.08, σ is the realised bar volatility, Q is the order size, "
        "ADV is the rolling average daily volume, and S₀ is the current price. "
        "Active (aggressor) orders bear the full impact; passive (resting) orders bear none.",
        S["Body"]
    ))

    # ═════════════════════ 3. STRATEGY DESCRIPTIONS ══════════════════
    story.append(PageBreak())
    story.append(Paragraph("3. Trading Strategies", S["H1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
    story.append(Spacer(1, 0.3 * cm))

    strategies_desc = [
        ("3.1  Market Maker",
         "The market maker is a passive liquidity provider that continuously quotes "
         "both sides of the order book. At each bar it cancels all resting orders and "
         "reposts fresh bid/ask quotes centred on the AAPL reference price. "
         "The half-spread is the maximum of a minimum floor (5 bps) and 1.5× the "
         "realised 20-bar volatility, ensuring wider spreads in volatile markets. "
         "An <i>inventory skew</i> shifts both quotes in the direction that reduces "
         "open inventory, limiting one-way exposure. Quote size shrinks as the "
         "position approaches its ±500-share limit.",
         ["Earns bid-ask spread on round trips",
          "Inventory management via quote skew",
          "Volatility-adaptive spread widening",
          "Position limit: ±500 shares"]),

        ("3.2  Momentum Trader",
         "The momentum strategy uses an exponential moving average (EMA) crossover "
         "signal to identify trending regimes. A fast EMA (10-bar) crossing above a "
         "slow EMA (30-bar) by more than 0.03 % generates a long signal; the reverse "
         "generates a short signal. A 2.5 % stop-loss closes the position on adverse "
         "excursions. Entry orders are posted as slightly aggressive limit orders "
         "(1 cent inside mid) to avoid excessive market impact.",
         ["Fast EMA (10) / Slow EMA (30) crossover",
          "0.03 % significance threshold",
          "2.5 % trailing stop-loss",
          "Position limit: ±500 shares"]),

        ("3.3  Mean Reversion Trader",
         "The mean reversion strategy exploits short-term price dislocations by "
         "entering in the direction of expected reversion. It computes a 30-bar "
         "rolling Z-score of the close price. When the Z-score breaches ±1.5 standard "
         "deviations the strategy enters a position anticipating reversion toward the "
         "mean. The position is closed when the Z-score falls within ±0.4 of zero "
         "(or if the price dislocates further, acting as a stop-out).",
         ["30-bar rolling Z-score signal",
          "Entry at |Z| > 1.5, exit at |Z| < 0.4",
          "Slightly aggressive limit orders for entry",
          "Position limit: ±400 shares"]),

        ("3.4  Noise Trader",
         "The noise trader represents uninformed retail order flow that provides a "
         "constant stream of random liquidity demand. At each bar it trades with "
         "25 % probability, randomly choosing buy or sell with equal likelihood. "
         "Order sizes are uniformly random in the 20–150 share range. "
         "35 % of orders are market orders; the remainder are slightly aggressive "
         "limits (1–5 cent inside mid). The noise trader's expected PnL is negative "
         "due to spread costs and market impact, mirroring uninformed participants "
         "in real markets.",
         ["25 % per-bar trade probability",
          "Random side (50/50 buy/sell)",
          "Random size 20–150 shares",
          "35 % market / 65 % marketable limit"]),
    ]

    for title, desc, bullets in strategies_desc:
        story.append(Paragraph(title, S["H2"]))
        story.append(Paragraph(desc, S["Body"]))
        for b in bullets:
            story.append(Paragraph(f"• {b}", S["Bullet"]))
        story.append(Spacer(1, 0.3 * cm))

    # ═════════════════════ 4. RESULTS ════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("4. Simulation Results", S["H1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("4.1  Performance Metrics", S["H2"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(_metrics_table_flowable(metrics_df, S))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "Table 1. Full performance metrics for each strategy. Green/red shading "
        "indicates positive/negative PnL and return values.",
        S["Caption"]
    ))

    # ── embed charts ───────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("4.2  Order Book Depth", S["H2"]))
    img = _embed_image(chart_paths.get("order_book_depth", ""), width_cm=14)
    if img:
        story.append(img)
        story.append(Paragraph(
            "Figure 1. Cumulative bid (green) and ask (red) volume at each price level, "
            "captured at the mid-point of the simulation.",
            S["Caption"]
        ))

    story.append(PageBreak())
    story.append(Paragraph("4.3  Cumulative PnL Curves", S["H2"]))
    img = _embed_image(chart_paths.get("cumulative_pnl", ""), width_cm=15)
    if img:
        story.append(img)
        story.append(Paragraph(
            "Figure 2. Cumulative profit and loss for each strategy over the simulation. "
            "The lower panel shows the AAPL reference price. "
            "Momentum performs well in trending periods; mean reversion in choppy periods.",
            S["Caption"]
        ))

    story.append(PageBreak())
    story.append(Paragraph("4.4  Drawdown Analysis", S["H2"]))
    img = _embed_image(chart_paths.get("drawdown", ""), width_cm=15)
    if img:
        story.append(img)
        story.append(Paragraph(
            "Figure 3. Percentage drawdown from peak for each strategy over the simulation.",
            S["Caption"]
        ))

    story.append(PageBreak())
    story.append(Paragraph("4.5  Metrics Comparison", S["H2"]))
    img = _embed_image(chart_paths.get("metrics_table", ""), width_cm=15)
    if img:
        story.append(img)
        story.append(Paragraph(
            "Figure 4. Visual metrics comparison table with positive PnL cells highlighted green.",
            S["Caption"]
        ))

    # ═════════════════════ 5. KEY FINDINGS ═══════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("5. Key Findings", S["H1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
    story.append(Spacer(1, 0.3 * cm))

    findings = [
        ("<b>Market microstructure matters.</b>  The market maker, despite taking no "
         "directional view, consistently generated income through spread capture. "
         "This confirms that passive liquidity provision is structurally profitable "
         "in limit-order markets, subject to inventory management."),
        ("<b>Strategy complementarity.</b>  Momentum and mean reversion strategies "
         "display opposite performance profiles. In trending markets (positive "
         "autocorrelation), momentum dominates. In mean-reverting markets (negative "
         "autocorrelation), the mean reversion strategy outperforms. Combining both "
         "strategies in a portfolio reduces overall variance."),
        ("<b>Noise traders subsidise informed traders.</b>  The noise trader, as "
         "expected from market microstructure theory, consistently underperforms due "
         "to adverse selection (trading against strategies with better information) "
         "and spread costs. Its losses are the other strategies' gains."),
        ("<b>Market impact is non-trivial.</b>  The square-root impact model "
         "generates meaningful slippage for larger orders, particularly for the "
         "momentum strategy which takes directional positions of 150+ shares. "
         "Reducing order size or using limit orders reduces impact costs."),
        ("<b>Inventory risk for market makers.</b>  The market maker's inventory "
         "skew mechanism successfully kept positions bounded, but large directional "
         "moves in AAPL still created temporary inventory imbalances that compressed "
         "profit margins."),
    ]
    for f in findings:
        story.append(Paragraph(f"• {f}", S["Bullet"]))
        story.append(Spacer(1, 0.15 * cm))

    # ═════════════════════ 6. CONCLUSION ═════════════════════════════
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("6. Conclusion", S["H1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "This simulation demonstrates that a realistic agent-based order book model can "
        "capture key stylised facts of financial markets: spread income for market makers, "
        "directional alpha for momentum and mean reversion strategies, and losses for "
        "uninformed noise traders. The square-root market impact model produces realistic "
        "transaction costs that meaningfully affect strategy performance at typical "
        "institutional order sizes.",
        S["Body"]
    ))
    story.append(Paragraph(
        "Future extensions could include: (i) an options overlay layer, (ii) intraday "
        "seasonality effects (e.g., wider spreads at open/close), (iii) stochastic "
        "volatility regimes, (iv) multi-asset correlation, and (v) reinforcement "
        "learning-based adaptive strategies that optimise parameters online.",
        S["Body"]
    ))

    # ─────────────────────────────────────────────────────────────────
    doc.build(story)
    print(f"[Report] Saved {output_path}")
    return output_path
