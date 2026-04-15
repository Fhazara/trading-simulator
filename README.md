# AAPL Algorithmic Trading Simulator

A research-grade, agent-based trading simulation built on real AAPL tick data.  
Four competing strategies share a single limit order book with price-time priority matching and square-root market impact.

## Quick Start

```bash
cd /home/agent/trading-simulator
python3 simulate.py
```

All outputs are written to `./output/`.

## What It Does

| Step | Description |
|------|-------------|
| 1 | Fetches ~4 700 AAPL 5-minute bars via **yfinance** (auto-falls back to synthetic GBM data) |
| 2 | Runs four strategies simultaneously in a shared limit order book |
| 3 | Applies square-root market impact to each execution |
| 4 | Computes full performance metrics for every strategy |
| 5 | Saves **4 PNG charts** to `./output/` |
| 6 | Generates a **research-style PDF report** to `./output/simulation_report.pdf` |

---

## Project Structure

```
trading-simulator/
├── simulate.py              # ← single entry point
├── data_fetcher.py          # yfinance AAPL data + synthetic fallback
├── order_book.py            # limit order book engine
├── market_impact.py         # Almgren-Chriss square-root impact model
├── simulation.py            # agent-based simulation engine
├── metrics.py               # performance metric calculations
├── visualization.py         # PNG chart generation
├── report.py                # ReportLab PDF report generator
├── strategies/
│   ├── __init__.py
│   ├── base.py              # abstract BaseStrategy
│   ├── market_maker.py      # passive market-making
│   ├── momentum.py          # EMA-crossover trend-following
│   ├── mean_reversion.py    # Bollinger-band mean reversion
│   └── noise_trader.py      # random uninformed order flow
└── output/                  # generated at runtime
    ├── order_book_depth.png
    ├── cumulative_pnl.png
    ├── drawdown.png
    ├── metrics_table.png
    └── simulation_report.pdf
```

---

## Order Book Engine

The `LimitOrderBook` class implements:

- **Price-time priority (FIFO)** – orders at the same price are filled oldest-first
- **Limit orders** – rest in book until matched or cancelled; immediately match if marketable
- **Market orders** – sweep available liquidity at best prices; remainder cancelled
- **Cancellations** – O(1) cancel of any active order
- **Depth query** – cumulative bid/ask volume at up to N price levels

---

## Market Impact Model

Orders face a **square-root permanent impact**:

```
impact_per_share = η × σ × √(Q / ADV) × S₀
```

| Symbol | Meaning | Default |
|--------|---------|---------|
| η (eta) | impact coefficient | 0.08 |
| σ | realised 20-bar bar volatility | rolling |
| Q | order size (shares) | — |
| ADV | rolling average daily volume | rolling |
| S₀ | current price | — |

Only **aggressor** (active) orders bear impact. Passive (resting) orders do not.

---

## Strategies

### Market Maker
- Continuously quotes both bid and ask around the AAPL reference price
- Spread = max(5 bps floor, 1.5 × realised vol)
- Inventory skew shifts quotes to reduce one-way exposure
- Position limit: ±500 shares

### Momentum
- EMA crossover: fast (10-bar) vs slow (30-bar) with 0.03 % significance threshold
- Enters with slightly aggressive limit orders; exits on reverse signal or 2.5 % stop-loss
- Position limit: ±500 shares

### Mean Reversion
- 30-bar rolling Z-score: enter when |Z| > 1.5, exit when |Z| < 0.4
- Captures mean-reversion after short-term dislocations
- Position limit: ±400 shares

### Noise Trader
- 25 % per-bar probability of trading, random side, random size (20–150 shares)
- 35 % market orders / 65 % aggressive limits
- Models uninformed retail order flow; structurally unprofitable

---

## Performance Metrics

| Metric | Description |
|--------|-------------|
| Total PnL ($) | Absolute dollar profit/loss |
| Total Return (%) | Return on initial capital |
| Ann. Return (%) | Annualised return |
| Sharpe Ratio | Annualised mean / std of returns |
| Sortino Ratio | Annualised mean / downside-std |
| Max Drawdown (%) | Largest peak-to-trough decline |
| Calmar Ratio | Annualised return / max drawdown |
| VaR 95% ($) | 5th percentile one-bar dollar loss |
| CVaR 95% ($) | Expected loss below VaR (Expected Shortfall) |
| Win Rate (%) | Fraction of bars with positive return |
| Num Trades | Total executed fills |

---

## Output Files

| File | Description |
|------|-------------|
| `order_book_depth.png` | Cumulative bid/ask depth snapshot at mid-simulation |
| `cumulative_pnl.png` | Per-strategy cumulative PnL + AAPL price panel |
| `drawdown.png` | Per-strategy drawdown curves |
| `metrics_table.png` | Colour-coded metrics comparison table |
| `simulation_report.pdf` | Full research report (cover, design, results, findings) |

---

## Dependencies

```
yfinance      # AAPL tick data
numpy         # numerical computations
pandas        # data manipulation
matplotlib    # chart generation
scipy         # statistical utilities
reportlab     # PDF generation
```

Install all:
```bash
pip install yfinance numpy pandas matplotlib scipy reportlab
```

---

## Design Notes

- **Zero-sum by construction** (excluding market impact): one agent's gain is another's loss
- **Warmup period**: strategies skip the first 20–35 bars to accumulate signal history
- **Bars per year**: 252 trading days × 78 five-minute bars = 19,656 (used for annualisation)
- **Synthetic fallback**: if yfinance is unavailable, GBM with μ = 0.005 % / bar and σ = 0.18 % / bar is used
