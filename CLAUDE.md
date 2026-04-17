# AAPL Algorithmic Trading Simulator — Project Reference

## What This Project Is

An agent-based, multi-strategy algorithmic trading simulator driven by real AAPL 5-minute OHLCV bars (fetched via yfinance, with a synthetic GBM fallback). Five competing strategies share a single central limit order book (CLOB) with strict price-time priority matching. Each bar, every strategy independently analyses price history and submits orders; the engine matches them, applies square-root market impact to aggressors, and marks each portfolio to market at the bar close.

The simulator produces four PNG charts, a research-style PDF report, and an interactive Plotly HTML dashboard, all written to `./output/`.

**Entry point:** `python3 simulate.py`

---

## File Structure

```
trading-simulator/
├── simulate.py              # Entry point: orchestrates all steps (data → sim → metrics → outputs)
├── simulation.py            # TradingSimulation engine: bar-by-bar loop, background depth, fill dispatch
├── order_book.py            # LimitOrderBook: price-time priority CLOB, Order/Trade dataclasses
├── market_impact.py         # MarketImpactModel: Almgren-Chriss square-root impact
├── data_fetcher.py          # fetch_aapl_data() via yfinance; _generate_synthetic() GBM fallback
├── regime.py                # RegimeDetector + compute_adx(): ADX-based TRENDING/RANGING classifier
├── metrics.py               # compute_metrics() / all_metrics(): full per-strategy metric suite
├── visualization.py         # generate_all(): 4 Matplotlib PNGs saved to output/
├── report.py                # generate_report(): multi-page ReportLab PDF
├── dashboard.py             # generate_dashboard(): self-contained interactive Plotly HTML
├── strategies/
│   ├── __init__.py          # Re-exports all five strategy classes
│   ├── base.py              # BaseStrategy (ABC): cash/position accounting, apply_fill, mark_to_market
│   ├── market_maker.py      # MarketMakerStrategy: vol-adaptive spread, inventory skew
│   ├── momentum.py          # MomentumStrategy: EMA crossover + ADX regime gate + stop-loss
│   ├── mean_reversion.py    # MeanReversionStrategy: rolling Z-score + ADX regime gate
│   ├── noise_trader.py      # NoiseTraderStrategy: random uninformed order flow
│   └── trend_follower.py    # TrendFollowerStrategy: Donchian channel breakout + ATR stops
└── output/                  # Generated at runtime
    ├── order_book_depth.png
    ├── cumulative_pnl.png
    ├── drawdown.png
    ├── metrics_table.png
    ├── simulation_report.pdf
    └── dashboard.html
```

### File responsibilities in detail

| File | Key classes / functions | Role |
|------|------------------------|------|
| `simulate.py` | `main()` | Auto-installs deps, wires all modules, prints metrics, runs steps 1–7 |
| `simulation.py` | `TradingSimulation` | Owns the order book and impact model; seeds background depth per bar; dispatches fills |
| `order_book.py` | `LimitOrderBook`, `Order`, `Trade` | FIFO CLOB, limit/market orders, O(1) cancel-by-ID, `purge_inactive()` |
| `market_impact.py` | `MarketImpactModel` | Rolling ADV, `impact_per_share = η·σ·√(Q/ADV)·S₀` |
| `data_fetcher.py` | `fetch_aapl_data()` | 60-day / 5-min AAPL bars; GBM synthetic fallback (seed=42, μ=0.005%/bar, σ=0.18%/bar) |
| `regime.py` | `RegimeDetector`, `compute_adx()` | Wilder-smoothed ADX; classifies each bar as TRENDING_UP/DOWN/RANGING/UNKNOWN; result cached per step |
| `metrics.py` | `compute_metrics()`, `all_metrics()` | Sharpe, Sortino, Calmar, VaR 95%, CVaR 95%, win rate, drawdown, annualised return |
| `visualization.py` | `generate_all()` | 4 dark-theme Matplotlib PNGs (depth snapshot, cumulative PnL, drawdown, metrics table) |
| `report.py` | `generate_report()` | 6-section ReportLab PDF: cover, design, strategies, results, findings, conclusion |
| `dashboard.py` | `generate_dashboard()` | 4-panel Plotly HTML: price+regime bands, PnL, drawdown, metric bar chart with dropdown |
| `strategies/base.py` | `BaseStrategy`, `OrderSpec` | Abstract base; tracks cash, position, value/PnL history; `apply_fill()` handles impact direction |

---

## Last Simulation Results

Recorded run on real AAPL 5-minute data (60-day window, ~4,700 bars).

| Strategy | Total PnL ($) | Sharpe Ratio | Notes |
|----------|--------------|--------------|-------|
| **MarketMaker** | **+$10,844** | **1.69** | Best absolute PnL and best risk-adjusted return |
| Momentum | -$3,531 | — | EMA crossover; losses partly from trending periods |
| MeanReversion | -$4,269 | — | Rolling Z-score; losses from trending regimes |
| NoiseTrader | -$21,470 | — | Random; structurally unprofitable (spread + impact costs) |
| TrendFollower | -$6,505 | — | Donchian breakout; ATR-scaled stops limited damage |

All strategies started with $100,000 initial capital. Market Maker is the only consistently profitable strategy, earning spread income while managing inventory. NoiseTrader losses subsidise informed strategies as predicted by microstructure theory.

---

## Key Design Decisions

### 1. Background depth at 30 bps per level (`_BG_LEVEL_BPS = 30`)

Background liquidity is seeded at 12 price levels on each side at 30 bps spacing. This ensures market orders always have a counterparty of last resort. The 30 bps floor is deliberately **wider than the market maker's typical half-spread** (~5–12 bps depending on realised vol), so the MM's quotes are always tighter and fill first. If background depth were tighter than the MM, the MM would be skipped and never earn spread.

### 2. Cancel-by-ID for O(1) background order removal

`_seed_background_depth()` stores the order IDs of the 24 background orders placed each bar in `self._bg_ids`. On the next bar it calls `cancel_order(oid)` per ID directly — O(n_levels) = O(24) — instead of using `cancel_all_orders("_bg_")` which scans the entire `orders` dict and becomes O(dict-size) as the simulation grows. This is a deliberate performance choice documented in `simulation.py`.

### 3. `purge_inactive()` every 200 bars

`LimitOrderBook.orders` is an append-only dict of all orders ever submitted. Filled and cancelled orders accumulate there and slow down any full-dict iteration. `purge_inactive()` deletes entries where `o.is_active == False`. It is safe because the bid/ask price-level structures hold live Order object references; the dict entries being deleted are already inactive. Called at `step % 200 == 199` in the main loop.

### 4. Regime detection via ADX (Average Directional Index)

`RegimeDetector` uses Wilder's 14-bar smoothed ADX with a threshold of 25:
- **ADX ≥ 25 + (+DI > -DI)** → `TRENDING_UP`
- **ADX ≥ 25 + (-DI > +DI)** → `TRENDING_DOWN`
- **ADX < 25** → `RANGING`
- **Insufficient history** → `UNKNOWN`

Results are cached per `(step)` so multiple strategies calling `detect()` at the same bar pay zero recomputation cost.

**Momentum** uses the regime gate to suppress shorts in uptrends and longs in downtrends; it disables all new entries in RANGING (where EMA crossovers whipsaw). **MeanReversion** allows new entries only in RANGING/UNKNOWN; in trending regimes it force-closes any open position to avoid "catching a falling knife." **TrendFollower** has no regime gate — breakouts ARE the regime confirmation.

### 5. Cancel-and-refresh every bar (strategies)

All five strategies call `order_book.cancel_all_orders(self.trader_id)` at the start of `generate_orders()`. This prevents stale resting limits from filling at out-of-date prices across multiple bars when a signal persists.

### 6. Almgren-Chriss square-root market impact (η = 0.08)

`impact_per_share = η × σ × √(Q / ADV) × S₀`

Only the **aggressor** (active order submitter) bears impact. The passive resting side bears none. ADV is a rolling 78-bar (≈ 1 trading day) average of bar volumes, scaled to a daily estimate.

### 7. Annualisation basis: 252 × 78 = 19,656 bars/year

Metrics (Sharpe, Sortino, Ann. Return) annualise using 5-min bars per trading year, not calendar days. This is correct for intraday data.

---

## Coding Conventions

- **`from __future__ import annotations`** in every module — enables PEP 563 deferred evaluation, needed for forward references in type hints.
- **`TYPE_CHECKING` imports** — heavy cross-module imports (e.g., `LimitOrderBook` inside strategies) are wrapped in `if TYPE_CHECKING:` to avoid circular imports at runtime.
- **`__slots__`** on `OrderSpec` — reduces per-instance memory for the high-frequency `OrderSpec` objects created every bar.
- **Dataclasses** for `Order` and `Trade` — clean field access, auto-`__init__`, no boilerplate.
- **`itertools.count`** for monotonically increasing order and trade IDs — simpler and safer than manual counters.
- **`deque` per price level** — FIFO queue for orders at the same price; `popleft()` is O(1).
- **`Optional[float]` for prices** — market orders have `price=None`; None propagates safely through matching logic.
- **Docstring convention** — module-level docstring describes the file's role, then a section showing the key logic / algorithm, then performance notes where relevant.
- **Private helpers prefixed `_`** — internal functions (`_match`, `_apply_fill`, `_seed_background_depth`, `_ema`, `_atr`) are prefixed to signal they are not part of the public API.
- **Constants at module top in SCREAMING_SNAKE_CASE** — e.g., `_BG_LEVELS`, `_BG_LEVEL_BPS`, `BARS_PER_YEAR`, `PALETTE`.
- **`Agg` matplotlib backend** — set at import time in `visualization.py` to avoid display errors in headless environments.
- **Auto-install in `simulate.py`** — `_ensure_packages()` runs before any project imports so the script is zero-setup from a bare Python environment.

---

## Improvements Already Made

These were added on top of the initial commit (which had 4 strategies and no regime detection):

1. **ADX regime gate on Momentum** — suppresses counter-trend EMA crossover entries; prevents going long into the $280→$244 AAPL downtrend that caused −$10k losses in the original version.
2. **ADX regime gate on MeanReversion** — disables new entries in trending regimes; force-closes open positions when regime flips to trending to avoid "buying dips that didn't revert."
3. **TrendFollowerStrategy added** — fifth strategy; Donchian channel breakout (20-bar high/low), ATR-scaled stops (2×ATR) and targets (3×ATR), dynamic position sizing (1% capital risk ÷ stop distance). No regime gate needed — breakout IS the confirmation.
4. **Cancel-by-ID for background orders** — replaced `cancel_all_orders("_bg_")` O(dict-size) scan with direct ID-based cancellations that are O(n_levels); stored as `self._bg_ids`.
5. **`purge_inactive()` every 200 bars** — prevents unbounded `orders` dict growth slowing late-simulation iterations.
6. **Interactive HTML dashboard** (`dashboard.py`) — Plotly 4-panel dashboard: AAPL price with ADX regime colour bands, cumulative PnL, drawdown, per-metric bar chart with dropdown selector. Saved as a self-contained HTML file (no server needed).
7. **`RegimeDetector` caching** — per-step memoisation so multiple strategies calling `detect()` on the same bar pay zero recomputation.
8. **`NoiseTraderStrategy` cancel-and-refresh** — added `cancel_all_orders()` at bar start to prevent aggressive resting limits from the prior bar accumulating position beyond `max_position`.

---

## Still on the Improvement List

From the report conclusion and known gaps:

1. **Walk-forward / out-of-sample backtesting** — current setup runs in-sample over the fetched window; need rolling re-fit windows to validate parameter robustness.
2. **Parameter sensitivity analysis** — no grid search or sensitivity sweep over key parameters (spread bps, EMA spans, Z-score thresholds, ATR multipliers, ADX threshold).
3. **Multi-asset support** — all strategies trade a single ticker (AAPL); pairs trading and cross-asset stat-arb strategies would require a multi-instrument order book or correlated price feeds.
4. **Hawkes process background order arrivals** — current background depth is deterministically re-seeded every bar at fixed levels; a Hawkes process would produce more realistic clustered liquidity dynamics.
5. **Live paper-trading via broker API** — the `OrderSpec` / `generate_orders()` interface is clean enough to wire to a real broker (e.g., Alpaca), but no adapter exists yet.
6. **Reinforcement learning adaptive strategies** — strategies with fixed signal parameters; an RL agent could adapt position sizing or signal thresholds online.
7. **Transaction cost model for cancellations** — exchanges charge for excessive cancel-to-trade ratios; not modelled here.
8. **Partial-fill tracking in strategies** — `apply_fill()` is called per trade but strategies don't distinguish partial fills from full fills when deciding next-bar actions; this can cause slight over/under-sizing on large orders.
