"""
simulation.py
Agent-based trading simulation engine.

At each bar:
  1. Background depth is seeded at ±N price levels (wider than any strategy's
     typical spread) so market orders always have a counterparty of last resort.
  2. Strategies generate and submit orders; immediate fills are processed.
  3. All strategies mark portfolios to market at the bar close.

Performance notes
─────────────────
• Background orders are tracked by ID and cancelled O(1) per order per bar,
  avoiding an O(dict-size) scan via cancel_all_orders.
• LimitOrderBook.purge_inactive() is called every 200 bars to reclaim memory
  from the orders dict, keeping iteration fast.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from order_book import LimitOrderBook, Side, OrderType, Trade
from market_impact import MarketImpactModel
from strategies.base import BaseStrategy, OrderSpec

# ── Background liquidity constants ───────────────────────────────────────────
# Background depth must be WIDER than typical strategy spreads so strategies
# (especially the market maker) get filled before background does.
#   Market maker typical half-spread ≈ max(5 bps, 0.5 × rv × mid)
#   With rv ≈ 0.002 and mid ≈ 250: half-spread ≈ max(0.125, 0.25) = 0.25
#   Background level-1 at 30 bps = 0.75.  MM is tighter → MM fills first.
_BG_LEVELS           = 12
_BG_LEVEL_BPS        = 30          # spacing per level – must be > MM spread
_BG_SHARES_PER_LEVEL = 500         # ample depth so market orders always fill
_BG_TRADER_ID        = "_bg_"


class TradingSimulation:
    """
    Multi-agent simulation driven by real AAPL price data.

    Parameters
    ----------
    data       : OHLCV DataFrame (from data_fetcher)
    strategies : list of BaseStrategy instances
    impact_eta : square-root impact coefficient η
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategies: List[BaseStrategy],
        impact_eta: float = 0.08,
    ) -> None:
        self.data       = data.reset_index(drop=True)
        self.strategies = strategies
        self.order_book = LimitOrderBook()
        self.impact     = MarketImpactModel(eta=impact_eta)

        self._strat_map: Dict[str, BaseStrategy] = {
            s.trader_id: s for s in strategies
        }

        # Background order IDs from the most recent seed (for fast cancel)
        self._bg_ids: List[int] = []

        self.price_history:  List[float]          = []
        self.spread_history: List[Optional[float]] = []
        self.depth_snapshot: Optional[Dict]        = None
        self.all_trades:     List[Trade]           = []

    # ── Background liquidity ──────────────────────────────────────────────────

    def _seed_background_depth(self, ref_price: float) -> None:
        """
        Cancel last bar's background orders by their stored IDs (O(n_levels)),
        then post fresh depth at ±_BG_LEVELS price levels.
        """
        # Direct cancel – O(n_bg_levels), avoids scanning the full orders dict
        for oid in self._bg_ids:
            self.order_book.cancel_order(oid)
        self._bg_ids.clear()

        tick = _BG_LEVEL_BPS / 10_000
        for i in range(1, _BG_LEVELS + 1):
            off = ref_price * tick * i
            bid_px = round(ref_price - off, 2)
            ask_px = round(ref_price + off, 2)
            if bid_px > 0 and ask_px > bid_px:
                o_bid, _ = self.order_book.submit_order(
                    _BG_TRADER_ID, Side.BID, OrderType.LIMIT,
                    _BG_SHARES_PER_LEVEL, bid_px,
                )
                o_ask, _ = self.order_book.submit_order(
                    _BG_TRADER_ID, Side.ASK, OrderType.LIMIT,
                    _BG_SHARES_PER_LEVEL, ask_px,
                )
                self._bg_ids.append(o_bid.order_id)
                self._bg_ids.append(o_ask.order_id)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the full simulation bar-by-bar."""
        n_bars = len(self.data)
        print(f"\n[Sim] Starting simulation: {n_bars:,} bars, "
              f"{len(self.strategies)} strategies")
        t0 = time.time()

        for step in range(n_bars):
            bar       = self.data.iloc[step]
            ref_price = float(bar["close"])
            self.order_book.timestamp = step

            # Update rolling ADV for market impact
            self.impact.update(float(bar["volume"]))

            # Rolling bar volatility
            lookback = min(20, step)
            if lookback > 1:
                recent = self.data["log_returns"].iloc[
                    max(0, step - lookback): step + 1
                ].dropna()
                bar_sigma = float(recent.std()) if len(recent) > 1 else 0.002
            else:
                bar_sigma = 0.002

            # ── background depth ──────────────────────────────────────
            self._seed_background_depth(ref_price)

            # ── strategy orders ───────────────────────────────────────
            for strat in self.strategies:
                specs: List[OrderSpec] = strat.generate_orders(
                    self.data, self.order_book, step
                )
                for spec in specs:
                    try:
                        order, trades = self.order_book.submit_order(
                            trader_id=strat.trader_id,
                            side=spec.side,
                            order_type=spec.order_type,
                            quantity=spec.quantity,
                            price=spec.price,
                        )
                        for trade in trades:
                            self._process_trade(
                                trade,
                                active_trader=strat.trader_id,
                                ref_price=ref_price,
                                bar_sigma=bar_sigma,
                            )
                    except Exception:
                        pass

            # ── mark to market ────────────────────────────────────────
            for strat in self.strategies:
                strat.mark_to_market(ref_price)

            self.price_history.append(ref_price)
            self.spread_history.append(self.order_book.spread())

            # Depth snapshot at mid-simulation
            if step == n_bars // 2:
                self.depth_snapshot = self.order_book.depth(levels=15)

            # Periodic memory reclaim – keeps orders dict lean
            if step % 200 == 199:
                self.order_book.purge_inactive()

        elapsed = time.time() - t0
        strat_trades = [
            t for t in self.order_book.trades
            if t.buyer_id != _BG_TRADER_ID or t.seller_id != _BG_TRADER_ID
        ]
        print(f"[Sim] Done in {elapsed:.1f}s – "
              f"{len(self.order_book.trades):,} total fills, "
              f"{len(strat_trades):,} strategy fills")

        if self.depth_snapshot is None:
            self.depth_snapshot = self.order_book.depth(levels=15)

        self.all_trades = strat_trades

    # ── Fill processing ───────────────────────────────────────────────────────

    def _process_trade(
        self,
        trade: Trade,
        active_trader: str,
        ref_price: float,
        bar_sigma: float,
    ) -> None:
        if trade.buyer_id == _BG_TRADER_ID and trade.seller_id == _BG_TRADER_ID:
            return

        impact = self.impact.impact_per_share(
            trade.quantity, ref_price, sigma=bar_sigma
        )
        buyer  = self._strat_map.get(trade.buyer_id)
        seller = self._strat_map.get(trade.seller_id)

        if buyer is not None:
            buyer.apply_fill(trade,
                             impact_per_share=impact if trade.buyer_id == active_trader else 0.0)
        if seller is not None:
            seller.apply_fill(trade,
                              impact_per_share=impact if trade.seller_id == active_trader else 0.0)

    # ── Summary ───────────────────────────────────────────────────────────────

    def strategy_summary(self) -> pd.DataFrame:
        rows = []
        for s in self.strategies:
            rows.append({
                "Strategy":      s.trader_id,
                "Trades":        s.trade_count,
                "Final PnL ($)": round(s.pnl_history[-1], 2) if s.pnl_history else 0,
                "Final Pos":     int(round(s.position)),
            })
        return pd.DataFrame(rows)
