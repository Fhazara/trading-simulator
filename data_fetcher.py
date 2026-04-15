"""
data_fetcher.py
Fetches real AAPL tick data via yfinance with a synthetic fallback.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_aapl_data(period: str = "60d", interval: str = "5m") -> pd.DataFrame:
    """
    Fetch AAPL OHLCV bars from Yahoo Finance.

    Returns a DataFrame with lowercase columns:
        open, high, low, close, volume, returns, log_returns
    """
    try:
        print(f"[Data] Fetching AAPL {interval} bars for the last {period}...")
        ticker = yf.Ticker("AAPL")
        raw = ticker.history(period=period, interval=interval, auto_adjust=True)

        if raw.empty:
            raise ValueError("yfinance returned an empty DataFrame.")

        data = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        data.columns = ["open", "high", "low", "close", "volume"]
        data = data.dropna()
        data = data[data["close"] > 0]

        data["returns"] = data["close"].pct_change()
        data["log_returns"] = np.log(data["close"]).diff()

        print(f"[Data] Fetched {len(data):,} bars  "
              f"({data.index[0]} → {data.index[-1]})")
        print(f"[Data] Price range: ${data['close'].min():.2f} – ${data['close'].max():.2f}")
        return data

    except Exception as exc:
        print(f"[Data] WARNING – yfinance fetch failed ({exc}). "
              "Using synthetic AAPL-like data.")
        return _generate_synthetic(n_bars=2000)


def _generate_synthetic(n_bars: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Geometric Brownian Motion with realistic AAPL parameters."""
    rng = np.random.default_rng(seed)
    start_price = 182.0
    mu = 0.00005       # per 5-min bar (~10 % annualised)
    sigma = 0.0018     # per 5-min bar (~20 % annualised vol)

    idx = pd.bdate_range(start="2024-01-02 09:30", periods=n_bars, freq="5min",
                         tz="America/New_York")

    log_ret = rng.normal(mu - 0.5 * sigma ** 2, sigma, n_bars)
    closes = start_price * np.exp(np.cumsum(log_ret))

    intrabar_range = closes * rng.uniform(0.0005, 0.0025, n_bars)
    opens = np.roll(closes, 1)
    opens[0] = start_price
    highs = np.maximum(opens, closes) + rng.exponential(intrabar_range * 0.5)
    lows = np.minimum(opens, closes) - rng.exponential(intrabar_range * 0.5)
    lows = np.maximum(lows, closes * 0.99)
    volumes = rng.lognormal(10.5, 0.8, n_bars).astype(int)

    data = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows,
         "close": closes, "volume": volumes},
        index=idx,
    )
    data["returns"] = data["close"].pct_change()
    data["log_returns"] = np.log(data["close"]).diff()
    return data
