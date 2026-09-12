"""
price_signals.py
Layer 2 (Quant & ML) - Computes trend and anomaly signals from the daily
price data Layer 1 saved in data/{ticker}_prices.parquet.

Signals computed:
  - Moving averages (50-day, 200-day) + golden/death cross flag
  - Rolling volatility (20-day annualized)
  - Daily return z-score (flags unusually large single-day moves)
  - RSI (14-day) - simple momentum/overbought-oversold indicator

Input:  data/{ticker}_prices.parquet
Output: data/{ticker}_price_signals.parquet   (same date index, extra columns)
        Also prints any rows currently flagged as anomalous (|z-score| > 2.5)

Usage:
    python price_signals.py --tickers NKE AAPL MSFT
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_TICKERS = ["NKE"]

ANOMALY_Z_THRESHOLD = 2.5


def load_prices(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker}_prices.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run fetch_prices.py for {ticker} first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Daily returns
    out["daily_return"] = out["close"].pct_change()

    # Moving averages + trend cross
    out["ma_50"] = out["close"].rolling(window=50, min_periods=50).mean()
    out["ma_200"] = out["close"].rolling(window=200, min_periods=200).mean()
    out["trend_signal"] = np.select(
        [out["ma_50"] > out["ma_200"], out["ma_50"] < out["ma_200"]],
        ["bullish", "bearish"],
        default="neutral",
    )

    # Rolling volatility (20-day, annualized assuming ~252 trading days)
    out["volatility_20d"] = out["daily_return"].rolling(window=20, min_periods=20).std() * np.sqrt(252)

    # Z-score of daily return (flags unusually large single-day moves)
    roll_mean = out["daily_return"].rolling(window=60, min_periods=60).mean()
    roll_std = out["daily_return"].rolling(window=60, min_periods=60).std()
    out["return_zscore"] = (out["daily_return"] - roll_mean) / roll_std
    out["is_anomalous"] = out["return_zscore"].abs() > ANOMALY_Z_THRESHOLD

    # RSI
    out["rsi_14"] = compute_rsi(out["close"], window=14)

    return out


def main():
    parser = argparse.ArgumentParser(description="Compute price trend/anomaly signals.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    args = parser.parse_args()

    for ticker in args.tickers:
        ticker = ticker.upper()
        try:
            prices = load_prices(ticker)
            signals = compute_signals(prices)

            out_path = DATA_DIR / f"{ticker}_price_signals.parquet"
            signals.to_parquet(out_path, index=False)

            latest = signals.iloc[-1]
            print(f"[OK] {ticker}: {len(signals)} rows -> {out_path}")
            print(
                f"     latest close={latest['close']:.2f}  "
                f"trend={latest['trend_signal']}  "
                f"vol_20d={latest['volatility_20d']:.3f}  "
                f"rsi_14={latest['rsi_14']:.1f}"
            )

            anomalies = signals[signals["is_anomalous"]].tail(5)
            if len(anomalies):
                print(f"     recent anomalous move(s):")
                print(anomalies[["date", "close", "daily_return", "return_zscore"]].to_string(index=False))
            print()
        except Exception as e:
            print(f"[FAIL] {ticker}: {e}")


if __name__ == "__main__":
    main()
