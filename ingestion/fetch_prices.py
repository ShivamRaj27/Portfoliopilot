"""
fetch_prices.py
Layer 1 (Data Ingestion) - Pulls historical daily price data for a list of
tickers using yfinance and saves each as a clean Parquet file.

Usage:
    python fetch_prices.py
"""

import os
import sys
import time
import pandas as pd
import yfinance as yf

# ---- Config -----------------------------------------------------------
TICKERS = ["NKE", "AAPL", "MSFT"]   # add/remove tickers as needed
PERIOD = "5y"                        # how much history to pull
INTERVAL = "1d"                      # daily bars
OUTPUT_DIR = "data"
# ------------------------------------------------------------------------


def fetch_price_history(ticker: str, period: str = PERIOD, interval: str = INTERVAL) -> pd.DataFrame:
    """Download historical OHLCV price data for a single ticker."""
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)

    if df.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'. Check the symbol.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df.insert(0, "ticker", ticker)

    expected_cols = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
    df = df[[c for c in expected_cols if c in df.columns]]

    return df


def save_parquet(df: pd.DataFrame, ticker: str, output_dir: str = OUTPUT_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{ticker}_prices.parquet")
    df.to_parquet(path, index=False)
    return path


def main():
    print(f"Fetching price history for: {', '.join(TICKERS)}")
    successes, failures = [], []

    for ticker in TICKERS:
        try:
            df = fetch_price_history(ticker)
            path = save_parquet(df, ticker)
            print(f"  [OK]   {ticker}: {len(df)} rows -> {path}")
            successes.append(ticker)
        except Exception as e:
            print(f"  [FAIL] {ticker}: {e}")
            failures.append(ticker)
        time.sleep(1)

    print(f"\nDone. {len(successes)} succeeded, {len(failures)} failed.")
    if failures:
        print(f"Failed tickers: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()