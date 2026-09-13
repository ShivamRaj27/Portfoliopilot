"""
feature_engineering.py
Layer 2 (Quant & ML) - Builds a supervised-learning-ready feature matrix
from the data already produced by fetch_prices.py, price_signals.py, and
compute_ratios.py. Shared by train_classifiers.py and lasso_selection.py so
both use IDENTICAL features and targets - important for comparing them fairly.

TARGETS (both computed from the SAME future price, just framed differently):
  target_direction : 1 if close_{t+1} > close_t, else 0   (classification)
  target_return    : (close_{t+1} - close_t) / close_t    (regression, for Lasso)

FEATURES (every one is computable using only data available AS OF day t -
no lookahead leakage):
  return_lag_1/2/3/5   : past daily returns, at various lags
  volatility_20d       : already computed in price_signals.py
  rsi_14                : already computed in price_signals.py
  return_zscore         : already computed in price_signals.py
  dist_from_ma50        : close / ma_50 - 1 (normalized trend distance)
  dist_from_ma200       : close / ma_200 - 1
  volume_ratio          : today's volume / its own 20-day rolling average
  latest ratio columns  : forward-filled from compute_ratios.py's annual
                           output (net_margin, current_ratio, debt_to_equity,
                           asset_turnover) - included so Lasso can show
                           whether slow-moving fundamentals matter at all
                           for next-day price movement (spoiler: probably not
                           much - and that's a legitimate, reportable finding,
                           not a bug).

IMPORTANT - avoiding lookahead leakage: the target is shifted, not the
features. Row t's features are all "as of end of day t"; row t's target
describes what happens on day t+1. The LAST row of the resulting dataframe
has no target (there is no "tomorrow" yet) and is dropped by default.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

FEATURE_COLUMNS = [
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    "volatility_20d", "rsi_14", "return_zscore",
    "dist_from_ma50", "dist_from_ma200", "volume_ratio",
    "net_margin", "current_ratio", "debt_to_equity", "asset_turnover",
]


def _load_price_signals(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker}_price_signals.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run price_signals.py for {ticker} first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _load_ratios_daily(ticker: str, dates: pd.Series) -> pd.DataFrame:
    """
    Ratios are annual (one row per fiscal year). To use them as a daily
    feature, forward-fill each year's values across every trading day until
    the next fiscal year's ratios are computed - i.e. "the most recently
    known ratio as of this date," never a future one.
    """
    path = DATA_DIR / f"{ticker}_ratios.parquet"
    cols = ["net_margin", "current_ratio", "debt_to_equity", "asset_turnover"]
    if not path.exists():
        return pd.DataFrame({c: np.nan for c in cols}, index=dates.index)

    ratios = pd.read_parquet(path).reset_index()
    # Treat each fiscal year's ratios as "known" starting roughly 3 months
    # after fiscal year-end (typical 10-K filing lag), to avoid pretending
    # we knew the ratio before the filing was actually public.
    ratios["known_from"] = pd.to_datetime(ratios["fiscal_year"].astype(str) + "-12-31") + pd.DateOffset(months=3)
    ratios = ratios.sort_values("known_from")

    daily = pd.DataFrame({"date": dates})

    # pandas is strict about merge_asof's two key columns having IDENTICAL
    # datetime precision (e.g. datetime64[ms] vs datetime64[us] will raise a
    # MergeError even though both are "just dates") - force both to the same
    # precision explicitly rather than relying on whatever precision each
    # side happened to come out of Parquet with.
    daily["date"] = daily["date"].astype("datetime64[ns]")
    ratios["known_from"] = ratios["known_from"].astype("datetime64[ns]")

    merged = pd.merge_asof(daily.sort_values("date"), ratios[["known_from"] + cols],
                            left_on="date", right_on="known_from", direction="backward")
    return merged[cols].reset_index(drop=True)


def build_feature_matrix(ticker: str, drop_last_row: bool = True) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by date with columns: FEATURE_COLUMNS +
    ['target_direction', 'target_return', 'close']. One row per trading day.
    Rows without enough history for all features (e.g. the first ~200 days,
    before rolling windows fill in) are dropped.
    """
    ticker = ticker.upper()
    signals = _load_price_signals(ticker)

    df = pd.DataFrame({"date": signals["date"], "close": signals["close"]})

    df["return_lag_1"] = signals["daily_return"]
    df["return_lag_2"] = signals["daily_return"].shift(1)
    df["return_lag_3"] = signals["daily_return"].shift(2)
    df["return_lag_5"] = signals["daily_return"].shift(4)

    df["volatility_20d"] = signals["volatility_20d"]
    df["rsi_14"] = signals["rsi_14"]
    df["return_zscore"] = signals["return_zscore"]

    df["dist_from_ma50"] = signals["close"] / signals["ma_50"] - 1
    df["dist_from_ma200"] = signals["close"] / signals["ma_200"] - 1

    vol_ma20 = signals["volume"].rolling(window=20, min_periods=20).mean()
    df["volume_ratio"] = signals["volume"] / vol_ma20

    ratios_daily = _load_ratios_daily(ticker, df["date"])
    df = pd.concat([df, ratios_daily], axis=1)

    # Targets - built from the NEXT row's close, then that next row's
    # features are irrelevant to this row (this row only "knows" up to
    # today's close).
    df["target_return"] = df["close"].pct_change().shift(-1)
    df["target_direction"] = (df["target_return"] > 0).astype(int)

    if drop_last_row:
        df = df.iloc[:-1]  # last row's target is unknown (no "tomorrow" yet)

    df = df.dropna(subset=FEATURE_COLUMNS + ["target_return", "target_direction"])
    return df.reset_index(drop=True)


def get_latest_feature_row(ticker: str) -> pd.Series:
    """
    Returns a single row of FEATURE_COLUMNS values "as of today" - i.e. the
    most recent trading day's features, with NO target attached (we don't
    know tomorrow's direction yet - that's the whole point of predicting it).

    This is deliberately separate from build_feature_matrix(), which always
    drops rows missing a target (needed for training/testing, wrong for live
    prediction). Used by tools.py's predict_direction() to feed a saved model
    a real, current feature vector.
    """
    ticker = ticker.upper()
    signals = _load_price_signals(ticker)

    df = pd.DataFrame({"date": signals["date"], "close": signals["close"]})
    df["return_lag_1"] = signals["daily_return"]
    df["return_lag_2"] = signals["daily_return"].shift(1)
    df["return_lag_3"] = signals["daily_return"].shift(2)
    df["return_lag_5"] = signals["daily_return"].shift(4)
    df["volatility_20d"] = signals["volatility_20d"]
    df["rsi_14"] = signals["rsi_14"]
    df["return_zscore"] = signals["return_zscore"]
    df["dist_from_ma50"] = signals["close"] / signals["ma_50"] - 1
    df["dist_from_ma200"] = signals["close"] / signals["ma_200"] - 1
    vol_ma20 = signals["volume"].rolling(window=20, min_periods=20).mean()
    df["volume_ratio"] = signals["volume"] / vol_ma20

    ratios_daily = _load_ratios_daily(ticker, df["date"])
    df = pd.concat([df, ratios_daily], axis=1)

    # Only require the FEATURE_COLUMNS to be present - unlike
    # build_feature_matrix, we do NOT require a target, since this row IS
    # "today" and has no known tomorrow.
    df = df.dropna(subset=FEATURE_COLUMNS)
    if df.empty:
        raise ValueError(f"No row with complete features found for {ticker}.")

    latest = df.iloc[-1]
    return latest


def chronological_split(df: pd.DataFrame, test_size: float = 0.2):
    """
    Splits BY DATE, not randomly - critical for time series. A random split
    would let a model train on future data and test on the past, silently
    inflating accuracy in a way that would not generalize to real use.
    """
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build and preview the ML feature matrix for a ticker.")
    parser.add_argument("--ticker", default="NKE")
    args = parser.parse_args()

    fm = build_feature_matrix(args.ticker)
    print(f"{args.ticker}: {len(fm)} rows, {len(FEATURE_COLUMNS)} features")
    print(fm[["date"] + FEATURE_COLUMNS + ["target_direction", "target_return"]].tail(10).to_string(index=False))