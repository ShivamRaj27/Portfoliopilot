"""
cluster_companies.py
Layer 2 (Quant & ML) - Unsupervised task: groups companies by their latest
financial ratio profile using K-Means, so you can answer "which companies
currently look financially similar to X?" - not based on sector/industry
labels, but purely on their actual numbers (margins, leverage, efficiency).

IMPORTANT - honest limitation: K-Means needs enough data points to produce
meaningful clusters. With only 2-3 tickers (as in this project's default
setup), clustering will trivially separate them without demonstrating much -
this script works correctly regardless of ticker count, but the RESULT is
only genuinely informative once you've run fetch_filings.py /
compute_ratios.py for a broader set of tickers (10+ is a reasonable target
for a course project demo).

Usage:
    python cluster_companies.py --tickers NKE AAPL MSFT ... (as many as you have)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent / "data"

CLUSTER_FEATURES = [
    "current_ratio", "net_margin", "gross_margin", "operating_margin",
    "debt_to_equity", "asset_turnover", "revenue_yoy_growth",
]


def load_latest_ratios(tickers: list) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        ticker = ticker.upper()
        path = DATA_DIR / f"{ticker}_ratios.parquet"
        if not path.exists():
            print(f"  [skip] {ticker}: no ratios file found - run compute_ratios.py first.")
            continue
        df = pd.read_parquet(path)
        if df.empty:
            print(f"  [skip] {ticker}: ratios file is empty.")
            continue
        latest = df.iloc[-1].copy()
        latest["ticker"] = ticker
        rows.append(latest)

    if not rows:
        raise ValueError("No usable ratio data found for any of the given tickers.")

    return pd.DataFrame(rows).reset_index(drop=True)


def cluster(df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    features = df[CLUSTER_FEATURES].copy()

    # Fill any missing ratio values with that feature's median across the
    # companies we DO have - simple, transparent, and avoids dropping a
    # whole company just because one ratio is missing.
    features = features.fillna(features.median())

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    out = df.copy()
    out["cluster"] = labels
    return out


def describe_clusters(clustered: pd.DataFrame):
    print("\n=== Cluster membership ===")
    for c in sorted(clustered["cluster"].unique()):
        members = clustered[clustered["cluster"] == c]["ticker"].tolist()
        print(f"  Cluster {c}: {members}")

    print("\n=== Cluster average profile ===")
    profile = clustered.groupby("cluster")[CLUSTER_FEATURES].mean().round(3)
    print(profile.to_string())


def main():
    parser = argparse.ArgumentParser(description="Cluster companies by financial ratio profile (K-Means).")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--n_clusters", type=int, default=None,
                         help="Number of clusters. Defaults to min(3, num_tickers - 1), at least 1.")
    args = parser.parse_args()

    print("Loading latest ratios...")
    df = load_latest_ratios(args.tickers)
    print(f"Loaded {len(df)} companies: {df['ticker'].tolist()}")

    n_clusters = args.n_clusters or max(1, min(3, len(df) - 1))
    if len(df) < 4:
        print(
            f"\nNote: only {len(df)} companies available - clustering into {n_clusters} "
            f"group(s) will be somewhat trivial. Results become genuinely useful once "
            f"you have data for a broader set of tickers (10+ recommended)."
        )

    clustered = cluster(df, n_clusters)
    describe_clusters(clustered)

    out_path = DATA_DIR / "company_clusters.parquet"
    clustered.to_parquet(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
