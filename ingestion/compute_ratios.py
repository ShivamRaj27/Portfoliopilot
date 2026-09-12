"""
compute_ratios.py
Layer 2 (Quant & ML) - Computes standard financial ratio families from the
XBRL data Layer 1 saved in data/{ticker}_financials.parquet.

Ratio families covered:
  - Liquidity:    current_ratio
  - Profitability: net_margin, gross_margin, operating_margin
  - Leverage:      debt_to_equity  (using Liabilities/StockholdersEquity)
  - Efficiency:    asset_turnover
  - Valuation:     (left as a placeholder - needs market cap / price data,
                    joined in separately since it's not a pure financials figure)

Input:  data/{ticker}_financials.parquet   (long format: ticker, tag, value, fiscal_year, ...)
Output: data/{ticker}_ratios.parquet       (wide format: one row per fiscal_year)

Usage:
    python compute_ratios.py --tickers NKE AAPL MSFT
"""

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_TICKERS = ["NKE"]

# Tags we need, pulled from the financials table's long format

# Tags we need, pulled from the financials table's long format
# Tags we need, pulled from the financials table's long format
REQUIRED_TAGS = [
    "Assets",
    "AssetsCurrent",
    "Liabilities",
    "LiabilitiesCurrent",
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "NetIncomeLoss",
    "GrossProfit",
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "OperatingIncomeLoss",
]

# Some concepts are reported under different XBRL tags depending on the
# company/year (e.g. Nike stopped using "Revenues" after adopting ASC 606).
# For each concept, take the first non-null value across its candidate tags,
# in priority order.
TAG_FALLBACKS = {
    "revenues": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                 "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
}


def coalesce_columns(wide: pd.DataFrame, candidate_cols: list) -> pd.Series:
    """Returns the first non-null value across a list of candidate columns, per row."""
    existing = [c for c in candidate_cols if c in wide.columns]
    if not existing:
        return pd.Series(index=wide.index, dtype="float64")
    result = wide[existing[0]].copy()
    for col in existing[1:]:
        result = result.fillna(wide[col])
    return result

# REQUIRED_TAGS = [
#     "Assets",
#     "AssetsCurrent",
#     "Liabilities",
#     "LiabilitiesCurrent",
#     "StockholdersEquity",
#     "Revenues",
#     "NetIncomeLoss",
#     "GrossProfit",
#     "OperatingIncomeLoss",
# ]


def load_financials(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker}_financials.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run fetch_filings.py for {ticker} first.")
    return pd.read_parquet(path)


def pivot_to_annual_wide(fin_long: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the long-format financials table and produces one row per fiscal
    year with each XBRL tag as a column, using only annual (10-K, fiscal
    period 'FY') figures so we're comparing like-for-like periods.
    """
    annual = fin_long[
        (fin_long["form_type"] == "10-K") & (fin_long["fiscal_period"] == "FY")
    ].copy()

    # A tag can appear more than once per fiscal year if a later filing
    # restates it - keep the most recently filed value for each (tag, fy).
    annual = annual.sort_values("filed_date").drop_duplicates(
        subset=["tag", "fiscal_year"], keep="last"
    )

    wide = annual.pivot_table(
        index="fiscal_year", columns="tag", values="value", aggfunc="last"
    )
    wide = wide.reindex(columns=REQUIRED_TAGS)  # ensure all expected columns exist, even if NaN
    wide = wide.sort_index()
    return wide


def compute_ratios(wide: pd.DataFrame) -> pd.DataFrame:
    """Computes ratio families from the wide annual financials table."""
    ratios = pd.DataFrame(index=wide.index)

    # Coalesce concepts that get reported under different tags depending on
    # the company/year (see TAG_FALLBACKS above).
    revenues = coalesce_columns(wide, TAG_FALLBACKS["revenues"])
    equity = coalesce_columns(wide, TAG_FALLBACKS["equity"])
    cost_of_revenue = coalesce_columns(wide, TAG_FALLBACKS["cost_of_revenue"])

    # Gross profit: use the reported tag if present, else derive it
    # (Revenues - Cost of Revenue) since many issuers don't tag it directly.
    gross_profit = wide["GrossProfit"]
    if "GrossProfit" not in wide.columns:
        gross_profit = pd.Series(index=wide.index, dtype="float64")
    gross_profit = gross_profit.fillna(revenues - cost_of_revenue)

    # --- Liquidity ---
    ratios["current_ratio"] = wide["AssetsCurrent"] / wide["LiabilitiesCurrent"]

    # --- Profitability ---
    ratios["net_margin"] = wide["NetIncomeLoss"] / revenues
    ratios["gross_margin"] = gross_profit / revenues
    ratios["operating_margin"] = wide["OperatingIncomeLoss"] / revenues

    # --- Leverage ---
    ratios["debt_to_equity"] = wide["Liabilities"] / equity

    # --- Efficiency ---
    ratios["asset_turnover"] = revenues / wide["Assets"]

    # --- Year-over-year change (useful signal for the agent layer) ---
    ratios["revenue_yoy_growth"] = revenues.pct_change()
    ratios["net_income_yoy_growth"] = wide["NetIncomeLoss"].pct_change()

    return ratios.round(4)

# def compute_ratios(wide: pd.DataFrame) -> pd.DataFrame:
#     """Computes ratio families from the wide annual financials table."""
#     ratios = pd.DataFrame(index=wide.index)

#     # --- Liquidity ---
#     ratios["current_ratio"] = wide["AssetsCurrent"] / wide["LiabilitiesCurrent"]

#     # --- Profitability ---
#     ratios["net_margin"] = wide["NetIncomeLoss"] / wide["Revenues"]
#     ratios["gross_margin"] = wide["GrossProfit"] / wide["Revenues"]
#     ratios["operating_margin"] = wide["OperatingIncomeLoss"] / wide["Revenues"]

#     # --- Leverage ---
#     ratios["debt_to_equity"] = wide["Liabilities"] / wide["StockholdersEquity"]

#     # --- Efficiency ---
#     ratios["asset_turnover"] = wide["Revenues"] / wide["Assets"]

#     # --- Year-over-year change (useful signal for the agent layer) ---
#     ratios["revenue_yoy_growth"] = wide["Revenues"].pct_change()
#     ratios["net_income_yoy_growth"] = wide["NetIncomeLoss"].pct_change()

#     return ratios.round(4)


def main():
    parser = argparse.ArgumentParser(description="Compute financial ratios from ingested SEC data.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    args = parser.parse_args()

    for ticker in args.tickers:
        ticker = ticker.upper()
        try:
            fin_long = load_financials(ticker)
            wide = pivot_to_annual_wide(fin_long)
            ratios = compute_ratios(wide)

            out_path = DATA_DIR / f"{ticker}_ratios.parquet"
            ratios.to_parquet(out_path)
            print(f"[OK] {ticker}: {len(ratios)} fiscal years -> {out_path}")
            print(ratios.tail(3).to_string())
            print()
        except Exception as e:
            print(f"[FAIL] {ticker}: {e}")


if __name__ == "__main__":
    main()
