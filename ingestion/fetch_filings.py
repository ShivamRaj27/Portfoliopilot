"""
Layer 1 - Data ingestion: SEC filings + XBRL financials

Pulls, for each ticker:
  1. XBRL-tagged financial figures (revenue, assets, liabilities, etc.)
     from SEC's companyfacts API
  2. 10-K / 10-Q filing metadata (accession number, filing date, document URL)
     from SEC's submissions API

...and writes tidy Parquet tables to data/. Designed to be run on a
schedule (GitHub Actions cron), never live in front of a user.

IMPORTANT: SEC EDGAR requires every request to send a descriptive
User-Agent header identifying you (name + email), or it will block you.

This is set below via DEFAULT_USER_AGENT, so the script works out of the
box. You can still override it per-machine with an environment variable
if you ever need to (e.g. a teammate running this with their own info):

    export SEC_USER_AGENT="Your Name your.email@example.com"
    python fetch_filings.py --tickers NKE

Usage:
    python fetch_filings.py --tickers NKE AAPL MSFT
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
TICKER_LOOKUP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# Used if the SEC_USER_AGENT environment variable isn't set. Change this
# to your own name/email if someone else on the team runs this script.
DEFAULT_USER_AGENT = "Shivam Raj sh25vm@gmail.com"

# The XBRL tags that feed the ratios in Layer 2 (liquidity, profitability,
# leverage, valuation, efficiency). Extend this list as Layer 2 needs more.
#
# NOTE: several concepts have more than one possible tag because companies
# changed which tag they use over time (most notably: many issuers stopped
# tagging plain "Revenues" after adopting ASC 606 in 2018 and switched to
# "RevenueFromContractWithCustomerExcludingAssessedTax" instead). We pull
# every variant here; compute_ratios.py coalesces across them per concept.
XBRL_TAGS_OF_INTEREST = [
    "Assets",
    "AssetsCurrent",
    "Liabilities",
    "LiabilitiesCurrent",
    "LiabilitiesNoncurrent",
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
    "CashAndCashEquivalentsAtCarryingValue",
    "InventoryNet",
    "EarningsPerShareDiluted",
]

DEFAULT_TICKERS = ["NKE"]
REQUEST_DELAY_SECONDS = 0.15  # keeps well under SEC's ~10 req/sec limit


def get_headers() -> dict:
    user_agent = os.environ.get("SEC_USER_AGENT", DEFAULT_USER_AGENT)
    return {"User-Agent": user_agent}


def load_ticker_to_cik_map(headers: dict) -> dict:
    """Returns {'NKE': '0000320187', ...} - CIKs zero-padded to 10 digits."""
    resp = requests.get(TICKER_LOOKUP_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    raw = resp.json()  # dict of {"0": {"cik_str": ..., "ticker": ..., "title": ...}, ...}

    mapping = {}
    for entry in raw.values():
        ticker = entry["ticker"].upper()
        cik = str(entry["cik_str"]).zfill(10)
        mapping[ticker] = cik
    return mapping


def fetch_companyfacts(cik: str, headers: dict) -> dict:
    url = COMPANYFACTS_URL.format(cik=cik)
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_submissions(cik: str, headers: dict) -> dict:
    url = SUBMISSIONS_URL.format(cik=cik)
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def flatten_financials(ticker: str, facts_json: dict) -> pd.DataFrame:
    """
    Flattens the nested companyfacts JSON into a tidy table:
        ticker, tag, unit, fiscal_year, fiscal_period, value, filed_date, form_type
    Only keeps tags in XBRL_TAGS_OF_INTEREST and the 'us-gaap' taxonomy.
    """
    rows = []
    us_gaap = facts_json.get("facts", {}).get("us-gaap", {})

    for tag in XBRL_TAGS_OF_INTEREST:
        tag_data = us_gaap.get(tag)
        if not tag_data:
            continue  # company may not report this tag - that's fine, skip it

        for unit, entries in tag_data.get("units", {}).items():
            for entry in entries:
                # Only keep annual (10-K) and quarterly (10-Q) figures, skip
                # amendments/other forms to avoid double-counting restatements.
                if entry.get("form") not in ("10-K", "10-Q"):
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "tag": tag,
                        "unit": unit,
                        "fiscal_year": entry.get("fy"),
                        "fiscal_period": entry.get("fp"),
                        "value": entry.get("val"),
                        "filed_date": entry.get("filed"),
                        "period_end": entry.get("end"),
                        "form_type": entry.get("form"),
                        "accession_number": entry.get("accn"),
                    }
                )

    return pd.DataFrame(rows)


def flatten_filings_metadata(ticker: str, cik: str, submissions_json: dict) -> pd.DataFrame:
    """
    Flattens the submissions JSON's 'recent' filings block into a tidy table
    of just 10-K / 10-Q filings, with a ready-to-use document URL for Layer 3
    (retrieval / RAG) to fetch the actual filing text from.
        ticker, accession_number, form_type, filing_date, document_url
    """
    recent = submissions_json.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    cik_no_padding = str(int(cik))  # SEC document URLs use the CIK without zero-padding

    rows = []
    for form, accn, filed, primary_doc in zip(forms, accessions, filing_dates, primary_docs):
        if form not in ("10-K", "10-Q"):
            continue
        accn_no_dashes = accn.replace("-", "")
        document_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_no_padding}/{accn_no_dashes}/{primary_doc}"
        )
        rows.append(
            {
                "ticker": ticker,
                "accession_number": accn,
                "form_type": form,
                "filing_date": filed,
                "document_url": document_url,
            }
        )

    return pd.DataFrame(rows)


def process_ticker(ticker: str, cik: str, headers: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    time.sleep(REQUEST_DELAY_SECONDS)
    facts_json = fetch_companyfacts(cik, headers)
    financials_df = flatten_financials(ticker, facts_json)

    time.sleep(REQUEST_DELAY_SECONDS)
    submissions_json = fetch_submissions(cik, headers)
    filings_df = flatten_filings_metadata(ticker, cik, submissions_json)

    return financials_df, filings_df


def main():
    parser = argparse.ArgumentParser(description="Fetch SEC financials and filing metadata for a list of tickers.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="Ticker symbols, e.g. NKE AAPL MSFT")
    args = parser.parse_args()

    headers = get_headers()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Looking up CIKs...")
    ticker_to_cik = load_ticker_to_cik_map(headers)

    failures = []
    for ticker in args.tickers:
        ticker = ticker.upper()
        cik = ticker_to_cik.get(ticker)
        if not cik:
            print(f"  !! '{ticker}' not found in SEC ticker list, skipping")
            failures.append(ticker)
            continue

        try:
            print(f"Fetching SEC data for {ticker} (CIK {cik}) ...")
            financials_df, filings_df = process_ticker(ticker, cik, headers)

            financials_path = DATA_DIR / f"{ticker}_financials.parquet"
            filings_path = DATA_DIR / f"{ticker}_filings_metadata.parquet"
            financials_df.to_parquet(financials_path, index=False)
            filings_df.to_parquet(filings_path, index=False)

            print(f"  -> {len(financials_df)} financial data points -> {financials_path}")
            print(f"  -> {len(filings_df)} filings (10-K/10-Q) -> {filings_path}")
        except Exception as e:
            print(f"  !! failed for {ticker}: {e}")
            failures.append(ticker)

    if failures:
        print(f"\nCompleted with failures: {failures}", file=sys.stderr)
        sys.exit(1)

    print("\nAll tickers fetched successfully.")


if __name__ == "__main__":
    main()
