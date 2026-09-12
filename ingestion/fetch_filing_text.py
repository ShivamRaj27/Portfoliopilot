"""
fetch_filing_text.py
Layer 3 (Retrieval & NLP) - Downloads the actual 10-K/10-Q document text for
each filing listed in data/{ticker}_filings_metadata.parquet (produced by
fetch_filings.py in Layer 1), strips HTML down to clean plain text, and
saves it so it can be chunked + embedded next.

Input:  data/{ticker}_filings_metadata.parquet  (ticker, accession_number,
        form_type, filing_date, document_url)
Output: data/{ticker}_filing_text.parquet
        (ticker, accession_number, form_type, filing_date, document_url, text)

Notes:
  - SEC EDGAR requires a descriptive User-Agent header, same as Layer 1.
  - By default this only pulls the N most recent filings per ticker (see
    MAX_FILINGS_PER_TICKER) - full filing text is large and you usually only
    need recent history for a research assistant demo. Raise the limit if
    your project needs deeper history.
  - Some very old filings (pre-2001) are plain text (.txt) instead of HTML;
    both are handled by the same HTML-stripping step (it's a no-op on
    already-plain text).

Usage:
    python fetch_filing_text.py --tickers NKE AAPL MSFT
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_TICKERS = ["NKE"]
DEFAULT_USER_AGENT = "Shivam Raj sh25vm@gmail.com"  # keep in sync with fetch_filings.py
REQUEST_DELAY_SECONDS = 0.3
MAX_FILINGS_PER_TICKER = 6  # most recent N filings - raise if you need more history


def get_headers() -> dict:
    user_agent = os.environ.get("SEC_USER_AGENT", DEFAULT_USER_AGENT)
    return {"User-Agent": user_agent}


def load_filings_metadata(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker}_filings_metadata.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run fetch_filings.py for {ticker} first.")
    df = pd.read_parquet(path)
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df.sort_values("filing_date", ascending=False).head(MAX_FILINGS_PER_TICKER)


def download_and_clean(url: str, headers: dict) -> str:
    """Downloads a filing document and strips it down to clean plain text."""
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.content, "html.parser")

    # Drop tags that add noise but no useful text (scripts, styles, nav)
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Collapse excessive blank lines/whitespace left over from HTML tables
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Download and clean filing text for a list of tickers.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    args = parser.parse_args()

    headers = get_headers()

    for ticker in args.tickers:
        ticker = ticker.upper()
        try:
            meta = load_filings_metadata(ticker)
        except FileNotFoundError as e:
            print(f"[FAIL] {ticker}: {e}")
            continue

        rows = []
        for _, row in meta.iterrows():
            try:
                print(f"  Fetching {ticker} {row['form_type']} filed {row['filing_date'].date()} ...")
                text = download_and_clean(row["document_url"], headers)
                rows.append({
                    "ticker": ticker,
                    "accession_number": row["accession_number"],
                    "form_type": row["form_type"],
                    "filing_date": row["filing_date"],
                    "document_url": row["document_url"],
                    "text": text,
                    "char_count": len(text),
                })
            except Exception as e:
                print(f"    !! failed: {e}")
            time.sleep(REQUEST_DELAY_SECONDS)

        if rows:
            out_df = pd.DataFrame(rows)
            out_path = DATA_DIR / f"{ticker}_filing_text.parquet"
            out_df.to_parquet(out_path, index=False)
            total_chars = out_df["char_count"].sum()
            print(f"[OK] {ticker}: {len(out_df)} filings, {total_chars:,} total chars -> {out_path}\n")
        else:
            print(f"[FAIL] {ticker}: no filings downloaded successfully\n")


if __name__ == "__main__":
    main()
