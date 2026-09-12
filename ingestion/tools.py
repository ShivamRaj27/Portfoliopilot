

"""
tools.py
Layer 4 (Agent Orchestration) - Thin wrapper functions around everything
built in Layers 1-3, packaged as simple, JSON-serializable functions the
agent can call as "tools." Each function reads local Parquet files only -
no live external API calls happen at answer-time (per your brief's design:
data is pre-fetched on a schedule, the live app never calls external APIs
in front of a user).

These wrappers are intentionally dependency-light so agent.py can import
just this file without pulling in sentence-transformers unless a filings
search is actually requested.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def _sanitize_for_json(obj):
    """
    Recursively replaces NaN/inf with None so the result is valid strict JSON
    (Python's json.dumps allows literal NaN by default, but many APIs -
    including Gemini's - reject it as invalid JSON).
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def get_ratios(ticker: str, num_years: int = 3) -> dict:
    """Returns the most recent `num_years` of computed financial ratios for a ticker."""
    ticker = ticker.upper()
    path = DATA_DIR / f"{ticker}_ratios.parquet"
    if not path.exists():
        return {"error": f"No ratios data found for {ticker}. Run compute_ratios.py first."}

    df = pd.read_parquet(path).tail(num_years)
    result = {
        "ticker": ticker,
        "years": df.index.tolist(),
        "ratios": df.reset_index().to_dict(orient="records"),
    }
    return _sanitize_for_json(result)


def get_price_signals(ticker: str) -> dict:
    """Returns the latest price trend/volatility/RSI signals and recent anomalies for a ticker."""
    ticker = ticker.upper()
    path = DATA_DIR / f"{ticker}_price_signals.parquet"
    if not path.exists():
        return {"error": f"No price signals found for {ticker}. Run price_signals.py first."}

    df = pd.read_parquet(path)
    latest = df.iloc[-1]
    anomalies = df[df["is_anomalous"]].tail(5)

    return _sanitize_for_json({
        "ticker": ticker,
        "as_of_date": str(latest["date"].date()),
        "latest_close": round(float(latest["close"]), 2),
        "trend_signal": latest["trend_signal"],
        "volatility_20d": round(float(latest["volatility_20d"]), 4) if pd.notna(latest["volatility_20d"]) else None,
        "rsi_14": round(float(latest["rsi_14"]), 1) if pd.notna(latest["rsi_14"]) else None,
        "recent_anomalies": [
            {
                "date": str(row["date"].date()),
                "close": round(float(row["close"]), 2),
                "daily_return": round(float(row["daily_return"]), 4),
                "return_zscore": round(float(row["return_zscore"]), 2),
            }
            for _, row in anomalies.iterrows()
        ],
    })


def search_filings(ticker: str, query: str, top_k: int = 5) -> dict:
    """
    Searches the ticker's SEC filing text for passages relevant to `query`.
    Imports retrieve.py lazily since it pulls in sentence-transformers,
    which is only needed when this specific tool is actually called.
    """
    from retrieve import retrieve_chunks  # local import - see docstring above

    ticker = ticker.upper()
    try:
        results = retrieve_chunks(ticker, query, top_k=top_k)
    except FileNotFoundError as e:
        return {"error": str(e)}

    return _sanitize_for_json({
        "ticker": ticker,
        "query": query,
        "passages": [
            {
                "text": row["chunk_text"],
                "form_type": row["form_type"],
                "filing_date": str(row["filing_date"].date()),
                "source_url": row["document_url"],
                "similarity_score": round(float(row["similarity_score"]), 3),
            }
            for _, row in results.iterrows()
        ],
    })


# Tool schemas in the format the Anthropic API expects (used by agent.py).
TOOL_SCHEMAS = [
    {
        "name": "get_ratios",
        "description": "Get recent computed financial ratios (liquidity, profitability, leverage, "
                        "efficiency, growth) for a stock ticker, derived from its SEC filings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. NKE"},
                "num_years": {"type": "integer", "description": "How many recent fiscal years to return", "default": 3},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_price_signals",
        "description": "Get the latest stock price trend (bullish/bearish), volatility, RSI, and any "
                        "recent anomalous single-day price moves for a ticker.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. NKE"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "search_filings",
        "description": "Search a company's actual SEC 10-K/10-Q filing text for passages relevant to a "
                        "question (e.g. risk factors, strategy, litigation). Returns grounded quotes with sources.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. NKE"},
                "query": {"type": "string", "description": "What to search for in the filing text"},
                "top_k": {"type": "integer", "description": "Number of passages to return", "default": 5},
            },
            "required": ["ticker", "query"],
        },
    },
]

# Maps tool name -> actual Python function, used by agent.py to execute tool calls.
TOOL_FUNCTIONS = {
    "get_ratios": get_ratios,
    "get_price_signals": get_price_signals,
    "search_filings": search_filings,
}
