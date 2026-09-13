

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
    """
    Returns the latest price trend/volatility/RSI signals for a ticker, plus
    recent anomalous days flagged by TWO independent methods:
      - zscore method: rule-based, |return z-score| > threshold
      - ml method:      Isolation Forest, trained on [daily_return,
                         volatility_20d, rsi_14] (see price_signals.py)
    Both are returned side by side so it's clear where they agree/disagree -
    useful for showing the ML anomaly detector isn't just duplicating the
    simple rule-based one.
    """
    ticker = ticker.upper()
    path = DATA_DIR / f"{ticker}_price_signals.parquet"
    if not path.exists():
        return {"error": f"No price signals found for {ticker}. Run price_signals.py first."}

    df = pd.read_parquet(path)
    latest = df.iloc[-1]
    zscore_anomalies = df[df["is_anomalous"]].tail(5)
    ml_anomalies = df[df["is_ml_anomalous"]].tail(5) if "is_ml_anomalous" in df.columns else df.iloc[0:0]

    return _sanitize_for_json({
        "ticker": ticker,
        "as_of_date": str(latest["date"].date()),
        "latest_close": round(float(latest["close"]), 2),
        "trend_signal": latest["trend_signal"],
        "volatility_20d": round(float(latest["volatility_20d"]), 4) if pd.notna(latest["volatility_20d"]) else None,
        "rsi_14": round(float(latest["rsi_14"]), 1) if pd.notna(latest["rsi_14"]) else None,
        "recent_anomalies_zscore_method": [
            {
                "date": str(row["date"].date()),
                "close": round(float(row["close"]), 2),
                "daily_return": round(float(row["daily_return"]), 4),
                "return_zscore": round(float(row["return_zscore"]), 2),
            }
            for _, row in zscore_anomalies.iterrows()
        ],
        "recent_anomalies_ml_method": [
            {
                "date": str(row["date"].date()),
                "close": round(float(row["close"]), 2),
                "daily_return": round(float(row["daily_return"]), 4),
                "ml_anomaly_score": round(float(row["ml_anomaly_score"]), 4) if pd.notna(row["ml_anomaly_score"]) else None,
            }
            for _, row in ml_anomalies.iterrows()
        ],
    })


def predict_direction(ticker: str) -> dict:
    """
    Uses the trained Random Forest / XGBoost / Logistic Regression models
    (from train_classifiers.py) to predict tomorrow's price direction from
    today's features.

    IMPORTANT - these models are only slightly better than a coin flip
    (test accuracy typically 45-55%, ROC-AUC just above 0.5). This is a
    known, honest limitation of next-day direction prediction in efficient
    markets, not a bug. The agent should present this as a weak signal
    alongside other evidence, never as a confident forecast.

    Imports feature_engineering / pickle lazily since they're only needed
    when this specific tool is actually called.
    """
    import pickle
    from feature_engineering import FEATURE_COLUMNS, get_latest_feature_row

    ticker = ticker.upper()
    model_path = Path(__file__).parent / "models" / f"{ticker}_classifiers.pkl"
    if not model_path.exists():
        return {"error": f"No trained classifier found for {ticker}. Run train_classifiers.py first."}

    try:
        latest = get_latest_feature_row(ticker)
    except (FileNotFoundError, ValueError) as e:
        return {"error": str(e)}

    with open(model_path, "rb") as f:
        models = pickle.load(f)

    X = latest[FEATURE_COLUMNS].astype(float).to_frame().T

    predictions = {}

    rf = models["random_forest"]["model"]
    predictions["random_forest"] = {
        "predicted_direction": "up" if rf.predict(X)[0] == 1 else "down",
        "probability_up": round(float(rf.predict_proba(X)[0][1]), 4),
    }

    xgb = models["xgboost"]["model"]
    predictions["xgboost"] = {
        "predicted_direction": "up" if xgb.predict(X)[0] == 1 else "down",
        "probability_up": round(float(xgb.predict_proba(X)[0][1]), 4),
    }

    logreg_bundle = models["logistic_regression"]
    X_scaled = logreg_bundle["scaler"].transform(X)
    logreg = logreg_bundle["model"]
    predictions["logistic_regression"] = {
        "predicted_direction": "up" if logreg.predict(X_scaled)[0] == 1 else "down",
        "probability_up": round(float(logreg.predict_proba(X_scaled)[0][1]), 4),
    }

    return _sanitize_for_json({
        "ticker": ticker,
        "as_of_date": str(latest["date"].date()),
        "predictions": predictions,
        "caveat": ("Next-day direction prediction is close to a coin flip in efficient "
                   "markets. These models typically score 45-55% test accuracy - treat "
                   "this as a weak signal, not a confident forecast."),
    })


def get_lasso_findings(ticker: str) -> dict:
    """
    Returns which features Lasso regression kept (non-zero coefficient) vs
    dropped (shrunk to exactly zero) when predicting next-day return
    magnitude for a ticker - i.e. genuine feature selection, not a
    prediction. See lasso_selection.py for the training/evaluation code.
    """
    ticker = ticker.upper()
    path = DATA_DIR / f"{ticker}_lasso_coefficients.parquet"
    if not path.exists():
        return {"error": f"No Lasso results found for {ticker}. Run lasso_selection.py first."}

    df = pd.read_parquet(path)
    kept = df[df["coefficient"] != 0].sort_values("abs_coefficient", ascending=False)
    dropped = df[df["coefficient"] == 0]["feature"].tolist()

    return _sanitize_for_json({
        "ticker": ticker,
        "features_kept": kept.to_dict(orient="records"),
        "features_dropped": dropped,
        "note": ("An empty 'features_kept' list means Lasso found no reliable linear "
                 "signal in ANY feature for predicting next-day return magnitude - a "
                 "legitimate, honest result for this kind of prediction task, not an error."),
    })


def get_cluster(ticker: str) -> dict:
    """
    Returns which financial-profile cluster a ticker belongs to (from
    cluster_companies.py's K-Means output), which other tickers share that
    cluster, and that cluster's average ratio profile.

    IMPORTANT - honest limitation: with only a handful of tickers in this
    project, clustering is only weakly informative (see cluster_companies.py
    docstring). The agent should mention this caveat when discussing results.
    """
    ticker = ticker.upper()
    path = DATA_DIR / "company_clusters.parquet"
    if not path.exists():
        return {"error": "No cluster data found. Run cluster_companies.py first."}

    df = pd.read_parquet(path)
    if ticker not in df["ticker"].values:
        return {"error": f"{ticker} was not included in the last cluster_companies.py run. "
                          f"Available tickers: {df['ticker'].tolist()}"}

    from cluster_companies import CLUSTER_FEATURES

    row = df[df["ticker"] == ticker].iloc[0]
    cluster_id = int(row["cluster"])

    same_cluster = df[df["cluster"] == cluster_id]
    cluster_profile = same_cluster[CLUSTER_FEATURES].mean().round(4).to_dict()

    return _sanitize_for_json({
        "ticker": ticker,
        "cluster_id": cluster_id,
        "cluster_members": same_cluster["ticker"].tolist(),
        "cluster_average_profile": cluster_profile,
        "total_tickers_considered": len(df),
        "caveat": ("With few tickers in this project, clustering is only weakly informative - "
                   "results become more meaningful with a broader set of companies (10+ "
                   "recommended). Treat cluster membership as a rough grouping, not a precise "
                   "classification.") if len(df) < 4 else None,
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
        "name": "predict_direction",
        "description": "Predict whether a stock's price will go up or down tomorrow, using trained "
                        "Random Forest, XGBoost, and Logistic Regression models on technical and "
                        "fundamental features. NOTE: this is a weak signal, close to a coin flip - "
                        "typical test accuracy is only 45-55%. Present with appropriate caution.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. NKE"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_lasso_findings",
        "description": "Get which financial/technical features Lasso regression found to have real "
                        "linear predictive signal (kept) vs none (dropped to zero) for next-day return "
                        "magnitude. Useful for explaining which factors matter, not for forecasting.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. NKE"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_cluster",
        "description": "Get which peer group (financial-profile cluster) a company belongs to, based "
                        "on K-Means clustering of its ratios (margins, leverage, efficiency, growth) - "
                        "not sector labels. Returns cluster members and the cluster's average profile.",
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
    "predict_direction": predict_direction,
    "get_lasso_findings": get_lasso_findings,
    "get_cluster": get_cluster,
    "search_filings": search_filings,
}
