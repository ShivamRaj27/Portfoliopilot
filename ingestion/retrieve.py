"""
retrieve.py
Layer 3 (Retrieval & NLP) - Given a natural-language question, embeds it
with the same model used in build_vector_store.py, and returns the top-k
most relevant filing text chunks (with source metadata) for a given ticker.

This is the function Layer 4 (Agent Orchestration) will call as one of its
tools - it returns grounded passages from actual SEC filings, which the
agent's LLM call then synthesizes into an answer (with citations back to
the filing/date/URL).

Usage (command line, for testing):
    python retrieve.py --ticker NKE --query "What are Nike's main supply chain risks?"

Usage (as an importable function, for Layer 4):
    from retrieve import retrieve_chunks
    results = retrieve_chunks("NKE", "What are Nike's main supply chain risks?", top_k=5)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # must match build_vector_store.py

_model = None  # lazy-loaded singleton so repeated calls don't reload the model


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def retrieve_chunks(ticker: str, query: str, top_k: int = 5) -> pd.DataFrame:
    """
    Returns the top_k most relevant filing text chunks for `query`, as a
    DataFrame with columns: chunk_text, form_type, filing_date,
    document_url, similarity_score (higher = more relevant, range -1 to 1).
    """
    ticker = ticker.upper()
    chunks_path = DATA_DIR / f"{ticker}_chunks.parquet"
    vectors_path = DATA_DIR / f"{ticker}_vector_store.npz"

    if not chunks_path.exists() or not vectors_path.exists():
        raise FileNotFoundError(
            f"Vector store for {ticker} not found - run build_vector_store.py --tickers {ticker} first."
        )

    chunks_df = pd.read_parquet(chunks_path)
    embeddings = np.load(vectors_path)["embeddings"]

    model = _get_model()
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    # Embeddings are pre-normalized, so cosine similarity = dot product
    scores = embeddings @ query_vec

    top_k = min(top_k, len(chunks_df))
    top_idx = np.argsort(-scores)[:top_k]

    results = chunks_df.iloc[top_idx].copy()
    results["similarity_score"] = scores[top_idx]
    return results[["chunk_text", "form_type", "filing_date", "document_url", "similarity_score"]].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Query the local filing-text vector store.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    results = retrieve_chunks(args.ticker, args.query, args.top_k)

    print(f"\nTop {len(results)} results for: \"{args.query}\" ({args.ticker})\n")
    for i, row in results.iterrows():
        print(f"--- Result {i + 1} (score={row['similarity_score']:.3f}, {row['form_type']}, {row['filing_date'].date()}) ---")
        preview = row["chunk_text"][:400].replace("\n", " ")
        print(preview + ("..." if len(row["chunk_text"]) > 400 else ""))
        print(f"Source: {row['document_url']}\n")


if __name__ == "__main__":
    main()
