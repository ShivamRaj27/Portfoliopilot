"""
build_vector_store.py
Layer 3 (Retrieval & NLP) - Chunks the downloaded filing text and embeds
each chunk using a local sentence-transformer model, then saves a simple
local vector store (numpy array of embeddings + a parallel metadata table)
that retrieve.py can search over.

Why a local embedding model instead of an API: it's free, needs no API key,
and runs fine on a laptop for a project-sized amount of text - good fit for
a student group project. Swap in an API-based embedding model later if you
want higher quality and don't mind the cost/key management.

Input:  data/{ticker}_filing_text.parquet  (from fetch_filing_text.py)
Output: data/{ticker}_vector_store.npz     (embeddings, as a numpy array)
        data/{ticker}_chunks.parquet       (parallel metadata: chunk text,
                                             ticker, form_type, filing_date,
                                             chunk_index, document_url)

Usage:
    python build_vector_store.py --tickers NKE AAPL MSFT
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_TICKERS = ["NKE"]

# Small, fast, good-enough-for-a-project embedding model (~80MB download,
# runs fine on CPU). Swap for a bigger model later if retrieval quality
# needs improving.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE_WORDS = 300     # ~300 words per chunk keeps context focused
CHUNK_OVERLAP_WORDS = 50   # overlap so a fact split across chunks isn't lost


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list:
    """Splits text into overlapping word-count-based chunks."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += step
    return chunks


def build_chunks_for_ticker(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker}_filing_text.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run fetch_filing_text.py for {ticker} first.")

    filings = pd.read_parquet(path)

    rows = []
    for _, filing in filings.iterrows():
        chunks = chunk_text(filing["text"])
        for i, chunk in enumerate(chunks):
            rows.append({
                "ticker": ticker,
                "accession_number": filing["accession_number"],
                "form_type": filing["form_type"],
                "filing_date": filing["filing_date"],
                "document_url": filing["document_url"],
                "chunk_index": i,
                "chunk_text": chunk,
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Chunk filing text and build a local vector store.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    args = parser.parse_args()

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (first run downloads it, ~80MB)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    for ticker in args.tickers:
        ticker = ticker.upper()
        try:
            chunks_df = build_chunks_for_ticker(ticker)
        except FileNotFoundError as e:
            print(f"[FAIL] {ticker}: {e}")
            continue

        if chunks_df.empty:
            print(f"[FAIL] {ticker}: no text chunks produced (empty filing text?)")
            continue

        print(f"Embedding {len(chunks_df)} chunks for {ticker}...")
        embeddings = model.encode(
            chunks_df["chunk_text"].tolist(),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # so cosine similarity = dot product
        )

        chunks_path = DATA_DIR / f"{ticker}_chunks.parquet"
        vectors_path = DATA_DIR / f"{ticker}_vector_store.npz"

        chunks_df.to_parquet(chunks_path, index=False)
        np.savez_compressed(vectors_path, embeddings=embeddings)

        print(f"[OK] {ticker}: {len(chunks_df)} chunks -> {chunks_path}")
        print(f"     embeddings {embeddings.shape} -> {vectors_path}\n")


if __name__ == "__main__":
    main()
