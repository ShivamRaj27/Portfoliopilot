"""
train_classifiers.py
Layer 2 (Quant & ML) - Trains and compares three classifiers predicting
next-day price direction (up/down) for a ticker:
  1. Logistic Regression - linear baseline, needs scaled features
  2. Random Forest        - non-linear, captures feature interactions
  3. XGBoost               - gradient-boosted trees, usually the strongest

This is a genuine supervised learning task: real train/test split (by DATE,
not randomly - see feature_engineering.chronological_split), real evaluation
metrics, and an honest comparison of a simple baseline against two more
complex models.

IMPORTANT - set expectations correctly: next-day stock direction is known to
be very close to a coin flip in efficient markets. An accuracy near 50-55%
is a REALISTIC and DEFENSIBLE result, not a failure - reporting this
honestly is better than overclaiming.

Usage:
    python train_classifiers.py --ticker NKE
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from feature_engineering import FEATURE_COLUMNS, build_feature_matrix, chronological_split

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"


def evaluate(name: str, y_true, y_pred, y_proba) -> dict:
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def train_and_compare(ticker: str) -> tuple:
    df = build_feature_matrix(ticker)
    train, test = chronological_split(df, test_size=0.2)

    X_train, y_train = train[FEATURE_COLUMNS], train["target_direction"]
    X_test, y_test = test[FEATURE_COLUMNS], test["target_direction"]

    print(f"{ticker}: {len(train)} train rows ({train['date'].min().date()} to {train['date'].max().date()}), "
          f"{len(test)} test rows ({test['date'].min().date()} to {test['date'].max().date()})")
    print(f"Base rate (always predict 'up'): {y_test.mean():.4f} of test days actually went up\n")

    results = []
    models = {}

    # --- 1. Logistic Regression (needs scaled features; linear baseline) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    pred = logreg.predict(X_test_scaled)
    proba = logreg.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate("Logistic Regression", y_test, pred, proba))
    models["logistic_regression"] = {"model": logreg, "scaler": scaler}

    # --- 2. Random Forest (no scaling needed; handles non-linearity) ---
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    proba = rf.predict_proba(X_test)[:, 1]
    results.append(evaluate("Random Forest", y_test, pred, proba))
    models["random_forest"] = {"model": rf}

    # --- 3. XGBoost (gradient boosting; usually strongest on tabular data) ---
    xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                         random_state=42, eval_metric="logloss")
    xgb.fit(X_train, y_train)
    pred = xgb.predict(X_test)
    proba = xgb.predict_proba(X_test)[:, 1]
    results.append(evaluate("XGBoost", y_test, pred, proba))
    models["xgboost"] = {"model": xgb}

    results_df = pd.DataFrame(results)

    # Feature importances from the tree models (Logistic Regression's
    # coefficients aren't directly comparable in magnitude without more
    # care, so importances here come from RF/XGBoost).
    importances = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "random_forest_importance": rf.feature_importances_,
        "xgboost_importance": xgb.feature_importances_,
    }).sort_values("xgboost_importance", ascending=False)

    return results_df, importances, models


def main():
    parser = argparse.ArgumentParser(description="Train and compare price-direction classifiers.")
    parser.add_argument("--ticker", default="NKE")
    args = parser.parse_args()
    ticker = args.ticker.upper()

    results_df, importances, models = train_and_compare(ticker)

    print("=== Model comparison (test set) ===")
    print(results_df.to_string(index=False))
    print()
    print("=== Feature importances (tree models) ===")
    print(importances.to_string(index=False))

    MODEL_DIR.mkdir(exist_ok=True)
    out_path = MODEL_DIR / f"{ticker}_classifiers.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(models, f)
    print(f"\nSaved trained models -> {out_path}")

    results_path = DATA_DIR / f"{ticker}_classifier_results.parquet"
    results_df.to_parquet(results_path, index=False)
    print(f"Saved comparison table -> {results_path}")


if __name__ == "__main__":
    main()
