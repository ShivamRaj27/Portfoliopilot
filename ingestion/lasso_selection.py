"""
lasso_selection.py
Layer 2 (Quant & ML) - Uses Lasso (L1-regularized linear regression) to
predict next-day RETURN MAGNITUDE (a continuous value, not up/down), and -
more importantly - to answer "which features actually matter?"

Lasso's L1 penalty shrinks unhelpful features' coefficients all the way to
EXACTLY ZERO (unlike Ridge/plain linear regression, which only shrinks them
smaller). A feature with a zero coefficient is one Lasso decided, at the
chosen regularization strength, contributes nothing useful once the other
features are accounted for - this is genuine, principled feature selection,
not just "look at the biggest coefficient."

We sweep several regularization strengths (alpha) via cross-validation
(LassoCV) rather than picking one arbitrarily, and report R^2 honestly -
next-day return magnitude is very hard to predict, so a low R^2 (even
negative, meaning "worse than predicting the mean") is a realistic, reportable
result, not a sign something is broken.

Usage:
    python lasso_selection.py --ticker NKE
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from feature_engineering import FEATURE_COLUMNS, build_feature_matrix, chronological_split

DATA_DIR = Path(__file__).parent / "data"


def run_lasso(ticker: str):
    df = build_feature_matrix(ticker)
    train, test = chronological_split(df, test_size=0.2)

    X_train, y_train = train[FEATURE_COLUMNS], train["target_return"]
    X_test, y_test = test[FEATURE_COLUMNS], test["target_return"]

    # Lasso is scale-sensitive (like Logistic Regression) - features must be
    # standardized or the penalty unfairly hits large-magnitude features harder.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # LassoCV tries a range of alpha (regularization strength) values using
    # cross-validation on the training set, and picks the best one - rather
    # than us guessing a single alpha.
    model = LassoCV(cv=5, random_state=42, max_iter=10000)
    model.fit(X_train_scaled, y_train)

    pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)

    coefs = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "coefficient": model.coef_,
    })
    coefs["abs_coefficient"] = coefs["coefficient"].abs()
    coefs = coefs.sort_values("abs_coefficient", ascending=False)

    kept = coefs[coefs["coefficient"] != 0]
    dropped = coefs[coefs["coefficient"] == 0]

    return {
        "alpha": model.alpha_,
        "r2": r2,
        "mae": mae,
        "coefficients": coefs,
        "kept_features": kept["feature"].tolist(),
        "dropped_features": dropped["feature"].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Lasso regression feature selection.")
    parser.add_argument("--ticker", default="NKE")
    args = parser.parse_args()
    ticker = args.ticker.upper()

    result = run_lasso(ticker)

    print(f"{ticker} - Lasso Regression (predicting next-day return magnitude)")
    print(f"Selected alpha (via 5-fold CV): {result['alpha']:.6f}")
    print(f"Test R^2: {result['r2']:.4f}   Test MAE: {result['mae']:.4f}")
    print(
        "\nNote: R^2 near zero (or negative) is expected and honest here - "
        "next-day return magnitude is close to unpredictable from these "
        "features alone. The useful output of this script is feature "
        "SELECTION, not prediction accuracy.\n"
    )

    print("=== Feature coefficients (sorted by |coefficient|) ===")
    print(result["coefficients"].to_string(index=False))

    print(f"\nFeatures Lasso KEPT (non-zero coefficient): {result['kept_features']}")
    print(f"Features Lasso DROPPED (shrunk to exactly zero): {result['dropped_features']}")

    out_path = DATA_DIR / f"{ticker}_lasso_coefficients.parquet"
    result["coefficients"].to_parquet(out_path, index=False)
    print(f"\nSaved coefficients -> {out_path}")


if __name__ == "__main__":
    main()
