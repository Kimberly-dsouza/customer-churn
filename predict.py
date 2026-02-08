"""
Simple CLI script to load the saved model and (optionally) scaler and make predictions.

Feature order expected (11 features):
[CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember,
 EstimatedSalary, Geography_Germany, Geography_Spain, Gender_Male]

Usage examples (Windows PowerShell):
# single prediction by comma-separated values
python predict.py --values "619,30,3,5000,1,0,1,20000,1,0,1"

# using a CSV (first row will be used if multiple rows)
python predict.py --csv sample_input.csv

# set a custom decision threshold
python predict.py --values "..." --threshold 0.4
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

MODEL_PATHS = ["churn_predict_model", "churn_predict_model.pkl", "churn_predict_model.joblib"]
SCALER_PATHS = ["scaler.joblib", "scaler.pkl", "scaler.save", "scaler"]

FEATURE_NAMES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Geography_Germany",
    "Geography_Spain",
    "Gender_Male",
]


def find_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def load_model(path: Optional[str] = None):
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at: {path}")
        return joblib.load(path)

    p = find_existing(MODEL_PATHS)
    if not p:
        raise FileNotFoundError("No model file found (looked for: {}).".format(", ".join(MODEL_PATHS)))
    return joblib.load(p)


def load_scaler() -> Optional[object]:
    p = find_existing(SCALER_PATHS)
    if not p:
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None


def parse_values(s: str) -> np.ndarray:
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} values (got {len(parts)}). Order: {FEATURE_NAMES}")
    arr = np.array([float(x) for x in parts]).reshape(1, -1)
    return arr


def main():
    parser = argparse.ArgumentParser(description="Predict churn using saved model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--values", type=str, help="Comma-separated feature values in expected order")
    group.add_argument("--csv", type=str, help="Path to CSV file with columns matching FEATURE_NAMES or raw values in order")
    parser.add_argument("--model", type=str, help="Path to saved model (joblib). If omitted, script searches common names.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold to classify as churn")
    args = parser.parse_args()

    try:
        model = load_model(args.model)
    except Exception as e:
        print("ERROR loading model:", e)
        sys.exit(1)

    scaler = load_scaler()
    if scaler is None:
        print("NOTICE: No scaler found. Predictions will be made without input scaling.")
    else:
        print(f"Loaded scaler from disk; will apply scaling before prediction.")

    # prepare data
    if args.values:
        try:
            X = parse_values(args.values)
        except Exception as e:
            print("ERROR parsing values:", e)
            sys.exit(1)
        df = pd.DataFrame(X, columns=FEATURE_NAMES)
    else:
        if not os.path.exists(args.csv):
            print("ERROR: CSV not found:", args.csv)
            sys.exit(1)
        df = pd.read_csv(args.csv)
        # if CSV contains multiple rows, we will predict for all; if columns missing, try to interpret first row as raw values
        if set(FEATURE_NAMES).issubset(df.columns):
            df = df[FEATURE_NAMES]
        else:
            # If the CSV does not have named columns but has the right number of columns, take first row or all rows
            if df.shape[1] == len(FEATURE_NAMES):
                df.columns = FEATURE_NAMES
            else:
                print(f"CSV does not contain expected columns and does not have {len(FEATURE_NAMES)} columns.")
                print("Expected feature order:", FEATURE_NAMES)
                sys.exit(1)

    X_input = df.values.astype(float)

    if scaler is not None:
        try:
            X_input = scaler.transform(X_input)
        except Exception as e:
            print("WARNING: scaler.transform failed:", e)
            print("Proceeding without scaling.")
    # Predict
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_input)[:, 1]
        else:
            # fallback: use decision_function if available and map to pseudo-probabilities
            if hasattr(model, "decision_function"):
                scores = model.decision_function(X_input)
                # simple sigmoid mapping
                probs = 1.0 / (1.0 + np.exp(-scores))
            else:
                probs = model.predict(X_input).astype(float)
        preds = (probs >= args.threshold).astype(int)
    except Exception as e:
        print("ERROR during prediction:", e)
        sys.exit(1)

    # Output results
    out = pd.DataFrame(df, columns=FEATURE_NAMES)
    out["churn_probability"] = probs
    out["predicted_churn"] = preds
    out["recommended_action"] = out["predicted_churn"].apply(lambda x: "Offer retention" if x == 1 else "No action")

    pd.set_option("display.max_columns", None)
    print("\nPredictions:")
    print(out.to_string(index=False))

    # Summary counts
    n = len(out)
    n_churn = int(out["predicted_churn"].sum())
    print(f"\nSummary: {n} input(s). {n_churn} predicted churners (threshold={args.threshold}).")

    # Exit code 0 on success
    sys.exit(0)


if __name__ == "__main__":
    main()