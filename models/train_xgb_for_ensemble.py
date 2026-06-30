"""
models/train_xgb_for_ensemble.py — MODULE 13b: XGBoost (isolated process)
Trains XGBoost on the same windowed alignment as the LSTM and saves its
test-set predictions to disk, to be combined afterward in a third script.
Run:  python -m models.train_xgb_for_ensemble
"""

import glob
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

FEATURE_COLS = [
    "rsi", "macd", "macd_signal", "macd_diff",
    "sma_10", "sma_30", "daily_return", "volatility_10", "sentiment_score",
]
WINDOW = 10


def build_single_day_split():
    train_X, train_y, test_X, test_y = [], [], [], []
    for path in sorted(glob.glob("data/processed/*_features.csv")):
        df = pd.read_csv(path)
        n_seq = len(df) - WINDOW + 1
        cut = int(n_seq * 0.8)
        aligned = df.iloc[WINDOW - 1:].reset_index(drop=True)
        train_X.append(aligned.iloc[:cut][FEATURE_COLS])
        train_y.append(aligned.iloc[:cut]["target"])
        test_X.append(aligned.iloc[cut:][FEATURE_COLS])
        test_y.append(aligned.iloc[cut:]["target"])
    return (pd.concat(train_X), pd.concat(train_y),
            pd.concat(test_X), pd.concat(test_y))


def main():
    print("1. Building the matching single-day split (window-aligned)...")
    X_train, y_train, X_test, y_test = build_single_day_split()
    print(f"   Train rows: {len(X_train)}   Test rows: {len(X_test)}")

    print("2. Training XGBoost...")
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                           eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, (probs > 0.5).astype(int))
    print(f"\n3. XGBoost accuracy: {acc*100:.1f}%")

    np.savez("data/models/xgb_test_predictions.npz",
             probs=probs, labels=y_test.values)
    print("4. Saved XGBoost predictions to data/models/xgb_test_predictions.npz")


if __name__ == "__main__":
    main()
