"""
models/train_model.py — MODULE 7: Prediction Model (honest evaluation)
Trains and compares three classifiers (XGBoost, Random Forest, Logistic
Regression) to predict 3-day up/down movement. Uses a TIME-BASED split
(train on earlier days, test on later days, no shuffling) to avoid
look-ahead leakage from overlapping horizons.
Run:  python -m models.train_model
"""

import glob
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pickle

FEATURE_COLS = [
    "rsi", "macd", "macd_signal", "macd_diff",
    "sma_10", "sma_30", "daily_return", "volatility_10",
    "sentiment_score",
]


def load_all_features():
    """Load each ticker's table, keeping its time order, then stack.
    We split each ticker by time so the test set is always 'later' days."""
    train_frames, test_frames = [], []
    for path in sorted(glob.glob("data/processed/*_features.csv")):
        df = pd.read_csv(path)
        cut = int(len(df) * 0.8)          # first 80% = train, last 20% = test
        train_frames.append(df.iloc[:cut])
        test_frames.append(df.iloc[cut:])
    if not train_frames:
        return None, None
    return pd.concat(train_frames, ignore_index=True), pd.concat(test_frames, ignore_index=True)


def main():
    print("1. Loading features with a TIME-BASED split (no shuffle)...")
    train_df, test_df = load_all_features()
    if train_df is None:
        print("   No feature files found — run the feature builder first.")
        return

    X_train, y_train = train_df[FEATURE_COLS], train_df["target"]
    X_test,  y_test  = test_df[FEATURE_COLS],  test_df["target"]
    print(f"   Train: {len(X_train)} earlier rows   Test: {len(X_test)} later rows")

    # Baseline: always guess the majority class from training
    majority = int(y_train.mode()[0])
    baseline = accuracy_score(y_test, [majority] * len(y_test))
    print(f"\n   Baseline accuracy: {baseline*100:.1f}%  (always guess {'up' if majority==1 else 'down'})")

    # Scale features (helps Logistic Regression; harmless for the others)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            eval_metric="logloss", random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=5, random_state=42,
        ),
        "LogisticRegression": LogisticRegression(max_iter=1000),
    }

    print("\n2. Training and comparing models...\n")
    results = {}
    best_name, best_acc, best_model = None, 0, None
    for name, model in models.items():
        if name == "LogisticRegression":
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        print(f"   {name:>20}: {acc*100:.1f}%")
        if acc > best_acc:
            best_name, best_acc, best_model = name, acc, model

    print(f"\n   {'='*45}")
    print(f"   Best model: {best_name} at {best_acc*100:.1f}%")
    print(f"   Baseline:   {baseline*100:.1f}%")
    diff = (best_acc - baseline) * 100
    print(f"   {'beats baseline by' if diff>0 else 'below baseline by'} {abs(diff):.1f} points")
    print(f"   {'='*45}")

    # Detailed report for the best model
    if best_name == "LogisticRegression":
        best_preds = best_model.predict(X_test_s)
    else:
        best_preds = best_model.predict(X_test)
    print(f"\n   Detailed report ({best_name}):")
    print(classification_report(y_test, best_preds, target_names=["Down", "Up"], zero_division=0))

    # Save the best model
    with open("data/models/best_model.pkl", "wb") as f:
        pickle.dump({"model": best_model, "name": best_name, "scaler": scaler,
                     "features": FEATURE_COLS}, f)
    print(f"3. Saved best model ({best_name}) to data/models/best_model.pkl")


if __name__ == "__main__":
    main()
