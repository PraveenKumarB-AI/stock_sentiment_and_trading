"""
models/train_model.py — MODULE 7 & 11: Prediction Model + MLflow Tracking
Trains and compares three classifiers (XGBoost, Random Forest, Logistic
Regression) to predict 3-day up/down movement, using a TIME-BASED split.
Each model's hyperparameters and metrics are logged to MLflow as a separate
run, so results can be compared in the MLflow UI. The best model is still
saved to data/models/best_model.pkl for the dashboard to use.
Run:           python -m models.train_model
View results:  mlflow ui --port 5001   (then open http://127.0.0.1:5001)
"""

import glob
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURE_COLS = [
    "rsi", "macd", "macd_signal", "macd_diff",
    "sma_10", "sma_30", "daily_return", "volatility_10",
    "sentiment_score",
]


def load_all_features():
    train_frames, test_frames = [], []
    for path in sorted(glob.glob("data/processed/*_features.csv")):
        df = pd.read_csv(path)
        cut = int(len(df) * 0.8)
        train_frames.append(df.iloc[:cut])
        test_frames.append(df.iloc[cut:])
    if not train_frames:
        return None, None
    return pd.concat(train_frames, ignore_index=True), pd.concat(test_frames, ignore_index=True)


def main():
    mlflow.set_experiment("stock_prediction_models")

    print("1. Loading features with a TIME-BASED split (no shuffle)...")
    train_df, test_df = load_all_features()
    if train_df is None:
        print("   No feature files found — run the feature builder first.")
        return

    X_train, y_train = train_df[FEATURE_COLS], train_df["target"]
    X_test,  y_test  = test_df[FEATURE_COLS],  test_df["target"]
    print(f"   Train: {len(X_train)} earlier rows   Test: {len(X_test)} later rows")

    majority = int(y_train.mode()[0])
    baseline = accuracy_score(y_test, [majority] * len(y_test))
    print(f"\n   Baseline accuracy: {baseline*100:.1f}%  (always guess {'up' if majority==1 else 'down'})")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model_configs = {
        "XGBoost": {
            "model": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                    eval_metric="logloss", random_state=42),
            "params": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
            "scaled": False,
        },
        "RandomForest": {
            "model": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
            "params": {"n_estimators": 200, "max_depth": 5},
            "scaled": False,
        },
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000),
            "params": {"max_iter": 1000},
            "scaled": True,
        },
    }

    print("\n2. Training, comparing, and logging each model to MLflow...\n")
    best_name, best_acc, best_model = None, 0, None

    for name, cfg in model_configs.items():
        with mlflow.start_run(run_name=name):
            model = cfg["model"]
            if cfg["scaled"]:
                model.fit(X_train_s, y_train)
                preds = model.predict(X_test_s)
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)

            # Log what we ran (params) and what we got (metrics)
            mlflow.log_param("model_type", name)
            for p_name, p_val in cfg["params"].items():
                mlflow.log_param(p_name, p_val)
            mlflow.log_param("horizon_days", 3)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("baseline_accuracy", baseline)
            mlflow.log_metric("beats_baseline_by", acc - baseline)

            try:
                mlflow.sklearn.log_model(model, "model")
            except Exception as e:
                print(f"   (model artifact logging skipped: {e})")

            print(f"   {name:>20}: {acc*100:.1f}%   (logged to MLflow)")

            if acc > best_acc:
                best_name, best_acc, best_model = name, acc, model

    print(f"\n   {'='*45}")
    print(f"   Best model: {best_name} at {best_acc*100:.1f}%")
    print(f"   Baseline:   {baseline*100:.1f}%")
    diff = (best_acc - baseline) * 100
    print(f"   {'beats baseline by' if diff>0 else 'below baseline by'} {abs(diff):.1f} points")
    print(f"   {'='*45}")

    if best_name == "LogisticRegression":
        best_preds = best_model.predict(X_test_s)
    else:
        best_preds = best_model.predict(X_test)
    print(f"\n   Detailed report ({best_name}):")
    print(classification_report(y_test, best_preds, target_names=["Down", "Up"], zero_division=0))

    with open("data/models/best_model.pkl", "wb") as f:
        pickle.dump({"model": best_model, "name": best_name, "scaler": scaler,
                     "features": FEATURE_COLS}, f)
    print(f"3. Saved best model ({best_name}) to data/models/best_model.pkl")


if __name__ == "__main__":
    main()
