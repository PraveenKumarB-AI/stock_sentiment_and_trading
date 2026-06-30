"""
models/view_mlflow_runs.py — view logged MLflow runs without the web UI.
Useful as a fallback when the MLflow server itself won't start (e.g. due
to a library/Python version mismatch) — the same logged data, read directly.
Run:  python -m models.view_mlflow_runs
"""

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
runs = mlflow.search_runs(experiment_names=["stock_prediction_models"])

cols = ["tags.mlflow.runName", "metrics.accuracy",
        "metrics.baseline_accuracy", "metrics.beats_baseline_by"]
table = runs[cols].rename(columns={
    "tags.mlflow.runName": "model",
    "metrics.accuracy": "accuracy",
    "metrics.baseline_accuracy": "baseline",
    "metrics.beats_baseline_by": "beats_baseline_by",
})
table = table.sort_values("accuracy", ascending=False).reset_index(drop=True)

print("\nMLflow-logged model comparison (stock_prediction_models experiment):\n")
print(table.to_string(index=False))
print(f"\nBest: {table.iloc[0]['model']} at {table.iloc[0]['accuracy']*100:.1f}%")
