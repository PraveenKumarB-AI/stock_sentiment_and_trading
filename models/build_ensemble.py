"""
models/build_ensemble.py — MODULE 13c: Combine LSTM + XGBoost
Loads the two saved prediction files, checks they're aligned on the same
test labels, and reports XGBoost alone, LSTM alone, and the averaged
ensemble against baseline.
Run:  python -m models.build_ensemble
"""

import numpy as np
from sklearn.metrics import accuracy_score

lstm = np.load("data/models/lstm_test_predictions.npz")
xgb = np.load("data/models/xgb_test_predictions.npz")

lstm_probs, lstm_labels, baseline = lstm["probs"], lstm["labels"], float(lstm["baseline"])
xgb_probs, xgb_labels = xgb["probs"], xgb["labels"]

assert len(lstm_probs) == len(xgb_probs), "Length mismatch — re-run both scripts"
assert np.array_equal(lstm_labels, xgb_labels), "Label mismatch — test sets are not aligned"
print("Alignment check passed: both models evaluated on identical test days.\n")

y_test = lstm_labels
lstm_acc = accuracy_score(y_test, (lstm_probs > 0.5).astype(int))
xgb_acc = accuracy_score(y_test, (xgb_probs > 0.5).astype(int))

ensemble_probs = (lstm_probs + xgb_probs) / 2
ensemble_acc = accuracy_score(y_test, (ensemble_probs > 0.5).astype(int))

print(f"{'='*50}")
print(f"XGBoost alone:        {xgb_acc*100:.1f}%")
print(f"LSTM alone:           {lstm_acc*100:.1f}%")
print(f"Ensemble (average):   {ensemble_acc*100:.1f}%")
print(f"Baseline:             {baseline*100:.1f}%")
print(f"{'='*50}")
