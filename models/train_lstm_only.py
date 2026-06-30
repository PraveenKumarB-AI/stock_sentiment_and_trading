"""
models/train_lstm_only.py — MODULE 13a: LSTM (isolated process)
Trains the LSTM and saves its test-set predictions + true labels to disk,
so a separate XGBoost process can be combined into an ensemble afterward
without the two libraries sharing a process (avoids a libomp crash).
Run:  python -m models.train_lstm_only
"""

import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "rsi", "macd", "macd_signal", "macd_diff",
    "sma_10", "sma_30", "daily_return", "volatility_10", "sentiment_score",
]
WINDOW = 10
EPOCHS = 15
BATCH_SIZE = 32

torch.manual_seed(42)
np.random.seed(42)


def build_sequences():
    train_X, train_y, test_X, test_y = [], [], [], []
    for path in sorted(glob.glob("data/processed/*_features.csv")):
        df = pd.read_csv(path)
        feats = df[FEATURE_COLS].values
        targets = df["target"].values
        seqs, labels = [], []
        for i in range(WINDOW, len(df) + 1):
            seqs.append(feats[i - WINDOW:i])
            labels.append(targets[i - 1])
        seqs = np.array(seqs); labels = np.array(labels)
        cut = int(len(seqs) * 0.8)
        train_X.append(seqs[:cut]);  train_y.append(labels[:cut])
        test_X.append(seqs[cut:]);   test_y.append(labels[cut:])
    X_train = np.concatenate(train_X); y_train = np.concatenate(train_y)
    X_test  = np.concatenate(test_X);  y_test  = np.concatenate(test_y)
    return X_train, y_train, X_test, y_test


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


def main():
    print("1. Building time-ordered sequences (window =", WINDOW, "days)...")
    X_train, y_train, X_test, y_test = build_sequences()
    print(f"   Train sequences: {len(X_train)}   Test sequences: {len(X_test)}")

    n_feat = X_train.shape[2]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_feat))
    X_train = scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    X_test  = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(SeqDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    print("2. Training LSTM...")
    model = LSTMClassifier(n_features=n_feat)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        if epoch == 1 or epoch % 5 == 0:
            print(f"   Epoch {epoch:>2}/{EPOCHS}   loss: {total_loss/len(X_train):.4f}")

    print("\n3. Evaluating LSTM on the held-out test set...")
    model.eval()
    probs_chunks = []
    with torch.no_grad():
        for xb, _ in test_loader:
            probs_chunks.append(torch.sigmoid(model(xb)).numpy())
    lstm_probs = np.concatenate(probs_chunks)
    lstm_acc = accuracy_score(y_test, (lstm_probs > 0.5).astype(int))

    majority = int(pd.Series(y_train).mode()[0])
    baseline = accuracy_score(y_test, [majority] * len(y_test))
    print(f"   LSTM accuracy:     {lstm_acc*100:.1f}%")
    print(f"   Baseline accuracy: {baseline*100:.1f}%")

    np.savez("data/models/lstm_test_predictions.npz",
             probs=lstm_probs, labels=y_test, baseline=baseline)
    torch.save(model.state_dict(), "data/models/lstm_model.pt")
    print("\n4. Saved LSTM predictions to data/models/lstm_test_predictions.npz")
    print("   Saved LSTM weights to data/models/lstm_model.pt")


if __name__ == "__main__":
    main()
