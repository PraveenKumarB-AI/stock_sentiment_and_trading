"""
processing/feature_builder.py — MODULE 6: Feature Builder
Combines technical indicators (Module 5) with a sentiment signal (Module 4)
into one labelled table per ticker, where each row is labelled with whether
the NEXT day's price closed up (1) or down (0). This is the training data
for the prediction model.

Note: sentiment is collapsed to one recent signal per stock and attached to
each day, a simplification for a first working model. Date-level alignment
can be added later.

Run:  python -m processing.feature_builder AAPL
"""

import sys
import pandas as pd

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META"]


def sentiment_signal(ticker):
    """Turn the scored headlines into one number per stock:
    (positives - negatives) / total, ranging -1 (all bad) to +1 (all good)."""
    path = f"data/processed/{ticker}_sentiment.csv"
    try:
        s = pd.read_csv(path)
    except FileNotFoundError:
        print(f"   {path} not found — run the sentiment scorer first.")
        return 0.0, 0

    total = len(s)
    if total == 0:
        return 0.0, 0
    pos = (s["sentiment"] == "positive").sum()
    neg = (s["sentiment"] == "negative").sum()
    score = (pos - neg) / total
    return round(score, 4), total


def build_features(ticker):
    ticker = ticker.upper()
    print(f"1. Loading indicators for {ticker}...")
    ind_path = f"data/processed/{ticker}_indicators.csv"
    try:
        df = pd.read_csv(ind_path)
    except FileNotFoundError:
        print(f"   {ind_path} not found — run the indicators step first.")
        return

    print(f"   {len(df)} days loaded.")

    print("2. Computing sentiment signal...")
    sent_score, n_headlines = sentiment_signal(ticker)
    print(f"   Sentiment score: {sent_score}  (from {n_headlines} headlines)")
    df["sentiment_score"] = sent_score

    print("3. Building the label (next-day up/down)...")
    # Label = 1 if the next day's close is higher than today's, else 0
    df["next_close"] = df["Close"].shift(-1)
    df["target"] = (df["next_close"] > df["Close"]).astype(int)

    # Keep only the columns the model will use, drop rows with missing values
    feature_cols = [
        "rsi", "macd", "macd_signal", "macd_diff",
        "sma_10", "sma_30", "daily_return", "volatility_10",
        "sentiment_score",
    ]
    keep = feature_cols + ["target"]
    clean = df[keep].dropna()

    out_path = f"data/processed/{ticker}_features.csv"
    clean.to_csv(out_path, index=False)
    print(f"4. Saved to {out_path}")

    up = (clean["target"] == 1).sum()
    down = (clean["target"] == 0).sum()
    print(f"\n   {len(clean)} usable rows.")
    print(f"   Up days: {up}   Down days: {down}")
    print(f"   Features per row: {len(feature_cols)}")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    build_features(ticker)
