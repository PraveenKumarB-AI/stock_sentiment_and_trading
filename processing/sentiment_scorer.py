"""
processing/sentiment_scorer.py — MODULE 4: Sentiment Scorer
Reads the collected headline CSVs for a ticker (news + social), scores each
headline with FinBERT (positive / negative / neutral), and saves the scored
rows to data/processed/<TICKER>_sentiment.csv.
Run:  python -m processing.sentiment_scorer AAPL
"""

import sys
import csv
import glob
import os
from transformers import pipeline

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META"]

# Load FinBERT once. First run downloads the model (a few hundred MB).
print("Loading FinBERT model (first run downloads it, please wait)...")
finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    truncation=True,
)
print("Model loaded.\n")


def read_headlines(ticker):
    """Gather all headlines for a ticker from its news + social CSVs."""
    headlines = []
    for path in glob.glob(f"data/raw/{ticker}_*.csv"):
        # Only the text sources have a 'title' or 'body' column
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("title") or row.get("body") or ""
                if text.strip():
                    source_file = os.path.basename(path)
                    headlines.append((source_file, text.strip()))
    return headlines


def score_sentiment(ticker):
    ticker = ticker.upper()
    print(f"1. Reading headlines for {ticker}...")
    headlines = read_headlines(ticker)
    print(f"   Found {len(headlines)} headlines.")

    if not headlines:
        print("   No headlines found — run the collectors first.")
        return

    print("2. Scoring with FinBERT...")
    results = []
    pos = neg = neu = 0
    for source_file, text in headlines:
        out = finbert(text)[0]          # e.g. {'label': 'positive', 'score': 0.97}
        label = out["label"].lower()
        score = round(out["score"], 4)
        results.append([ticker, source_file, label, score, text])
        if label == "positive":
            pos += 1
        elif label == "negative":
            neg += 1
        else:
            neu += 1

    out_path = f"data/processed/{ticker}_sentiment.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "source_file", "sentiment", "confidence", "headline"])
        writer.writerows(results)

    print(f"3. Saved to {out_path}")
    print(f"\n   Sentiment breakdown for {ticker}:")
    print(f"   Positive: {pos}   Negative: {neg}   Neutral: {neu}")
    print("\n   Samples:")
    for r in results[:5]:
        print(f"   [{r[2]:>8}] {r[4][:70]}")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    score_sentiment(ticker)
