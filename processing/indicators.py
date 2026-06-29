"""
processing/indicators.py — MODULE 5: Technical Indicators
Reads a ticker's price CSV (from Module 1), computes technical indicators
(RSI, MACD, moving averages, volatility), and saves the enriched table to
data/processed/<TICKER>_indicators.csv.
Run:  python -m processing.indicators AAPL
"""

import sys
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META"]


def compute_indicators(ticker):
    ticker = ticker.upper()
    print(f"1. Loading price data for {ticker}...")

    in_path = f"data/raw/{ticker}_prices.csv"
    try:
        df = pd.read_csv(in_path)
    except FileNotFoundError:
        print(f"   {in_path} not found — run the price collector first.")
        return

    if "Close" not in df.columns:
        print("   No 'Close' column found in price data.")
        return

    print(f"   {len(df)} trading days loaded.")

    print("2. Computing indicators...")
    close = df["Close"]

    # RSI — momentum, 0..100
    df["rsi"] = RSIIndicator(close=close, window=14).rsi()

    # MACD — trend (the MACD line and its signal line)
    macd = MACD(close=close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # Moving averages — the average price over the last N days
    df["sma_10"] = SMAIndicator(close=close, window=10).sma_indicator()
    df["sma_30"] = SMAIndicator(close=close, window=30).sma_indicator()

    # Volatility — daily return and a 10-day rolling standard deviation
    df["daily_return"] = close.pct_change()
    df["volatility_10"] = df["daily_return"].rolling(window=10).std()

    # Bollinger Bands — price channel based on volatility
    bb = BollingerBands(close=close, window=20)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()

    out_path = f"data/processed/{ticker}_indicators.csv"
    df.to_csv(out_path, index=False)
    print(f"3. Saved to {out_path}")

    # Show the most recent day's indicator values
    print(f"\n   Most recent indicators for {ticker}:")
    latest = df.iloc[-1]
    for col in ["Close", "rsi", "macd", "macd_signal", "sma_10", "sma_30", "volatility_10"]:
        val = latest[col]
        print(f"   {col:>15}: {val:.4f}" if pd.notna(val) else f"   {col:>15}: (n/a)")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    compute_indicators(ticker)
