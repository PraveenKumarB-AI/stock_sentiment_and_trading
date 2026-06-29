"""
ingestion/price_collector.py — MODULE 1: Price Data Collector
Downloads historical daily price data for a stock ticker using yfinance
and saves it as a CSV in data/raw/.
Run:  python -m ingestion.price_collector AAPL
"""

import sys
import yfinance as yf


def collect_prices(ticker, period="1y"):
    """Download daily price history for one ticker and save to CSV."""
    ticker = ticker.upper()
    print(f"1. Downloading {period} of price data for {ticker}...")

    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if df.empty:
        print(f"   No data found for {ticker}. Check the ticker symbol.")
        return

    print(f"   Got {len(df)} trading days.")

    out_path = f"data/raw/{ticker}_prices.csv"
    df.to_csv(out_path)
    print(f"2. Saved to {out_path}")

    print("\n   Most recent days:")
    print(df[["Open", "High", "Low", "Close", "Volume"]].tail().to_string())


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    collect_prices(ticker)
