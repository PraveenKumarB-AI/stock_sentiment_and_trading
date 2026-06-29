"""
ingestion/social_collector.py — MODULE 3: Social/Discussion Collector
Fetches recent news/discussion headlines for a ticker via yfinance
(Yahoo Finance), which needs no API key. Saves them as a CSV in data/raw/.
This is our second text source alongside NewsAPI.
Run:  python -m ingestion.social_collector AAPL
"""

import sys
import csv
import yfinance as yf

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META"]


def collect_social(ticker):
    """Fetch recent Yahoo Finance news items for one ticker and save to CSV."""
    ticker = ticker.upper()
    print(f"1. Fetching Yahoo Finance news for {ticker}...")

    stock = yf.Ticker(ticker)
    news_items = stock.news

    if not news_items:
        print("   No news items found.")
        return False

    print(f"   Got {len(news_items)} items.")

    out_path = f"data/raw/{ticker}_social.csv"
    rows = []
    for item in news_items:
        # yfinance wraps the actual fields in a 'content' dict
        content = item.get("content", item)
        title = content.get("title", "")
        pub = content.get("pubDate", content.get("providerPublishTime", ""))
        provider = ""
        prov = content.get("provider")
        if isinstance(prov, dict):
            provider = prov.get("displayName", "")
        if title:
            rows.append([ticker, str(pub), title, provider])

    if not rows:
        print("   News items had no readable titles.")
        return False

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "published_at", "title", "source"])
        writer.writerows(rows)

    print(f"2. Saved to {out_path}")

    print("\n   Sample headlines:")
    for r in rows[:5]:
        print(f"   - {r[2][:90]}")

    return True


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    collect_social(ticker)
