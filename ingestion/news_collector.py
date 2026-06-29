"""
ingestion/news_collector.py — MODULE 2: News Collector
Fetches recent news headlines for a stock ticker using NewsAPI
and saves them as a CSV in data/raw/.
Run:  python -m ingestion.news_collector AAPL
"""

import os
import sys
import csv
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# Map tickers to a focused search query (company + financial context)
SEARCH_QUERIES = {
    "AAPL": '"Apple" AND (stock OR shares OR earnings OR iPhone OR revenue)',
    "MSFT": '"Microsoft" AND (stock OR shares OR earnings OR Azure OR revenue)',
    "TSLA": '"Tesla" AND (stock OR shares OR earnings OR EV OR Musk OR revenue)',
    "NVDA": '"Nvidia" AND (stock OR shares OR earnings OR chips OR GPU OR revenue)',
    "AMZN": '"Amazon" AND (stock OR shares OR earnings OR AWS OR revenue)',
    "GOOGL": '"Google" AND (stock OR shares OR earnings OR Alphabet OR revenue)',
    "META": '"Meta" AND (stock OR shares OR earnings OR Facebook OR revenue)',
}


def collect_news(ticker):
    """Fetch recent headlines for one ticker and save to CSV."""
    ticker = ticker.upper()
    query = SEARCH_QUERIES.get(ticker, f'"{ticker}"')

    if not NEWSAPI_KEY:
        print("   No NEWSAPI_KEY found in .env — stopping.")
        return

    print(f"1. Fetching news for {ticker}...")

    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 50,
        "apiKey": NEWSAPI_KEY,
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "ok":
        print(f"   API error: {data.get('message', 'unknown error')}")
        return

    articles = data.get("articles", [])
    print(f"   Got {len(articles)} headlines.")

    if not articles:
        print("   No articles found.")
        return

    out_path = f"data/raw/{ticker}_news.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "published_at", "title", "source"])
        for a in articles:
            writer.writerow([
                ticker,
                a.get("publishedAt", ""),
                a.get("title", ""),
                a.get("source", {}).get("name", ""),
            ])

    print(f"2. Saved to {out_path}")

    print("\n   Sample headlines:")
    for a in articles[:5]:
        print(f"   - {a.get('title', '')}")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    collect_news(ticker)
