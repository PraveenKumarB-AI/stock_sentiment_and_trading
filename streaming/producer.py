"""
streaming/producer.py — MODULE 14: Kafka Producer
Continuously fetches the latest price for each tracked ticker and publishes
it as a JSON message to the 'stock-prices' Kafka topic, once per interval.
Run:  python -m streaming.producer
Stop: Ctrl+C
"""

import json
import time
from datetime import datetime, timezone
import yfinance as yf
from kafka import KafkaProducer

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META"]
TOPIC = "stock-prices"
INTERVAL_SECONDS = 30

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)


def fetch_price(ticker):
    stock = yf.Ticker(ticker)
    info = stock.fast_info
    return {
        "ticker": ticker,
        "price": round(float(info["lastPrice"]), 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main():
    print(f"Producer started. Publishing to topic '{TOPIC}' every {INTERVAL_SECONDS}s.")
    print("Press Ctrl+C to stop.\n")
    try:
        while True:
            for ticker in TICKERS:
                try:
                    msg = fetch_price(ticker)
                    producer.send(TOPIC, value=msg)
                    print(f"  Sent: {msg}")
                except Exception as e:
                    print(f"  Error fetching {ticker}: {e}")
            producer.flush()
            print(f"--- batch sent, sleeping {INTERVAL_SECONDS}s ---\n")
            time.sleep(INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nProducer stopped.")
    finally:
        producer.close()


if __name__ == "__main__":
    main()
