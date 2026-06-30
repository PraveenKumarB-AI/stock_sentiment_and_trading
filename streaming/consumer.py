"""
streaming/consumer.py — MODULE 14: Kafka Consumer
Continuously reads messages from the 'stock-prices' topic and logs them.
Runs independently of the producer — this is the point of Kafka: producer
and consumer don't know about each other, they only know about the topic.
Run:  python -m streaming.consumer
Stop: Ctrl+C
"""

import json
from kafka import KafkaConsumer

TOPIC = "stock-prices"

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers="localhost:9092",
    auto_offset_reset="earliest",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    group_id="stock-price-logger",
)


def main():
    print(f"Consumer started. Listening on topic '{TOPIC}'.")
    print("Press Ctrl+C to stop.\n")
    try:
        for message in consumer:
            data = message.value
            print(f"  Received: {data['ticker']} = ${data['price']}  ({data['timestamp']})")
    except KeyboardInterrupt:
        print("\nConsumer stopped.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
