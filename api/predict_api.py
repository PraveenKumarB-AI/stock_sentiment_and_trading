"""
api/predict_api.py — MODULE 12: Prediction API
Serves the trained model's 3-day up/down signal over HTTP, plus read-only
endpoints for sentiment and supported tickers. Mirrors the auth +
rate-limiting pattern from the financial-doc-intelligence project:
read-only endpoints are open, the prediction endpoints require an API key
and are rate-limited.

Run:        uvicorn api.predict_api:app --reload --port 8001
Docs (UI):  http://127.0.0.1:8001/docs
"""

import os
import pickle
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "stock-signal-demo-key-2026")

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META"]
FEATURE_COLS = [
    "rsi", "macd", "macd_signal", "macd_diff",
    "sma_10", "sma_30", "daily_return", "volatility_10", "sentiment_score",
]

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Stock Sentiment & Signal API", version="1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_model_bundle = None


def get_model():
    global _model_bundle
    if _model_bundle is None:
        with open("data/models/best_model.pkl", "rb") as f:
            _model_bundle = pickle.load(f)
    return _model_bundle


def check_api_key(x_api_key: str):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def run_prediction(ticker: str):
    bundle = get_model()
    feats = pd.read_csv(f"data/processed/{ticker}_features.csv")
    latest = feats[FEATURE_COLS].iloc[[-1]]
    model_input = bundle["scaler"].transform(latest) if bundle["name"] == "LogisticRegression" else latest
    pred = int(bundle["model"].predict(model_input)[0])
    ind = pd.read_csv(f"data/processed/{ticker}_indicators.csv").iloc[-1]
    return {
        "ticker": ticker,
        "model": bundle["name"],
        "signal": "up" if pred == 1 else "down",
        "horizon_days": 3,
        "latest_close": round(float(ind["Close"]), 2),
        "rsi": round(float(ind["rsi"]), 2),
        "macd": round(float(ind["macd"]), 2),
    }


@app.get("/")
def root():
    return {"status": "ok", "service": "stock-sentiment-signal-api", "tickers": len(TICKERS)}


@app.get("/tickers")
def tickers():
    return {"tickers": TICKERS}


@app.get("/sentiment/{ticker}")
def sentiment(ticker: str):
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail="Unknown ticker")
    try:
        df = pd.read_csv(f"data/processed/{ticker}_sentiment.csv")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No sentiment data for this ticker")
    counts = df["sentiment"].value_counts()
    pos, neg, neu = int(counts.get("positive", 0)), int(counts.get("negative", 0)), int(counts.get("neutral", 0))
    total = pos + neg + neu
    score = round((pos - neg) / total, 4) if total else 0.0
    return {"ticker": ticker, "positive": pos, "negative": neg, "neutral": neu, "total": total, "score": score}


@app.get("/predict/{ticker}")
@limiter.limit("20/minute")
def predict(request: Request, ticker: str, x_api_key: str = Header(None, alias="X-API-Key")):
    check_api_key(x_api_key)
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail="Unknown ticker")
    try:
        return run_prediction(ticker)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No feature data for this ticker")


@app.get("/predict-all")
@limiter.limit("10/minute")
def predict_all(request: Request, x_api_key: str = Header(None, alias="X-API-Key")):
    check_api_key(x_api_key)
    results = []
    for t in TICKERS:
        try:
            results.append(run_prediction(t))
        except FileNotFoundError:
            continue
    return {"predictions": results}
