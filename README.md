# Real Time Stock Sentiment And Trading Signal Engine

A machine-learning pipeline that reads live financial news, Reddit discussion, and stock price data, scores the market's mood with a finance-trained language model, and predicts short-term price movement. Built end to end as a demonstration of real ML engineering.

> This is a learning and portfolio project, not a real trading tool. The predictions exist to demonstrate engineering skill and must not be used for actual trading decisions. Nothing here is financial advice.

**Live demo:** https://stocksentimentandtrading-v7nhvnxmjsoe5xjpmc9us2.streamlit.app

**Live demo:** https://stocksentimentandtrading-v7nhvnxmjsoe5xjpmc9us2.streamlit.app

## What it does

For a set of well-known stocks — Apple, Microsoft, Tesla, Nvidia, Amazon, Google, and Meta — the system gathers three kinds of signal: how the news is talking about each company, how Reddit is talking about it, and how its price has been moving. It turns the text into sentiment scores and the prices into technical indicators, then combines them to estimate whether a stock is likely to move up or down in the near term. The result is shown on a live dashboard, and ultimately updates in real time as fresh data arrives.

The project is built in small, numbered modules, each one a self-contained step that produces something working before the next is added.

## Why it is built this way

Predicting markets is genuinely hard, and the goal here is not to beat the market — it is to show that every piece of a modern ML system can be built, connected, and deployed: data collection, natural-language processing, feature engineering, model training, experiment tracking, an API, a dashboard, and real-time streaming. That full pipeline is what the project demonstrates.

## Tech stack

Everything used here is free and open-source.

| Layer | Tools |
|---|---|
| Data collection | yfinance, NewsAPI, PRAW (Reddit) |
| Natural language | Hugging Face Transformers, FinBERT, PyTorch |
| Data and features | pandas, NumPy, the ta technical-analysis library |
| Modeling | scikit-learn, XGBoost |
| MLOps and serving | MLflow, FastAPI, Uvicorn |
| Storage | PostgreSQL |
| Interface and deploy | Streamlit, Streamlit Community Cloud |
| Streaming and infra | Apache Kafka, Docker |
| Version control and CI | Git, GitHub, GitHub Actions |

## Roadmap

The project is built in modules. Status is marked as the build progresses.

**Core — a working, deployed app**

- [x] **Module 1 — Price Data Collector.** Daily price history per ticker via yfinance.
- [x] **Module 2 — News Collector.** Finance-focused news headlines per ticker via NewsAPI.
- [x] **Module 3 — Social/News Collector.** A second stream of finance headlines per ticker via Yahoo Finance (yfinance), used after Reddit and StockTwits API access proved closed.
- [x] **Module 4 — Sentiment Scorer.** Scores every collected headline with pre-trained FinBERT (positive / negative / neutral).
- [x] **Module 5 — Technical Indicators.** RSI, MACD, moving averages, Bollinger Bands, and volatility computed from price data with the ta library.
- [x] **Module 6 — Feature Builder.** Merges indicators and a sentiment signal into one table per stock, labelled with next-day up/down movement.
- [x] **Module 7 — Prediction Model.** Compares XGBoost, Random Forest, and Logistic Regression on a 3-day horizon with a time-based split. Best model beats baseline; absolute accuracy stays near coin-flip, the honest reality of short-term price prediction.
- [x] **Module 8 — Dashboard.** A dark, midnight-blue-and-gold Streamlit dashboard showing each stock's price, indicators, news sentiment, and the model's 3-day signal.
- [x] **Module 9 — Deploy (free).** Live on Streamlit Community Cloud.

**Upgrades — stronger models and MLOps**

- [x] **Module 10 — Fine-Tune FinBERT.** Fine-tuned on the Financial PhraseBank dataset (3,876 train / 776 test sentences, 3 epochs, T4 GPU via Colab). Final F1: 83.1%, accuracy: 83.1%, on a held-out test set.

  | Epoch | Train Loss | Val Loss | Accuracy | F1 |
  |---|---|---|---|---|
  | 1 | 0.546 | 0.447 | 81.8% | 81.9% |
  | 2 | 0.340 | 0.425 | 82.5% | 82.7% |
  | 3 | 0.194 | 0.459 | 83.1% | 83.1% |

  Validation loss bottomed at epoch 1 while train loss kept falling — a mild sign of overfitting by epoch 3, worth noting honestly. The training notebook is included for full reproducibility; the resulting model weights (~418MB) are kept local rather than committed, since large binaries don't belong in a git repo — standard practice for trained models.
- [x] **Module 11 — Experiment Tracking.** Each model run (XGBoost, Random Forest, Logistic Regression) is logged to MLflow with its hyperparameters, accuracy, and baseline comparison.

  | Model | Accuracy | Baseline | Beats baseline by |
  |---|---|---|---|
  | RandomForest | 49.4% | 40.9% | +8.4 pts |
  | LogisticRegression | 47.1% | 40.9% | +6.2 pts |
  | XGBoost | 46.1% | 40.9% | +5.2 pts |

  Note: the MLflow web UI (`mlflow ui`) currently fails to start on Python 3.14 due to an `importlib.abc` incompatibility in MLflow's server code — a known lag between a brand-new Python release and the broader library ecosystem. Logged runs are fully intact in `mlflow.db` and viewable via `python -m models.view_mlflow_runs`, a small script included in the repo as a UI-independent fallback.
- [ ] **Module 12 — Prediction API.** FastAPI endpoint serving the model.
- [ ] **Module 13 — LSTM Ensemble.** LSTM combined with XGBoost into an ensemble.

**Stretch — real infrastructure**

- [ ] **Module 14 — Real-Time Streaming.** Live ingestion with Apache Kafka in Docker.
- [ ] **Module 15 — AWS Deployment.** Containerised API deployed to AWS.
- [ ] **Module 16 — Capstone.** The full system assembled, documented, and deployed.

## Project structure

```
stock_sentiment_and_trading/
├── ingestion/
│   ├── price_collector.py    Module 1 — price history via yfinance
│   ├── news_collector.py     Module 2 — news headlines via NewsAPI
│   └── social_collector.py   Module 3 — Yahoo Finance news via yfinance
├── processing/
│   ├── sentiment_scorer.py   Module 4 — FinBERT sentiment scoring
│   ├── indicators.py         Module 5 — technical indicators via ta
│   └── feature_builder.py    Module 6 — merge features + up/down label
├── models/
│   └── train_model.py        Module 7 — train and compare prediction models
├── ui/
│   └── dashboard.py          Module 8 — Streamlit dashboard
├── notebooks/
│   └── finetune_finbert.ipynb  Module 10 — FinBERT fine-tuning (Colab)
├── data/
│   └── raw/                  Collected data (kept local, not committed)
├── requirements.txt
├── .gitignore
└── README.md
```

## Running it locally

Clone the project and set up the environment:

```bash
git clone https://github.com/PraveenKumarB-AI/stock_sentiment_and_trading.git
cd stock_sentiment_and_trading
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Module 1 needs nothing extra:

```bash
python -m ingestion.price_collector AAPL
```

Module 2 needs a free NewsAPI key. Get one at https://newsapi.org/register, then create a file named `.env` in the project root containing:

```
NEWSAPI_KEY=your_key_here
```

Then run:

```bash
python -m ingestion.news_collector AAPL
```

The supported tickers are AAPL, MSFT, TSLA, NVDA, AMZN, GOOGL, and META.

## A note on the data

News search is never perfect — a few off-topic headlines slip through even with a focused query, and big-tech stories often mention several companies at once. That is expected and does not hurt the system, since sentiment is scored per stock and used in aggregate rather than headline by headline.

## Author

Praveen Kumar Botta — AI / ML Engineer
https://github.com/PraveenKumarB-AI

## License

MIT