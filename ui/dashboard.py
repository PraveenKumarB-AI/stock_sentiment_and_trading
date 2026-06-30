"""
ui/dashboard.py — MODULE 8: Streamlit Dashboard (midnight blue + gold)
Dark premium trading dashboard: sidebar nav, ticker strip, signal + sentiment
+ indicator panels, and a price chart. Uses the project's real data and model.
Run:  streamlit run ui/dashboard.py
"""

import glob
import pickle
import pandas as pd
import streamlit as st

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META"]
COMPANY = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corp.", "AMZN": "Amazon.com Inc.", "GOOGL": "Alphabet Inc.",
    "META": "Meta Platforms Inc.",
}
FEATURE_COLS = [
    "rsi", "macd", "macd_signal", "macd_diff",
    "sma_10", "sma_30", "daily_return", "volatility_10", "sentiment_score",
]

st.set_page_config(page_title="Market Signal Engine", page_icon="◆", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
  --bg:#0A0F1E; --panel:#111A30; --panel2:#16223C; --line:#243254;
  --ink:#EAF0FB; --muted:#8A98B8;
  --gold:#E8B65A; --gold2:#F2C879; --goldsoft:rgba(232,182,90,0.14);
  --up:#9ED39B; --up-bg:rgba(158,211,155,0.14);
  --down:#E58A93; --down-bg:rgba(229,138,147,0.14);
}
.stApp{ background:radial-gradient(1200px 600px at 80% -10%, #14203C 0%, #0A0F1E 55%); }
html, body, [class*="css"]{ font-family:'Inter',sans-serif; color:var(--ink); }
.mono{ font-family:'IBM Plex Mono',monospace; font-variant-numeric:tabular-nums; }
.block-container{ padding:1.5rem 2rem; max-width:1250px; }
#MainMenu, header, footer{ visibility:hidden; }

/* sidebar */
section[data-testid="stSidebar"]{ background:#070B16; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] *{ color:var(--ink); }
.brand{ font-size:21px; font-weight:700; padding:6px 4px 4px; letter-spacing:0.3px; }
.brand .g{ color:var(--gold); }
.brand-sub{ color:var(--muted); font-size:11px; letter-spacing:2px; text-transform:uppercase;
            padding:0 4px 18px; }
.navlabel{ color:var(--muted); font-size:11px; letter-spacing:1.5px; text-transform:uppercase;
           margin:18px 4px 8px; }

/* ticker strip */
.tape{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:20px; }
.tcard{ flex:1; min-width:120px; background:var(--panel); border:1px solid var(--line);
        border-radius:12px; padding:12px 14px; transition:border-color .2s; }
.tcard:hover{ border-color:var(--gold); }
.tcard-top{ display:flex; justify-content:space-between; align-items:center; }
.tcard-tkr{ font-family:'IBM Plex Mono',monospace; font-weight:600; font-size:13px; letter-spacing:0.5px; }
.tcard-sig{ font-size:11px; font-weight:600; padding:2px 7px; border-radius:5px; }
.sig-up{ background:var(--up-bg); color:var(--up); }
.sig-down{ background:var(--down-bg); color:var(--down); }
.sig-na{ background:rgba(138,152,184,0.14); color:var(--muted); }
.tcard-px{ font-family:'IBM Plex Mono',monospace; font-size:18px; font-weight:600; margin-top:8px;
           color:var(--gold2); }

/* panels */
.panel{ background:var(--panel); border:1px solid var(--line); border-radius:16px;
        padding:20px 22px; margin-bottom:16px; }
.panel-title{ font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:1.5px;
              color:var(--gold); margin-bottom:14px; }

/* hero */
.hero-tkr{ font-family:'IBM Plex Mono',monospace; font-size:30px; font-weight:600; letter-spacing:1px; }
.hero-co{ color:var(--muted); font-size:13px; margin-top:2px; }
.hero-px{ font-family:'IBM Plex Mono',monospace; font-size:38px; font-weight:600; color:var(--gold2); }
.badge{ display:inline-flex; align-items:center; gap:8px; padding:9px 18px; border-radius:9px;
        font-weight:600; font-size:15px; }
.badge-up{ background:var(--up-bg); color:var(--up); border:1px solid rgba(158,211,155,0.3); }
.badge-down{ background:var(--down-bg); color:var(--down); border:1px solid rgba(229,138,147,0.3); }
.badge-na{ background:rgba(138,152,184,0.12); color:var(--muted); }

.mini-label{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:1px; }

/* sentiment bar */
.sbar{ display:flex; height:34px; border-radius:9px; overflow:hidden; border:1px solid var(--line); }
.seg{ display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; }
.seg-pos{ background:var(--up); color:#0A0F1E; }
.seg-neu{ background:var(--gold); color:#0A0F1E; }
.seg-neg{ background:var(--down); color:#0A0F1E; }

.hl-pos{ color:var(--up); } .hl-neg{ color:var(--down); } .hl-neu{ color:var(--muted); }
</style>
""", unsafe_allow_html=True)


# ---------- data ----------
@st.cache_data
def load_features(t):
    try: return pd.read_csv(f"data/processed/{t}_features.csv")
    except FileNotFoundError: return None
@st.cache_data
def load_indicators(t):
    try: return pd.read_csv(f"data/processed/{t}_indicators.csv")
    except FileNotFoundError: return None
@st.cache_data
def load_sentiment(t):
    try: return pd.read_csv(f"data/processed/{t}_sentiment.csv")
    except FileNotFoundError: return None
@st.cache_resource
def load_model():
    try:
        with open("data/models/best_model.pkl", "rb") as f: return pickle.load(f)
    except FileNotFoundError: return None

def predict(bundle, t):
    f = load_features(t)
    if f is None or f.empty: return None
    latest = f[FEATURE_COLS].iloc[[-1]]
    if bundle["name"] == "LogisticRegression":
        latest = bundle["scaler"].transform(latest)
    return int(bundle["model"].predict(latest)[0])

def price_of(t):
    ind = load_indicators(t)
    return ind.iloc[-1]["Close"] if ind is not None and not ind.empty else None

bundle = load_model()

# ---------- sidebar ----------
with st.sidebar:
    st.markdown('<div class="brand">Market <span class="g">Signal</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Sentiment Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="navlabel">Markets</div>', unsafe_allow_html=True)
    ticker = st.radio("Select stock", TICKERS, label_visibility="collapsed",
                      format_func=lambda t: f"{t} · {COMPANY.get(t,'')}")
    st.markdown('<div class="navlabel">About</div>', unsafe_allow_html=True)
    st.caption("News + price → FinBERT sentiment → indicators → model signal. "
               "Engineering demo. Not financial advice.")

# ---------- ticker strip ----------
strip = '<div class="tape">'
for t in TICKERS:
    p = predict(bundle, t) if bundle else None
    sig = ("UP","sig-up") if p == 1 else ("DOWN","sig-down") if p == 0 else ("—","sig-na")
    px = price_of(t)
    pxs = f"${px:.2f}" if px is not None else "—"
    strip += (f'<div class="tcard"><div class="tcard-top">'
              f'<span class="tcard-tkr">{t}</span>'
              f'<span class="tcard-sig {sig[1]}">{sig[0]}</span></div>'
              f'<div class="tcard-px">{pxs}</div></div>')
strip += '</div>'
st.markdown(strip, unsafe_allow_html=True)

# ---------- main ----------
ind = load_indicators(ticker)
sent = load_sentiment(ticker)
pred = predict(bundle, ticker) if bundle else None

big, side = st.columns([2, 1])

with big:
    px = price_of(ticker)
    pxs = f"${px:.2f}" if px is not None else "—"
    if pred == 1: badge = '<span class="badge badge-up">▲ 3-DAY SIGNAL: UP</span>'
    elif pred == 0: badge = '<span class="badge badge-down">▼ 3-DAY SIGNAL: DOWN</span>'
    else: badge = '<span class="badge badge-na">— NO MODEL</span>'

    st.markdown(f"""
    <div class="panel">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div><div class="hero-tkr">{ticker}</div><div class="hero-co">{COMPANY.get(ticker,ticker)}</div></div>
        <div style="text-align:right;"><div class="mini-label">Latest Close</div>
          <div class="hero-px">{pxs}</div></div>
      </div>
      <div style="margin-top:16px;">{badge}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Price & Moving Averages</div>',
                unsafe_allow_html=True)
    if ind is not None and not ind.empty:
        st.line_chart(ind[["Close","sma_10","sma_30"]].tail(120),
                      color=["#E8B65A","#9ED39B","#E58A93"], height=260)
    st.markdown('</div>', unsafe_allow_html=True)

with side:
    st.markdown('<div class="panel"><div class="panel-title">Indicators</div>', unsafe_allow_html=True)
    if ind is not None and not ind.empty:
        r = ind.iloc[-1]
        rows = [("RSI (14d)", f"{r['rsi']:.1f}"), ("MACD", f"{r['macd']:.2f}"),
                ("Volatility 10d", f"{r['volatility_10']*100:.2f}%"), ("30d Avg", f"${r['sma_30']:.2f}")]
        for lab, val in rows:
            st.markdown(f'<div style="display:flex;justify-content:space-between;'
                        f'padding:9px 0;border-bottom:1px solid var(--line);">'
                        f'<span class="mini-label">{lab}</span>'
                        f'<span class="mono" style="font-weight:600;color:var(--gold2);">{val}</span></div>',
                        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">News Sentiment</div>', unsafe_allow_html=True)
    if sent is not None and not sent.empty:
        vc = sent["sentiment"].value_counts()
        pos, neg, neu = int(vc.get("positive",0)), int(vc.get("negative",0)), int(vc.get("neutral",0))
        total = pos+neg+neu or 1
        bar = '<div class="sbar">'
        for w, cls, n in [(pos/total*100,"seg-pos",pos),(neu/total*100,"seg-neu",neu),(neg/total*100,"seg-neg",neg)]:
            if w: bar += f'<div class="seg {cls}" style="width:{w}%">{n if w>10 else ""}</div>'
        bar += '</div>'
        st.markdown(bar, unsafe_allow_html=True)
        score = (pos-neg)/total
        st.markdown(f'<div style="margin-top:10px;color:var(--muted);font-size:13px;">Score '
                    f'<span class="mono" style="color:var(--gold2);font-weight:600;">{score:+.3f}</span> '
                    f'· {total} headlines</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="panel"><div class="panel-title">Recent Headlines</div>', unsafe_allow_html=True)
if sent is not None and not sent.empty:
    for _, row in sent[["sentiment","headline"]].head(10).iterrows():
        s = row["sentiment"]
        cls = "hl-pos" if s=="positive" else "hl-neg" if s=="negative" else "hl-neu"
        st.markdown(f'<div style="padding:9px 0;border-bottom:1px solid var(--line);">'
                    f'<span class="{cls} mono" style="font-size:11px;font-weight:600;">{s.upper():>8}</span>'
                    f'&nbsp;&nbsp;<span style="font-size:14px;">{row["headline"]}</span></div>',
                    unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
