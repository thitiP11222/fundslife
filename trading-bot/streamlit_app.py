from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import download_ohlcv
from src.technical_dashboard import (
    TechnicalConfig,
    add_dashboard_indicators,
    build_trade_plan,
)


st.set_page_config(
    page_title="Fundslife Trading Scanner",
    page_icon="📈",
    layout="wide",
)

st.title("Fundslife Technical Trading Dashboard")
st.caption(
    "Technical-analysis scanner for support/resistance, entry planning and chart-pattern confirmation. "
    "Yahoo Finance intraday data may be delayed and is not exchange-grade real-time data."
)

with st.sidebar:
    st.header("Scanner Settings")

    symbol = st.text_input("Ticker", value="NVDA").upper().strip()

    timeframe = st.selectbox(
        "Timeframe",
        ["15m", "30m", "1h", "1d"],
        index=2,
    )

    period_map = {
        "15m": ["5d", "1mo", "60d"],
        "30m": ["5d", "1mo", "60d"],
        "1h": ["1mo", "3mo", "6mo", "1y"],
        "1d": ["6mo", "1y", "2y", "5y"],
    }

    period = st.selectbox(
        "History",
        period_map[timeframe],
        index=min(1, len(period_map[timeframe]) - 1),
    )

    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()

    st.divider()
    st.caption(
        "For intraday use, refresh after a candle closes. "
        "Do not treat an unfinished candle as confirmed."
    )


@st.cache_data(ttl=60)
def load_data(ticker: str, interval: str, period_value: str) -> pd.DataFrame:
    return download_ohlcv(ticker, interval, period_value)


if not symbol:
    st.warning("Enter a ticker.")
    st.stop()

try:
    raw = load_data(symbol, timeframe, period)
except Exception as exc:
    st.error(f"Could not load {symbol}: {exc}")
    st.stop()

cfg = TechnicalConfig()

try:
    data = add_dashboard_indicators(raw, cfg)
    plan = build_trade_plan(raw, cfg)
except Exception as exc:
    st.error(f"Not enough data to analyze {symbol}: {exc}")
    st.stop()

left, middle, right, extra = st.columns(4)

with left:
    st.metric("Last price", f"USD {plan['close']:.2f}")

with middle:
    st.metric("Setup", plan["setup"])

with right:
    st.metric("Technical score", f"{plan['score']}/100")

with extra:
    st.metric("RSI 14", f"{plan['rsi']:.1f}")


fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name=symbol,
    )
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["EMA20"],
        mode="lines",
        name="EMA20",
    )
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["EMA50"],
        mode="lines",
        name="EMA50",
    )
)

for level in plan["supports"]:
    fig.add_hline(
        y=level,
        line_dash="dot",
        annotation_text=f"Support {level:.2f}",
        annotation_position="bottom left",
    )

for level in plan["resistances"]:
    fig.add_hline(
        y=level,
        line_dash="dash",
        annotation_text=f"Resistance {level:.2f}",
        annotation_position="top left",
    )

fig.add_hline(
    y=plan["pullback_entry"],
    line_dash="dot",
    annotation_text=f"Pullback entry {plan['pullback_entry']:.2f}",
)

fig.add_hline(
    y=plan["breakout_entry"],
    line_dash="dash",
    annotation_text=f"Breakout entry {plan['breakout_entry']:.2f}",
)

fig.add_hline(
    y=plan["stop"],
    line_dash="dashdot",
    annotation_text=f"Invalidation {plan['stop']:.2f}",
)

fig.add_hline(
    y=plan["target_1"],
    line_dash="dot",
    annotation_text=f"T1 {plan['target_1']:.2f}",
)

fig.add_hline(
    y=plan["target_2"],
    line_dash="dot",
    annotation_text=f"T2 {plan['target_2']:.2f}",
)

fig.update_layout(
    height=720,
    xaxis_rangeslider_visible=False,
    title=f"{symbol} — {timeframe}",
    margin=dict(l=20, r=20, t=50, b=20),
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Technical plan")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.write("**Nearest support**")
    st.write(f"USD {plan['nearest_support']:.2f}")
    st.write("**Nearest resistance**")
    st.write(f"USD {plan['nearest_resistance']:.2f}")

with c2:
    st.write("**Pullback entry zone**")
    st.write(f"Approx. USD {plan['pullback_entry']:.2f}")
    st.write("**Breakout confirmation**")
    st.write(f"Above USD {plan['breakout_entry']:.2f}")

with c3:
    st.write("**Invalidation / stop reference**")
    st.write(f"Below USD {plan['stop']:.2f}")
    st.write("**ATR 14**")
    st.write(f"USD {plan['atr']:.2f}")

with c4:
    st.write("**Target 1**")
    st.write(f"USD {plan['target_1']:.2f}")
    st.write("**Target 2**")
    st.write(f"USD {plan['target_2']:.2f}")

st.subheader("Chart patterns")

if plan["patterns"]:
    st.dataframe(
        pd.DataFrame(plan["patterns"]),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No supported chart pattern detected on the latest bars.")

st.subheader("Why this score?")

if plan["reasons"]:
    for reason in plan["reasons"]:
        st.write(f"- {reason}")
else:
    st.write("- No bullish confirmation currently qualifies.")

st.subheader("Indicator snapshot")

indicator_df = pd.DataFrame(
    {
        "Metric": [
            "Close",
            "EMA20",
            "EMA50",
            "RSI14",
            "ATR14",
            "Volume ratio vs MA20",
        ],
        "Value": [
            round(plan["close"], 2),
            round(plan["ema20"], 2),
            round(plan["ema50"], 2),
            round(plan["rsi"], 2),
            round(plan["atr"], 2),
            round(plan["volume_ratio"], 2),
        ],
    }
)

st.dataframe(
    indicator_df,
    use_container_width=True,
    hide_index=True,
)

st.warning(
    "Entry, stop and target levels are technical-analysis references, not guaranteed prices. "
    "Confirm liquidity, spread, news/events and position sizing before trading."
)
