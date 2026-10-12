"""Streamlit scanner with separate interactive TradingView chart."""
import html
import json
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from src.data import get_history
from src.patterns import scan, position_size

st.set_page_config(page_title="Fundslife | Pattern Scanner",page_icon="📈",layout="wide")
st.title("Fundslife · Smart Stock Pattern Scanner")
st.caption("Research-only technical signals · Historical OHLCV data · Not investment advice")
with st.sidebar:
    st.header("Scan settings")
    ticker=st.text_input("US ticker","NVDA").strip().upper()
    period=st.selectbox("History",["6mo","1y","2y"],index=1)
    timeframe=st.selectbox("Interval",["1d","1h"],index=0)
    capital=st.number_input("Capital (USD)",value=1000.0,min_value=1.0)
    risk_pct=st.slider("Risk per trade (%)",0.1,5.0,1.0,0.1)
    rr=st.slider("Target reward / risk",1.0,4.0,2.0,0.25)
    st.caption("Daily signals use last completed daily bar. On intraday charts, verify the latest candle is closed.")
@st.cache_data(ttl=900,show_spinner=False)
def load(symbol,p,i):
    return get_history(symbol,p,i)
try:
    raw=load(ticker,period,timeframe)
    # Exclude a possibly live intraday candle; safer than treating it as completed.
    if timeframe=="1h":raw=raw.iloc[:-1]
    df,signals=scan(raw,rr=rr)
except Exception as exc:
    st.error(f"Unable to analyze {ticker}: {exc}")
    st.stop()

last=df.iloc[-1]
trend="Bullish" if last.EMA20>last.EMA50 else "Bearish"
setup=signals[0] if signals else None
c1,c2,c3,c4=st.columns(4)
c1.metric("Last completed close",f"${last.Close:,.2f}")
c2.metric("EMA Trend",trend)
c3.metric("RSI (14)",f"{last.RSI14:.1f}")
c4.metric("Confirmed patterns",len(signals))
st.info("Signal status: "+(f"{setup.pattern} ({setup.direction}) — confirmed" if setup else "No newly confirmed entry on the last completed bar. WAIT / no trade."))

if signals:
    for s in signals:
        st.subheader(f"{s.pattern} · {s.direction} · score {s.score}/100")
        a,b,c=st.columns(3)
        a.metric("Trigger / hypothetical entry (USD)",f"${s.entry:,.2f}")
        b.metric("Stop loss (USD)",f"${s.stop:,.2f}")
        c.metric("Take profit (USD)",f"${s.target:,.2f}")
        st.write(s.explanation)
        size=position_size(s.entry,s.stop,capital,risk_pct)
        st.caption(f"Illustrative fractional sizing: {size['shares']:.4f} shares; notional ${size['notional']:.2f}; theoretical risk ${size['planned_risk']:.2f} (budget ${size['risk_budget']:.2f}). Excludes fees, FX, gaps and brokerage limitations.")
else:
    st.write("A pattern requires a **confirmed neckline cross** or **20-bar close breakout with volume confirmation**. A chart that merely resembles a pattern does not trigger a signal.")

tab_tv,tab_levels,tab_data=st.tabs(["TradingView interactive chart","Price levels & signals","Indicator data"])
with tab_tv:
    st.caption("TradingView widget is a separate chart and may use a different data source/delay. Python signals are NOT drawn onto this widget.")
    tv_symbol=("NASDAQ:"+ticker if ticker in {"NVDA","AAPL","MSFT","META","AMD","AMZN","GOOGL","TSLA"} else "AMEX:"+ticker if ticker in {"SPY","QQQ","IWM"} else ticker)
    options={"autosize":True,"symbol":tv_symbol,"interval":"D" if timeframe=="1d" else "60",
             "timezone":"exchange","theme":"dark","style":"1","locale":"en",
             "allow_symbol_change":False,"hide_top_toolbar":False,"hide_side_toolbar":False,
             "withdateranges":True,"details":False,"calendar":False,"support_host":"https://www.tradingview.com"}
    embed='<div class="tradingview-widget-container" style="height:610px;width:100%"><div class="tradingview-widget-container__widget" style="height:100%;width:100%"></div><script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>'+json.dumps(options)+'</script></div>'
    components.html(embed,height=630)
    st.markdown(f"[Open {html.escape(ticker)} directly on TradingView](https://www.tradingview.com/chart/?symbol={html.escape(tv_symbol)})")
with tab_levels:
    recent=df.tail(120)
    fig=go.Figure(data=[go.Candlestick(x=recent.index,open=recent.Open,high=recent.High,low=recent.Low,close=recent.Close,name="Price")])
    fig.add_trace(go.Scatter(x=recent.index,y=recent.EMA20,name="EMA20",line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=recent.index,y=recent.EMA50,name="EMA50",line=dict(width=1.5)))
    if setup:
        for title,price,dash in [("Entry",setup.entry,"solid"),("Stop",setup.stop,"dot"),("Target",setup.target,"dash")]:
            fig.add_hline(y=price,line_dash=dash,annotation_text=f"{title} {price:.2f}",annotation_position="top left")
    fig.update_layout(height=580,xaxis_rangeslider_visible=False,title=f"{ticker} · Python analysis levels (USD)",xaxis_title="Date",yaxis_title="Adjusted price (USD)")
    st.plotly_chart(fig,use_container_width=True)
with tab_data:
    st.dataframe(df[["Close","EMA20","EMA50","RSI14","ATR14","Prior20High","Prior20Low"]].tail(20).round(3),use_container_width=True)
    st.download_button("Download indicators CSV",df.to_csv().encode(),"scanner_indicators.csv","text/csv")
st.caption(f"Last input candle timestamp: {df.index[-1]} · Yahoo Finance via yfinance. Adjusted price data can differ from TradingView. Pattern score is heuristic, NOT a calibrated probability or win rate.")
