"""Fundslife Price Action V4 — decision support, not an order-entry system."""
from datetime import datetime,time
from zoneinfo import ZoneInfo
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scanner import indicators,signals,structure,pattern,backtest
from trade_plan import describe
from strategy_v5 import filtered_signals,compare_v4_v5,filter_diagnostics

st.set_page_config(page_title="Fundslife Price Action V4",layout="wide",page_icon="📈")
@st.cache_data(ttl=900)
def price(ticker):
    d=yf.Ticker(ticker).history(period="5y",interval="1d",auto_adjust=True)
    if d.empty:raise ValueError("No historical prices")
    d=d[["Open","High","Low","Close","Volume"]].dropna()
    d.index=pd.DatetimeIndex(d.index).tz_localize(None)
    now=datetime.now(ZoneInfo("America/New_York"))
    if d.index[-1].date()==now.date() and now.time()<time(16,15):
        d=d.iloc[:-1]
    return d

st.title("Fundslife | Price Action Scanner V4")
st.caption("Market Structure • Confirmed Pivot S/R • Double Top/Bottom candidates • Breakout • Risk-aware backtest")
with st.sidebar:
    ticker=st.text_input("Ticker","NVDA").upper().strip()
    hold=st.slider("Maximum hold days",2,15,5)
    stop=st.slider("ATR Stop",.5,4.,1.5,.25)
    rr=st.slider("Risk : Reward",1.,5.,2.,.25)
    capital=st.number_input("Account capital THB",value=1000.,min_value=100.)
    risk=st.number_input("Risk % per trade",value=1.,min_value=.1,max_value=10.)
if not ticker or len(ticker)>20 or not all(c.isalnum() or c in ".^- " for c in ticker):
    st.error("Invalid ticker");st.stop()
try:
    a=price(ticker); m=price("SPY")
    x=signals(a,m)
except Exception as exc:
    st.error(f"Market data could not be loaded: {exc}");st.stop()
trend=structure(x); pat=pattern(x)
last=x.iloc[-1];action=last.action
c1,c2,c3,c4=st.columns(4)
c1.metric("Direction",trend)
c2.metric("Pattern",pat)
c3.metric("Signal",action)
c4.metric("1% risk budget",f"฿{capital*risk/100:,.2f}")
if action=="WAIT":st.info("WAIT: no qualifying confirmed setup. Pattern candidates are observations, not entry signals.")
else:st.warning("Directional signal only. Options contract must independently pass full-premium risk and liquidity checks.")
st.caption(f'Closed candle: {x.index[-1].date()} · Support/Resistance use delayed pivot confirmation. Verify quotes in broker.')
left,right=st.columns([3,1])
with left:
    chart=x.tail(180)
    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=chart.index,open=chart.Open,high=chart.High,low=chart.Low,close=chart.Close,name="OHLC"))
    for name in ["ema20","ema50","support","resistance"]:
        fig.add_trace(go.Scatter(x=chart.index,y=chart[name],name=name,mode="lines",connectgaps=False))
    fig.update_layout(height=550,xaxis_rangeslider_visible=False)
    st.plotly_chart(fig,use_container_width=True)
with right:
    st.subheader("Levels")
    st.metric("Support",f'${last.support:.2f}' if pd.notna(last.support) else "—")
    st.metric("Resistance",f'${last.resistance:.2f}' if pd.notna(last.resistance) else "—")
    st.metric("RSI14",f"{last.rsi:.1f}")
    st.metric("ATR14",f"{last.atr:.2f}")
    st.metric("Volume ratio",f"{last.vol_ratio:.2f}×")
st.subheader("Trade Setup Report")
plan=describe(x)
st.write(plan["reason"])
report_data={
    "Trend":plan["ticker_trend"],
    "Pattern Candidate":plan["pattern_candidate"],
    "Entry Trigger":f'${plan["entry_trigger"]:.2f}' if plan["entry_trigger"] is not None else "WAIT",
    "Invalidation":f'${plan["invalidation"]:.2f}' if plan["invalidation"] is not None else "N/A",
}
st.table(pd.DataFrame(list(report_data.items()),columns=["Measure","Result"]))
st.caption(plan["disclaimer"])
st.subheader("V5 experimental filter comparison")
st.caption("Fixed entry filters: max ATR/price 4.5%, max EMA20 extension 1 ATR, min EMA trend gap 0.8%, and maximum opening gap 0.5 ATR. Parameters are hypotheses, NOT optimized from uploaded trade logs.")
try:
    comparison=compare_v4_v5(a,m,hold=hold,stop_atr=stop,rr=rr)
    st.dataframe(comparison,hide_index=True,use_container_width=True)
    st.subheader("Why did V5 make zero trades?")
    diagnostic=filter_diagnostics(a,m)
    st.dataframe(diagnostic,hide_index=True,use_container_width=True)
    st.caption("Counts use all V4 signal bars. Opening gap is checked at next session open, never assumed known before that open.")
    st.warning("Do not select V5 solely because it scores better on a viewed holdout; evaluate on fresh dates and other tickers.")
except Exception as error:
    st.warning(f"V5 comparison unavailable: {error}")
st.subheader("Backtest — V4 underlying directional proxy")
cut=max(90,int(len(x)*.70))
results=[]
for name,start,end in [("Earlier 70%",65,cut),("Final 30%",cut,len(x))]:
    deals,metrics=backtest(x,hold=hold,stop_atr=stop,rr=rr,start=start,end=end)
    results.append({"window":name,**metrics})
    if name=="Final 30%" and not deals.empty:
        st.download_button("Download recent trades",deals.to_csv(index=False),f"{ticker}_v4_trades.csv")
st.dataframe(pd.DataFrame(results),hide_index=True,use_container_width=True)
st.warning("Historical STOCK directional returns are NOT historical OPTIONS profit. No IV, premium, theta, bid-ask, exercise, or broker fills. Do not tune repeatedly on holdout and call it out-of-sample. No trained AI model.")
st.caption("No automated Dime orders; this tool is educational research and manual decision support.")
