"""Fundslife Options Scanner — one-screen research terminal (no broker orders)."""
from __future__ import annotations
from datetime import datetime, time
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from options_core.market import history, chain
from options_core.strategy_v2 import strategy_v2
from options_core.backtest_v3 import backtest_atr
from options_core.risk import option_risk_check

st.set_page_config(page_title="Fundslife | Swing Scanner",page_icon="📈",layout="wide")
@st.cache_data(ttl=900,show_spinner=False)
def get_daily(ticker):
    return history(ticker)
@st.cache_data(ttl=900,show_spinner=False)
def get_expiries(ticker):
    return tuple(yf.Ticker(ticker).options)
@st.cache_data(ttl=900,show_spinner=False)
def get_chain(ticker, expiry):
    return chain(ticker,expiry)
def completed(frame):
    """Drop today's NY bar unless at least 15 minutes after NYSE regular close."""
    now=datetime.now(ZoneInfo("America/New_York"))
    if len(frame) and frame.index[-1].date()==now.date() and now.time()<time(16,15):
        return frame.iloc[:-1].copy()
    return frame.copy()

st.title("Fundslife | Options Swing Scanner")
st.caption("Rule-based research terminal • Daily swing signals • Manual execution in Dime • NOT an AI-trained price predictor")
with st.sidebar:
    st.header("Setup")
    symbol=st.text_input("US Stock / ETF","NVDA").strip().upper()
    account=st.number_input("Capital (THB)",min_value=100.0,value=1000.0,step=100.0)
    risk=st.number_input("Risk per trade (%)",min_value=0.1,max_value=10.0,value=1.0,step=0.1)
    usd_thb=st.number_input("USD/THB — enter current broker rate",min_value=1.0,value=35.0,step=0.1)
    hold=st.slider("Maximum holding sessions",2,15,5)
    atr_stop=st.slider("Stop distance (ATR)",0.5,4.0,1.5,0.25)
    target=st.slider("Target (R)",0.5,5.0,2.0,0.25)
    trailing=st.slider("Trailing stop (ATR; 0 off)",0.0,4.0,0.0,0.25)
    fee=st.number_input("Proxy commission per side (bps)",min_value=0.0,value=10.0)
    st.warning("Strict 1% full-premium-loss rule often blocks all standard US option contracts with ฿1,000.")

if not symbol or len(symbol)>20 or not all(c.isalnum() or c in ".^-" for c in symbol):
    st.error("Invalid ticker symbol")
    st.stop()
try:
    asset=completed(get_daily(symbol))
    benchmark=completed(get_daily("SPY"))
    data=strategy_v2(asset,benchmark)
    if data.empty: raise ValueError("No daily signal bars")
except Exception as exc:
    st.error("Market data unavailable: "+str(exc))
    st.stop()
last=data.iloc[-1]
signal=str(last["action"])
completed_date=str(data.index[-1].date())
spy_last=benchmark.iloc[-1]
market_mode="Risk-on" if float(spy_last["Close"])>float(benchmark["Close"].ewm(span=20,adjust=False).mean().iloc[-1]) else "Cautious"
a,b,c,d=st.columns(4)
a.metric("Ticker",symbol)
b.metric("Latest signal",signal)
c.metric("Market mode",market_mode)
d.metric("Risk limit",f"฿{account*risk/100:,.2f}")
st.caption(f"Last evaluated daily bar: {completed_date} (New York session). Signal is for underlying direction only. Independently verify data freshness.")
if signal=="HOLD":st.info("NO SETUP: wait for a new confirmed signal. Do not force a trade.")
else:st.warning("Potential directional setup only. Must pass contract liquidity AND strict full-premium risk check.")

tab_scan,tab_contract,tab_test=st.tabs(["Technical Setup","Options Contract Check","Backtest & Robustness"])
with tab_scan:
    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,open=data["Open"],high=data["High"],low=data["Low"],close=data["Close"],name="Price"))
    for col,color in [("ema20","#37a2ff"),("ema50","#ed9c3b")]:
        fig.add_trace(go.Scatter(x=data.index,y=data[col],name=col.upper(),line={"color":color}))
    fig.update_layout(xaxis_rangeslider_visible=False,height=500,margin=dict(l=5,r=5,t=15,b=5))
    st.plotly_chart(fig,use_container_width=True)
    c1,c2,c3=st.columns(3)
    c1.metric("RSI14",f'{last["rsi"]:.1f}')
    c2.metric("ATR14",f'${last["atr14"]:.2f}')
    c3.metric("Volume / avg20",f'{last["volume_ratio"]:.2f}×')
    st.caption("CALL_BIAS / PUT_BIAS comes from EMA20 re-cross + EMA50 trend + RSI and SPY regime filters. A directional signal is not an options profitability estimate.")
with tab_contract:
    st.caption("Quotes from Yahoo Finance may be stale. Verify contract specs, liquidity, bid/ask, FX and all fees in Dime.")
    try:
        expiries=get_expiries(symbol)
        if not expiries: st.info("No options expiries returned for this ticker.")
        else:
            expiry=st.selectbox("Expiry",expiries)
            calls,puts=get_chain(symbol,expiry)
            kind=st.radio("Contract type",["CALL","PUT"],horizontal=True)
            oc=calls if kind=="CALL" else puts
            cols=[c for c in ["contractSymbol","strike","bid","ask","volume","openInterest","impliedVolatility"] if c in oc]
            st.dataframe(oc[cols].head(100),hide_index=True,use_container_width=True)
            valid=oc[(oc["ask"]>0)&(oc["bid"]>=0)&(oc["bid"]<=oc["ask"])].copy()
            if valid.empty: st.warning("No usable bid/ask quotes.")
            else:
                contract=st.selectbox("Inspect contract",valid["contractSymbol"].astype(str).tolist())
                chosen=valid.loc[valid["contractSymbol"]==contract].iloc[0]
                fees=st.number_input("Commission USD per side, estimated",value=0.55,min_value=0.0,step=0.05)
                extra=st.number_input("Other charges USD per side, estimated",value=0.05,min_value=0.0,step=0.01)
                multiplier=st.number_input("Contract multiplier (confirm with broker)",value=100,min_value=1,step=1)
                result=option_risk_check(equity_thb=account,risk_pct=risk,fx_thb_per_usd=usd_thb,
                    ask_usd=float(chosen["ask"]),bid_usd=float(chosen["bid"]),multiplier=int(multiplier),
                    commission_usd=fees,other_fees_usd=extra)
                a,b,c=st.columns(3)
                a.metric("Premium + entry fees",f'฿{result["cost_thb"]:,.2f}')
                b.metric("Worst case / contract",f'฿{result["max_loss_thb"]:,.2f}')
                c.metric("Quoted spread",f'{result["spread_pct"]:.1f}%')
                if signal=="HOLD" or (signal=="CALL_BIAS" and kind!="CALL") or (signal=="PUT_BIAS" and kind!="PUT"):
                    st.error("NO TRADE: contract direction does not match a qualifying technical signal.")
                elif not result["approved"]:
                    st.error("NO TRADE: one contract exceeds allowable maximum full-premium loss or account buying power.")
                else:
                    st.success("Preliminary cost/risk check passes. Verify live contract liquidity and all broker conditions manually.")
                if float(chosen.get("openInterest",0) or 0)<100:
                    st.warning("Low/unknown open interest: independently verify liquidity.")
    except Exception as exc:
        st.warning("Unable to load contract chain: "+str(exc))
with tab_test:
    st.caption("Underlying-only OHLC proxy, not historical option premium P/L. Stop-first on ambiguous same-day OHLC bars; next-open entry.")
    split=max(70,int(len(data)*.70))
    outputs=[]
    for label,start,end in [("Earlier 70%",60,split),("Final 30%",split,len(data))]:
        trades,stat=backtest_atr(data,start=start,end=end,hold_days=hold,
            stop_atr=atr_stop,target_r=target,trailing_atr=trailing,fee_bps=fee)
        outputs.append({"period":label,**stat})
        if label=="Final 30%" and not trades.empty:
            st.download_button("Export holdout trades",trades.to_csv(index=False),file_name=f"{symbol}_holdout.csv")
    st.dataframe(pd.DataFrame(outputs),hide_index=True,use_container_width=True)
    st.warning("Never tune settings repeatedly against the Final 30% and then describe it as untouched out-of-sample. No actual option backtest, walk-forward PPO training, or broker execution is provided.")
