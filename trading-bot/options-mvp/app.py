"""Fundslife AI Stock Scanner — paper/research only, not an order router."""
from __future__ import annotations
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.market import history,chain
from src.engine import features, signals, backtest_underlying
from src.risk import option_risk_check
from src.alerts import dispatch_telegram
from src.strategy_v2 import compare

st.set_page_config(page_title="Fundslife | Options Swing Lab",layout="wide")
st.title("Fundslife | AI Stock Scanner")
st.caption("Daily swing signals + options risk gate · Manual execution in Dime · Not investment advice")
with st.sidebar:
    st.header("Account & strategy")
    equity=st.number_input("Capital (THB)",min_value=1.0,value=1000.0,step=100.0)
    risk=st.number_input("Risk per trade (%)",min_value=0.1,max_value=100.0,value=1.0,step=0.1)
    fx=st.number_input("THB per USD — enter live conversion",min_value=1.0,value=35.0,step=0.1)
    tickers=st.text_input("Watchlist (comma-separated)","SPY,QQQ,NVDA,AMD,AAPL")
    hold=st.slider("Underlying backtest holding days",1,15,5)
    fee=st.number_input("Underlying proxy fees (bps per side)",min_value=0.0,value=10.0)
    st.info("Risk rule: one option contract must fit worst-case premium loss, not just stop-loss.")
symbols=list(dict.fromkeys(s.strip().upper() for s in tickers.split(",") if s.strip()))[:12]
data={};rows=[]
for symbol in symbols:
    try:
        df=history(symbol)
        sig=signals(features(df))
        last=sig.iloc[-1]
        rows.append({"symbol":symbol,"price_usd":round(float(last["Close"]),2),
                     "signal":str(last["action"]),"rsi":round(float(last["rsi"]),1),
                     "ema20":round(float(last["ema20"]),2),
                     "ema50":round(float(last["ema50"]),2),
                     "atr":round(float(last["atr14"]),2),
                     "volume_ratio":round(float(last["volume_ratio"]),2),
                     "last_bar":str(sig.index[-1].date())})
        data[symbol]=df
    except Exception as e:
        st.warning(f"{symbol}: {e}")
if not rows:
    st.error("No valid market data. Retry later or use different symbols.")
    st.stop()
scan=pd.DataFrame(rows)
st.subheader("Scanner — underlying directional bias only")
st.dataframe(scan,use_container_width=True,hide_index=True)
st.caption("CALL_BIAS / PUT_BIAS are technical research flags, not option-contract recommendations.")
selected=st.selectbox("Inspect stock / ETF",list(data))
st.subheader("V2 Research: Regime + EMA Pullback")
st.caption("Compares V1 and V2 on the same asset and chronological split. Underlying-only proxy; not option P&L.")

df=data[selected]
sig=signals(features(df))
try:
    benchmark=data.get("SPY")
    if benchmark is None:
        benchmark=history("SPY")
    v2_signal,comparison=compare(df,benchmark,hold_days=hold,fee_bps=fee)
    st.metric("Latest V2 directional signal",str(v2_signal.iloc[-1]["action"]))
    st.dataframe(comparison,use_container_width=True,hide_index=True)
    st.caption("Out-of-sample = final 30% of daily history. No parameters trained on that holdout; validate across more tickers and periods before use.")
except Exception as exc:
    st.warning(f"V2 comparison unavailable: {exc}")
left,right=st.columns([2,1])
with left:
    st.subheader(f"{selected} technical chart")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=sig.index,y=sig["Close"],name="Close"))
    fig.add_trace(go.Scatter(x=sig.index,y=sig["ema20"],name="EMA20"))
    fig.add_trace(go.Scatter(x=sig.index,y=sig["ema50"],name="EMA50"))
    fig.update_layout(height=390,xaxis_title="Daily bars",yaxis_title="USD")
    st.plotly_chart(fig,use_container_width=True)
with right:
    st.subheader("Latest signal")
    last=sig.iloc[-1]
    st.metric("Direction",last["action"])
    st.metric("Risk budget",f"฿{equity*risk/100:,.2f}")
    st.write("RSI:",round(float(last["rsi"]),1))
    st.write("ATR:",round(float(last["atr14"]),2))
    st.write("As-of session:",sig.index[-1].date())
st.subheader("Underlying proxy backtest")
_,trades,stats=backtest_underlying(df,fee_bps=fee,hold_days=hold)
a,b,c,d=st.columns(4)
a.metric("Trades",stats["trades"]);b.metric("Win rate",f'{stats["win_rate"]:.1%}')
c.metric("Compounded return",f'{stats["compounded_return"]:.1%}')
d.metric("Max drawdown",f'{stats["max_drawdown"]:.1%}')
st.caption("NOT an options backtest: next-open executions, directional underlying returns, fixed holding days; no IV, Greeks, option bid-ask or expiration.")
if not trades.empty:
    st.dataframe(trades.tail(50),use_container_width=True,hide_index=True)
    st.download_button("Download proxy trades CSV",trades.to_csv(index=False),file_name=f"{selected}_proxy_trades.csv")
st.subheader("Options Chain — manual contract feasibility")
st.caption("Yahoo option quotes may be stale. Verify expiry, multiplier, spread, available buying power and fees inside Dime before any order.")
try:
    import yfinance as yf
    expiries=list(yf.Ticker(selected).options)
    if expiries:
        expiry=st.selectbox("Expiration",expiries)
        calls,puts=chain(selected,expiry)
        side=st.radio("Type",["CALL","PUT"],horizontal=True)
        oc=calls if side=="CALL" else puts
        cols=["contractSymbol","strike","bid","ask","volume","openInterest","impliedVolatility"]
        show=oc[[c for c in cols if c in oc.columns]].copy()
        st.dataframe(show.head(80),use_container_width=True,hide_index=True)
        valid=oc[(oc["ask"]>0)&(oc["bid"]>=0)&(oc["ask"]>=oc["bid"])].copy()
        if valid.empty:
            st.warning("No usable quotes from provider")
        else:
            valid=valid.sort_values("strike")
            contract=st.selectbox("Contract to risk-check",valid["contractSymbol"].astype(str).tolist())
            row=valid.loc[valid["contractSymbol"]==contract].iloc[0]
            commission=st.number_input("Estimated commission USD per side",min_value=0.,value=0.55,step=0.05)
            other_fees=st.number_input("Estimated other fees USD per side",min_value=0.,value=0.05,step=0.01)
            multiplier=st.number_input("Contract multiplier — verify contract specification",min_value=1,value=100)
            check=option_risk_check(equity_thb=equity,risk_pct=risk,fx_thb_per_usd=fx,
                                    ask_usd=float(row["ask"]),bid_usd=float(row["bid"]),
                                    multiplier=int(multiplier),commission_usd=commission,
                                    other_fees_usd=other_fees)
            st.write(check)
            if check["approved"]:
                st.success("Within strict full-premium risk budget based on entered assumptions")
            else:
                st.error("NO TRADE: contract fails strict full-premium risk budget")
    else:
        st.info("No available option-chain expiries from Yahoo Finance")
except Exception as e:
    st.warning(f"Options chain currently unavailable: {e}")
st.subheader("Telegram alert — on demand")
st.caption("Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env variables. This page does not run automatically when closed.")
if st.button("Send current scan summary to Telegram"):
    msg="Fundslife daily scanner (informational)\n"+"\n".join(f'{r["symbol"]}: {r["signal"]} as of {r["last_bar"]}' for r in rows)
    try:
        if dispatch_telegram(msg):
            st.success("Alert submitted to Telegram")
        else:
            st.warning("Telegram credentials are not set")
    except Exception as e:
        st.error(f"Telegram failed: {e}")
