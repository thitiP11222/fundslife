"""Yahoo Finance public-data research adapter. Data may be delayed or incomplete."""
from __future__ import annotations
import pandas as pd
import yfinance as yf

def history(symbol:str,period:str="2y")->pd.DataFrame:
    symbol=symbol.strip().upper()
    if not symbol or len(symbol)>20 or not all(c.isalnum() or c in ".^-" for c in symbol):
        raise ValueError("Invalid ticker")
    frame=yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
    if frame.empty:
        raise RuntimeError(f"No Yahoo Finance data returned for {symbol}")
    cols=["Open","High","Low","Close","Volume"]
    x=frame[cols].dropna().copy()
    x.index=pd.DatetimeIndex(x.index).tz_localize(None)
    return x

def chain(symbol:str,expiry:str):
    ticker=yf.Ticker(symbol)
    if expiry not in ticker.options:
        raise ValueError("Expiration unavailable from quote provider")
    oc=ticker.option_chain(expiry)
    return oc.calls.copy(),oc.puts.copy()
