"""Run reproducible cross-ticker, chronological OHLC proxy validation."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf
from scanner import signals,backtest

def fetch(ticker: str, period: str="5y")->pd.DataFrame:
    d=yf.Ticker(ticker).history(period=period,interval="1d",auto_adjust=True)
    if d.empty:raise RuntimeError(f"No historical data for {ticker}")
    out=d[["Open","High","Low","Close","Volume"]].dropna().copy()
    out.index=pd.DatetimeIndex(out.index).tz_localize(None)
    return out

def evaluate(symbols: list[str], *, hold:int=5, stop_atr:float=1.5, rr:float=2.0,
             train_fraction:float=0.7, period:str="5y") -> tuple[pd.DataFrame,pd.DataFrame]:
    if not 0.5<=train_fraction<=0.9:raise ValueError("train_fraction must be 0.5–0.9")
    spy=fetch("SPY",period)
    summaries=[];trades_out=[]
    for sym in dict.fromkeys(s.strip().upper() for s in symbols if s.strip()):
        try:
            x=signals(fetch(sym,period),spy)
            split=max(80,int(len(x)*train_fraction))
            for label,start,end in [("Earlier sample",65,split),("Later holdout",split,len(x))]:
                deals,metrics=backtest(x,start=start,end=end,hold=hold,stop_atr=stop_atr,rr=rr)
                summaries.append({"ticker":sym,"period":label,**metrics,"bars":end-start})
                if not deals.empty:
                    deals.insert(0,"ticker",sym)
                    deals.insert(1,"period",label)
                    trades_out.append(deals)
        except Exception as exc:
            summaries.append({"ticker":sym,"period":"ERROR","error":str(exc)})
    return pd.DataFrame(summaries),pd.concat(trades_out,ignore_index=True) if trades_out else pd.DataFrame()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--tickers",default="SPY,QQQ,NVDA,AMD,AAPL,MSFT")
    p.add_argument("--out",default="reports")
    a=p.parse_args()
    summary,trades=evaluate(a.tickers.split(","))
    output=Path(a.out);output.mkdir(parents=True,exist_ok=True)
    summary.to_csv(output/"v4_summary.csv",index=False)
    trades.to_csv(output/"v4_trades.csv",index=False)
    print(summary.to_string(index=False))
    print(f"Saved results to {output.resolve()}")
