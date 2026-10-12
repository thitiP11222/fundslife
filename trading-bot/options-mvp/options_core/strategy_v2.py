"""Regime-filtered trend pullback, using only close-of-bar information."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .engine import features, signals

def strategy_v2(asset: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    a=features(asset)
    b=features(benchmark)
    # Market index must use only data from same or earlier date.
    regime=b[["Close","ema20","ema50"]].rename(columns={"Close":"spy_close","ema20":"spy_ema20","ema50":"spy_ema50"})
    a=a.join(regime.reindex(a.index,method="ffill"))
    up=(a["spy_close"]>a["spy_ema20"])&(a["spy_ema20"]>a["spy_ema50"])
    down=(a["spy_close"]<a["spy_ema20"])&(a["spy_ema20"]<a["spy_ema50"])
    # Yesterday's price was on / past the other side of EMA20; today confirms recovery.
    bull=(a["Close"]>a["ema20"])&(a["Close"].shift(1)<=a["ema20"].shift(1))&(
        a["ema20"]>a["ema50"])&(a["Close"]>a["Close"].shift(1))&a["rsi"].between(45,65)&up
    bear=(a["Close"]<a["ema20"])&(a["Close"].shift(1)>=a["ema20"].shift(1))&(
        a["ema20"]<a["ema50"])&(a["Close"]<a["Close"].shift(1))&a["rsi"].between(35,55)&down
    # Volatility cap: avoid exceptionally large daily ranges.
    atr_pct=a["atr14"]/a["Close"]
    valid=(atr_pct<0.065)&(atr_pct>0)&a["spy_close"].notna()
    a["signal"]=np.select([bull&valid,bear&valid],[1,-1],default=0).astype(int)
    a["action"]=a["signal"].map({1:"CALL_BIAS",-1:"PUT_BIAS",0:"HOLD"})
    return a

def evaluate_signals(x:pd.DataFrame,*,hold_days:int=5,fee_bps:float=10.0)->tuple[pd.DataFrame,dict]:
    """Simple non-overlapping directional underlying proxy, fill at next day's open."""
    if hold_days<1 or fee_bps<0: raise ValueError("Invalid test parameters")
    deals=[]
    i=60
    while i+hold_days+1<len(x):
        s=int(x["signal"].iloc[i])
        if s==0:
            i+=1
            continue
        entry=i+1; exit_at=entry+hold_days
        p=float(x["Open"].iloc[entry]);q=float(x["Open"].iloc[exit_at])
        if p>0 and q>0:
            net=s*(q/p-1)-2*fee_bps/10000
            deals.append({"signal_date":str(x.index[i].date()),"entry_date":str(x.index[entry].date()),
                "exit_date":str(x.index[exit_at].date()),"direction":"CALL_BIAS" if s==1 else "PUT_BIAS",
                "net_return":net})
        i=exit_at+1
    t=pd.DataFrame(deals)
    if t.empty:return t,{"trades":0,"win_rate":0.,"mean_return":0.,"compounded_return":0.,"max_drawdown":0.}
    eq=(1+t["net_return"].clip(lower=-1)).cumprod()
    drawdown=eq/eq.cummax()-1
    return t,{"trades":len(t),"win_rate":float((t["net_return"]>0).mean()),
        "mean_return":float(t["net_return"].mean()),
        "compounded_return":float(eq.iloc[-1]-1),"max_drawdown":float(drawdown.min())}

def compare(asset:pd.DataFrame,benchmark:pd.DataFrame,*,hold_days:int=5,fee_bps:float=10.,test_fraction:float=.3):
    """Fixed chronological holdout. No training/tuning on test; descriptive OOS evaluation."""
    if not 0<test_fraction<1:raise ValueError("test_fraction must be between 0 and 1")
    old=signals(features(asset))
    new=strategy_v2(asset,benchmark)
    cutoff=max(70,int(len(new)*(1-test_fraction)))
    records=[]
    for name,x in [("V1 Breakout",old),("V2 Regime Pullback",new)]:
        for segment,subset in [("in_sample",x.iloc[:cutoff]),("out_of_sample",x.iloc[cutoff:])]:
            # maintain indicator history from full frame; only count signals within segment
            # trades entered/exited in same segment, avoiding cross-boundary returns
            tr,stats=evaluate_signals(subset,hold_days=hold_days,fee_bps=fee_bps)
            records.append({"strategy":name,"segment":segment,**stats})
    return new,pd.DataFrame(records)
