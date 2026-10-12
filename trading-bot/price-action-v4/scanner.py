"""Causal daily price-action scanner: pivots become visible only after right bars."""
from __future__ import annotations
import numpy as np
import pandas as pd

def indicators(df: pd.DataFrame, left: int=3, right: int=3) -> pd.DataFrame:
    if not {"Open","High","Low","Close","Volume"}.issubset(df.columns):
        raise ValueError("OHLCV columns required")
    if len(df)<80 or left<1 or right<1:
        raise ValueError("Need >=80 bars and positive pivot windows")
    x=df[["Open","High","Low","Close","Volume"]].astype(float).dropna().sort_index().copy()
    c=x.Close
    x["ema20"]=c.ewm(span=20,adjust=False).mean()
    x["ema50"]=c.ewm(span=50,adjust=False).mean()
    tr=pd.concat([x.High-x.Low,(x.High-c.shift()).abs(),(x.Low-c.shift()).abs()],axis=1).max(axis=1)
    x["atr"]=tr.ewm(alpha=1/14,adjust=False).mean()
    delta=c.diff()
    gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    x["rsi"]=(100-100/(1+gain/loss.replace(0,np.nan))).fillna(50)
    x["vol_ratio"]=x.Volume/x.Volume.rolling(20).mean().replace(0,np.nan)
    x["ph"]=np.nan; x["pl"]=np.nan
    h=x.High.to_numpy(); l=x.Low.to_numpy()
    # Pivot at t is only confirmed at t+right. Avoid plotting/using it at t.
    for i in range(left+right,len(x)):
        pivot=i-right
        if h[pivot]>np.max(np.r_[h[pivot-left:pivot],h[pivot+1:i+1]]):
            x.iloc[i,x.columns.get_loc("ph")]=h[pivot]
        if l[pivot]<np.min(np.r_[l[pivot-left:pivot],l[pivot+1:i+1]]):
            x.iloc[i,x.columns.get_loc("pl")]=l[pivot]
    x["resistance"]=x.ph.ffill()
    x["support"]=x.pl.ffill()
    # A breakout must exceed a PREVIOUSLY KNOWN level.
    resistance=x.resistance.shift(1)
    support=x.support.shift(1)
    buffer=x.atr*0.10
    x["breakout"]=(c>resistance+buffer)&(c.shift(1)<=resistance.shift(1)+buffer.shift(1))
    x["breakdown"]=(c<support-buffer)&(c.shift(1)>=support.shift(1)-buffer.shift(1))
    return x

def structure(x:pd.DataFrame)->str:
    highs=x.ph.dropna().tail(2).to_list()
    lows=x.pl.dropna().tail(2).to_list()
    if len(highs)<2 or len(lows)<2:return "RANGE / INSUFFICIENT PIVOTS"
    if highs[-1]>highs[-2] and lows[-1]>lows[-2]:return "UPTREND (HH/HL)"
    if highs[-1]<highs[-2] and lows[-1]<lows[-2]:return "DOWNTREND (LH/LL)"
    return "MIXED / RANGE"

def pattern(x:pd.DataFrame)->str:
    """Heuristic swing-pattern candidates, not ML forecasts."""
    lows=x.pl.dropna().tail(2)
    highs=x.ph.dropna().tail(2)
    if len(lows)==2 and abs(lows.iloc[-1]-lows.iloc[-2])<=x.atr.iloc[-1]*0.65:
        if x.Close.iloc[-1]>x.ema20.iloc[-1] and x.Close.iloc[-1]>x.Close.iloc[-2]:
            return "POSSIBLE DOUBLE BOTTOM"
    if len(highs)==2 and abs(highs.iloc[-1]-highs.iloc[-2])<=x.atr.iloc[-1]*0.65:
        if x.Close.iloc[-1]<x.ema20.iloc[-1] and x.Close.iloc[-1]<x.Close.iloc[-2]:
            return "POSSIBLE DOUBLE TOP"
    if bool(x.breakout.iloc[-1]):return "RESISTANCE BREAKOUT"
    if bool(x.breakdown.iloc[-1]):return "SUPPORT BREAKDOWN"
    return "NO CONFIRMED PATTERN"

def signals(asset:pd.DataFrame, spy:pd.DataFrame)->pd.DataFrame:
    x=indicators(asset)
    m=indicators(spy)
    # Align regime only to benchmark bars that existed at signal date.
    trend=m[["Close","ema20","ema50"]].rename(columns={"Close":"spy_close","ema20":"spy_ema20","ema50":"spy_ema50"})
    x=x.join(trend.reindex(x.index,method="ffill"))
    risk_on=(x.spy_close>x.spy_ema50)&(x.spy_ema20>x.spy_ema50)
    risk_off=(x.spy_close<x.spy_ema50)&(x.spy_ema20<x.spy_ema50)
    bullish=(x.breakout)&(x.Close>x.ema20)&(x.ema20>x.ema50)&risk_on&(x.vol_ratio>=.85)&x.rsi.between(45,75)
    bearish=(x.breakdown)&(x.Close<x.ema20)&(x.ema20<x.ema50)&risk_off&(x.vol_ratio>=.85)&x.rsi.between(25,55)
    x["signal"]=np.select([bullish,bearish],[1,-1],default=0).astype(int)
    x["action"]=x.signal.map({1:"CALL_BIAS",-1:"PUT_BIAS",0:"WAIT"})
    return x

def backtest(x:pd.DataFrame, *, hold:int=5, stop_atr:float=1.5, rr:float=2., fee_bps:float=10., slippage_bps:float=5., start:int=65, end:int|None=None):
    """Next-day-open entry; stop/target tested from entry bar onward; pessimistic stop-first ambiguity."""
    if hold<1 or stop_atr<=0 or rr<=0 or fee_bps<0 or slippage_bps<0:raise ValueError("Invalid parameters")
    end=min(len(x),end if end is not None else len(x))
    deals=[]; i=max(start,1)
    while i+1<end:
        side=int(x.signal.iloc[i])
        atr=float(x.atr.iloc[i])
        if side==0 or not np.isfinite(atr) or atr<=0:
            i+=1;continue
        at=i+1; opening=float(x.Open.iloc[at])
        if opening<=0:i+=1;continue
        slip=slippage_bps/10000
        entry=opening*(1+side*slip)
        stop=entry-side*atr*stop_atr
        target=entry+side*atr*stop_atr*rr
        expiry=min(at+hold-1,end-1)
        for j in range(at,expiry+1):
            op,hi,lo,close=(float(x[k].iloc[j]) for k in ("Open","High","Low","Close"))
            st=(op<=stop if side==1 else op>=stop)
            tg=(op>=target if side==1 else op<=target)
            if st: exit_price=op;why="STOP_GAP"
            elif tg: exit_price=op;why="TARGET_GAP"
            elif lo<=stop if side==1 else hi>=stop:exit_price=stop;why="STOP"
            elif hi>=target if side==1 else lo<=target:exit_price=target;why="TARGET"
            elif j==expiry:exit_price=close;why="TIME"
            else:continue
            exit_fill=exit_price*(1-side*slip)
            ret=side*(exit_fill/entry-1)-2*fee_bps/10000
            deals.append({"signal_date":str(x.index[i].date()),"entry_date":str(x.index[at].date()),"exit_date":str(x.index[j].date()),"direction":"CALL_BIAS" if side==1 else "PUT_BIAS","reason":why,"net_return":ret})
            i=j+1
            break
        else:i=expiry+1
    trades=pd.DataFrame(deals)
    if trades.empty:return trades,{"trades":0,"win_rate":0.,"profit_factor":None,"expectancy":0.,"max_drawdown":0.}
    rets=trades.net_return
    gains=rets[rets>0].sum(); losses=-rets[rets<0].sum()
    eq=(1+rets.clip(lower=-1)).cumprod().reset_index(drop=True)
    peak=pd.concat([pd.Series([1.]),eq],ignore_index=True).cummax().iloc[1:].reset_index(drop=True)
    return trades,{"trades":len(trades),"win_rate":float((rets>0).mean()),"profit_factor":float(gains/losses) if losses>0 else None,"expectancy":float(rets.mean()),"max_drawdown":float((eq/peak-1).min())}
