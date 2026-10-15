"""Causal, rule-based technical analysis: confirmed pivot extrema and breakouts.

All signals evaluated on the last COMPLETED input bar. The caller must not feed
a still-forming candle. Signal 'entry' is a proposed trigger, not an executed trade.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

@dataclass
class Signal:
    pattern: str
    direction: str
    status: str
    entry: float
    stop: float
    target: float
    score: int
    explanation: str
    detected_at: str
    def as_dict(self):
        return asdict(self)

def indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"].astype(float)
    d["EMA20"] = c.ewm(span=20, adjust=False).mean()
    d["EMA50"] = c.ewm(span=50, adjust=False).mean()
    delta = c.diff()
    gains = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gains / losses.replace(0, np.nan)
    d["RSI14"] = (100 - 100 / (1 + rs)).fillna(50)
    prev = c.shift()
    tr = pd.concat([(d["High"]-d["Low"]).abs(),(d["High"]-prev).abs(),(d["Low"]-prev).abs()],axis=1).max(axis=1)
    d["ATR14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    d["Prior20High"] = d["High"].shift(1).rolling(20).max()
    d["Prior20Low"] = d["Low"].shift(1).rolling(20).min()
    d["VolAvg20"] = d["Volume"].shift(1).rolling(20).mean()
    return d

def _stamp(d: pd.DataFrame) -> str:
    return str(d.index[-1])

def _signal(name, direction, entry, stop, risk_reward, score, explanation, d):
    risk = abs(float(entry)-float(stop))
    if risk <= 0:
        return None
    target = float(entry) + (risk_reward * risk if direction == "LONG" else -risk_reward * risk)
    return Signal(name, direction, "Confirmed on latest closed bar", round(float(entry),4),
                  round(float(stop),4), round(target,4), int(score), explanation, _stamp(d))

def _confirmed_pivots(values, kind, left=3, right=3):
    """Pivot at i only becomes knowable at i+right; no future-bar use at evaluation time."""
    out=[]
    for i in range(left,len(values)-right):
        window=values[i-left:i+right+1]
        v=values[i]
        if kind=="low" and v==min(window) and window.count(v)==1:
            out.append(i)
        if kind=="high" and v==max(window) and window.count(v)==1:
            out.append(i)
    return out

def scan(df: pd.DataFrame, rr: float=2.0, tolerance: float=0.025, max_age: int=10):
    """Return (enriched dataframe, latest confirmed signals), no hypothetical fills."""
    if len(df)<65: raise ValueError("Requires at least 65 completed OHLCV bars")
    if rr<=0 or not 0<tolerance<0.2 or max_age<1: raise ValueError("Invalid scanner parameters")
    d=indicators(df)
    last=d.iloc[-1]
    close=float(last["Close"])
    atr=float(last["ATR14"])
    if not np.isfinite(atr) or atr<=0: return d,[]
    signals=[]
    high=float(last["Prior20High"])
    low=float(last["Prior20Low"])
    volume_ok=bool(last["Volume"]>=last["VolAvg20"]*1.1) if np.isfinite(last["VolAvg20"]) else False
    bullish=bool(last["EMA20"]>last["EMA50"])
    bearish=bool(last["EMA20"]<last["EMA50"])
    if close>high and volume_ok:
        s=_signal("20-bar Breakout","LONG",close,min(high,close-1.5*atr),rr,70+(15 if bullish else 0),
                 "Close broke above the PREVIOUS 20-bar high with volume >= 1.1x prior 20-bar average.",d)
        if s:signals.append(s)
    if close<low and volume_ok:
        s=_signal("20-bar Breakdown","SHORT",close,max(low,close+1.5*atr),rr,70+(15 if bearish else 0),
                 "Close broke below the PREVIOUS 20-bar low with volume >= 1.1x prior 20-bar average.",d)
        if s:signals.append(s)

    # Double bottom/top neckline confirmations. Look only at extrema confirmed
    # before this bar, and require neckline crossing on the latest bar.
    lows=_confirmed_pivots(d["Low"].astype(float).tolist(),"low")
    highs=_confirmed_pivots(d["High"].astype(float).tolist(),"high")
    n=len(d)
    for kind, pivots, direction in [("Double Bottom",lows,"LONG"),("Double Top",highs,"SHORT")]:
        if len(pivots)<2:continue
        p1,p2=pivots[-2:]
        if not (5<=p2-p1<=55 and n-1-p2<=max_age):continue
        a,b=(float(d["Low"].iloc[j]) if direction=="LONG" else float(d["High"].iloc[j]) for j in [p1,p2])
        if abs(a-b)/max(abs(a),abs(b),1e-9)>tolerance:continue
        middle=d.iloc[p1:p2+1]
        neckline=float(middle["High"].max() if direction=="LONG" else middle["Low"].min())
        previous=float(d["Close"].iloc[-2])
        crossed=(previous<=neckline<close) if direction=="LONG" else (previous>=neckline>close)
        if not crossed:continue
        stop=(min(a,b)-0.5*atr) if direction=="LONG" else (max(a,b)+0.5*atr)
        if (direction=="LONG" and stop>=close) or (direction=="SHORT" and stop<=close):continue
        s=_signal(kind,direction,close,stop,rr,80,
                 f"Two confirmed swing {'lows' if direction=='LONG' else 'highs'} within {tolerance:.1%} tolerance; latest close crossed neckline {neckline:.2f}.",d)
        if s: signals.append(s)
    return d, sorted(signals,key=lambda s:s.score,reverse=True)

def position_size(entry:float,stop:float,capital:float=1000,risk_percent:float=1.0,max_allocation_percent:float=60.0):
    """Fractional-share theoretical sizing; no leverage, ignores fees/min order sizes."""
    if capital<=0 or not 0<risk_percent<=100 or not 0<max_allocation_percent<=100:
        raise ValueError("Invalid position-sizing inputs")
    distance=abs(entry-stop)
    if distance<=0 or entry<=0:raise ValueError("Invalid entry/stop")
    risk_cap=capital*risk_percent/100
    allocation_cap=capital*max_allocation_percent/100
    shares=min(risk_cap/distance, allocation_cap/entry)
    return {"shares":round(shares,6),"notional":round(shares*entry,2),
            "planned_risk":round(shares*distance,2),"risk_budget":round(risk_cap,2)}
