"""Leakage-conscious, daily-bar underlying swing signal and next-open backtest."""
from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED = {"Open", "High", "Low", "Close", "Volume"}

def features(frame: pd.DataFrame) -> pd.DataFrame:
    if not REQUIRED.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(REQUIRED-set(frame.columns))}")
    x = frame.copy().sort_index()
    for col in REQUIRED:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=list(REQUIRED))
    if (x["Close"] <= 0).any() or len(x) < 65:
        raise ValueError("Need >=65 valid positive-close daily bars")
    c, h, l = x["Close"], x["High"], x["Low"]
    x["ema20"] = c.ewm(span=20, adjust=False).mean()
    x["ema50"] = c.ewm(span=50, adjust=False).mean()
    change = c.diff()
    gain = change.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = (100 - 100/(1+rs)).fillna(50)
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    x["atr14"] = tr.ewm(alpha=1/14, adjust=False).mean()
    x["volume_ratio"] = x["Volume"] / x["Volume"].rolling(20).mean().replace(0, np.nan)
    x["breakout20"] = h.shift(1).rolling(20).max()
    x["breakdown20"] = l.shift(1).rolling(20).min()
    x["ret5"] = c.pct_change(5)
    x["vol20"] = c.pct_change().rolling(20).std()
    x["trend_strength"] = (x["ema20"] / x["ema50"] - 1)
    return x.replace([np.inf, -np.inf], np.nan)

def signals(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    long = ((out["Close"] > out["breakout20"]) & (out["ema20"] > out["ema50"]) &
            out["rsi"].between(50,75) & (out["volume_ratio"] >= 0.8))
    short = ((out["Close"] < out["breakdown20"]) & (out["ema20"] < out["ema50"]) &
             out["rsi"].between(25,50) & (out["volume_ratio"] >= 0.8))
    out["signal"] = np.select([long, short], [1,-1], default=0).astype(int)
    out["action"] = out["signal"].map({1:"CALL_BIAS", -1:"PUT_BIAS", 0:"HOLD"})
    return out

def backtest_underlying(df: pd.DataFrame, *, fee_bps: float=10., hold_days: int=5) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Signal at close t; simulated entry at t+1 open, exit at next open after hold_days.
    Underlying-only, equal notional, no options premium/greeks/execution approximation.
    """
    if hold_days < 1 or fee_bps < 0:
        raise ValueError("hold_days >=1 and fee_bps >=0 required")
    x = signals(features(df))
    trades=[]
    i=50
    while i < len(x)-hold_days-1:
        if int(x["signal"].iloc[i]) == 0:
            i+=1
            continue
        side=int(x["signal"].iloc[i])
        entry_i=i+1
        exit_i=entry_i+hold_days
        p0=float(x["Open"].iloc[entry_i]); p1=float(x["Open"].iloc[exit_i])
        if p0 <= 0 or p1 <= 0:
            i=exit_i+1
            continue
        gross=side*(p1/p0-1)
        net=gross-2*fee_bps/10000
        trades.append({"signal_date":str(x.index[i].date()),"entry_date":str(x.index[entry_i].date()),
                       "exit_date":str(x.index[exit_i].date()),"direction":"CALL_BIAS" if side==1 else "PUT_BIAS",
                       "entry":p0,"exit":p1,"gross_return":gross,"net_return":net})
        i=exit_i+1
    result=pd.DataFrame(trades)
    if result.empty:
        return x, result, {"trades":0,"win_rate":0.,"average_return":0.,"compounded_return":0.,"max_drawdown":0.}
    equity=(1+result["net_return"].clip(lower=-1)).cumprod()
    drawdown=equity/equity.cummax()-1
    stats={"trades":int(len(result)), "win_rate":float((result["net_return"]>0).mean()),
           "average_return":float(result["net_return"].mean()),
           "compounded_return":float(equity.iloc[-1]-1),
           "max_drawdown":float(drawdown.min())}
    return x, result, stats
