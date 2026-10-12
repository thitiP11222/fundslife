"""Explainable research plan; no options contract eligibility implied."""
from __future__ import annotations
import pandas as pd
from scanner import structure,pattern

def describe(x:pd.DataFrame)->dict:
    if x.empty:raise ValueError("No analysis bars")
    last=x.iloc[-1]
    direction=str(last["action"])
    spot=float(last["Close"])
    support=float(last["support"]) if pd.notna(last["support"]) else None
    resistance=float(last["resistance"]) if pd.notna(last["resistance"]) else None
    atr=float(last["atr"])
    if direction=="CALL_BIAS":
        trigger=resistance; invalidation=support if support is not None and support<spot else spot-1.5*atr
        reason="Confirmed resistance breakout with positive SPY regime and trend filters"
    elif direction=="PUT_BIAS":
        trigger=support; invalidation=resistance if resistance is not None and resistance>spot else spot+1.5*atr
        reason="Confirmed support breakdown with negative SPY regime and trend filters"
    else:
        trigger=None;invalidation=None
        reason="No confirmed entry under current filters; pattern alone does not authorize a trade"
    return {"ticker_trend":structure(x),"pattern_candidate":pattern(x),
            "signal":direction,"last_price":spot,"support":support,"resistance":resistance,
            "entry_trigger":trigger,"invalidation":invalidation,"reason":reason,
            "disclaimer":"Levels are research estimates based on confirmed daily pivots, not order instructions."}
