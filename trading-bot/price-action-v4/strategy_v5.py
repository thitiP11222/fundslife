"""Conservative, causal V5 filters. Research only; not an option pricing model."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scanner import signals,backtest

def filtered_signals(asset:pd.DataFrame,spy:pd.DataFrame,*,max_atr_pct:float=.045,
                     max_extension_atr:float=1.0,min_trend_pct:float=.008)->pd.DataFrame:
    x=signals(asset,spy).copy()
    # Filters use only information known at signal bar close.
    atr_ratio=x["atr"]/x["Close"]
    extension=(x["Close"]-x["ema20"]).abs()/x["atr"].replace(0,np.nan)
    trend_gap=(x["ema20"]-x["ema50"]).abs()/x["Close"]
    liquidity=x["Volume"].rolling(20).mean()>0
    eligible=(atr_ratio<=max_atr_pct)&(extension<=max_extension_atr)&(trend_gap>=min_trend_pct)&liquidity
    eligible=eligible.fillna(False)
    x["unfiltered_signal"]=x["signal"].copy()
    x["eligible"]=eligible
    x["signal"]=np.where(eligible,x["signal"],0).astype(int)
    x["action"]=pd.Series(x["signal"],index=x.index).map({1:"CALL_BIAS",-1:"PUT_BIAS",0:"WAIT"})
    return x

def backtest_with_gap_filter(x:pd.DataFrame,*,max_gap_atr:float=.5,
                             hold:int=5,stop_atr:float=1.5,rr:float=2.,
                             fee_bps:float=10.,slippage_bps:float=5.,
                             start:int=65,end:int|None=None):
    """Reject next-open gap entries without future information.
    Delegate stop/target daily OHLC simulation to same V4 baseline engine.
    """
    if max_gap_atr<0:raise ValueError("max_gap_atr must be nonnegative")
    y=x.copy()
    # At signal close, we cannot know next open. Convert signal to a trade eligibility
    # decision on the entry session only; historical simulator can examine that open.
    # We mask the prior signal, not revise any feature.
    open_next=y["Open"].shift(-1)
    gap=(open_next-y["Close"]).abs()
    eligible=(gap<=max_gap_atr*y["atr"]).fillna(False)
    y["signal"]=np.where(eligible,y["signal"],0).astype(int)
    return backtest(y,hold=hold,stop_atr=stop_atr,rr=rr,fee_bps=fee_bps,
                    slippage_bps=slippage_bps,start=start,end=end)

def compare_v4_v5(asset:pd.DataFrame,spy:pd.DataFrame,*,hold:int=5,stop_atr:float=1.5,rr:float=2.):
    baseline=signals(asset,spy)
    v5=filtered_signals(asset,spy)
    split=max(90,int(len(baseline)*.7))
    rows=[]
    for name,x in (("V4 Base",baseline),("V5 Filtered",v5)):
        for label,start,end in (("Earlier 70%",65,split),("Later 30%",split,len(x))):
            if name=="V4 Base":
                deals,stat=backtest(x,hold=hold,stop_atr=stop_atr,rr=rr,start=start,end=end)
            else:
                deals,stat=backtest_with_gap_filter(x,hold=hold,stop_atr=stop_atr,rr=rr,start=start,end=end)
            rows.append({"strategy":name,"window":label,**stat})
    return pd.DataFrame(rows)


def filter_diagnostics(asset:pd.DataFrame,spy:pd.DataFrame,*,max_atr_pct:float=.045,
                       max_extension_atr:float=1.0,min_trend_pct:float=.008,
                       max_gap_atr:float=.5)->pd.DataFrame:
    """Count rejects among *V4 signal bars*, without conflating with V5 trade count.

    Opening-gap information is only known on next session open, not at signal close.
    This diagnostic is historical analysis, not a live signal predictor.
    """
    x=signals(asset,spy)
    eligible=x["signal"]!=0
    atr=x["atr"].replace(0,np.nan)
    checks={
        "V4 raw signals":pd.Series(True,index=x.index),
        "ATR / price <= limit":(x["atr"]/x["Close"]<=max_atr_pct),
        "EMA20 extension <= limit":((x["Close"]-x["ema20"]).abs()/atr<=max_extension_atr),
        "EMA trend gap >= minimum":((x["ema20"]-x["ema50"]).abs()/x["Close"]>=min_trend_pct),
        "Volume data valid":(x["Volume"].rolling(20).mean()>0),
        "Next opening gap <= limit":((x["Open"].shift(-1)-x["Close"]).abs()<=max_gap_atr*x["atr"]),
    }
    rows=[]
    surviving=eligible.copy()
    for name,ok in checks.items():
        passed=(eligible&ok.fillna(False)).sum()
        surviving=surviving & ok.fillna(False)
        rows.append({"filter":name,"passes_individually":int(passed),
                     "remaining_after_previous_filters":int(surviving.sum()),
                     "rejected_at_this_step":int((eligible&~surviving).sum()) if name=="V4 raw signals" else None})
    for j in range(1,len(rows)):
        rows[j]["rejected_at_this_step"]=rows[j-1]["remaining_after_previous_filters"]-rows[j]["remaining_after_previous_filters"]
    return pd.DataFrame(rows)
