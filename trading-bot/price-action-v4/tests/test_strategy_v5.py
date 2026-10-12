import numpy as np
import pandas as pd
from strategy_v5 import filtered_signals, backtest_with_gap_filter,compare_v4_v5

def bars():
    idx=pd.date_range("2024-01-01",periods=180,freq="B")
    c=100+np.linspace(0,20,len(idx))+np.sin(np.arange(len(idx))/5)
    return pd.DataFrame({"Open":c,"High":c+1,"Low":c-1,"Close":c,"Volume":10000},index=idx)

def test_v5_never_adds_trade():
    a=bars()
    x=filtered_signals(a,a)
    assert (x["signal"]!=0).sum()<=(x["unfiltered_signal"]!=0).sum()
    assert len(compare_v4_v5(a,a))==4

def test_filter_uses_no_future_close():
    a=bars()
    baseline=filtered_signals(a,a).iloc[-2]["signal"]
    a.iloc[-1,a.columns.get_loc("Close")]=500
    assert filtered_signals(a,a).iloc[-2]["signal"]==baseline

def test_entry_gap_rejects_signal():
    a=bars()
    from scanner import signals
    x=signals(a,a)
    x["signal"]=0
    x.iloc[100,x.columns.get_loc("signal")]=1
    x.iloc[101,x.columns.get_loc("Open")]=x.iloc[100]["Close"]+25
    trades,st=backtest_with_gap_filter(x,start=100,max_gap_atr=.5)
    assert trades.empty and st["trades"]==0
