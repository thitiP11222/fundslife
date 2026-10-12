import numpy as np
import pandas as pd
import pytest
from src.backtest import backtest, signal_history, walk_forward

def prices(n=100):
    idx=pd.date_range("2025-01-01",periods=n)
    return pd.DataFrame({"Open":[100.]*n,"High":[102.]*n,"Low":[99.]*n,
                         "Close":[100.]*n,"Volume":[100.]*n},index=idx)

def history_at(bar=65,entry=100,stop=95,score=80):
    return pd.DataFrame([{"bar_index":bar,"direction":"LONG","score":score,
                          "entry":entry,"stop":stop,"target":110,"pattern":"test"}])

def test_signal_history_stable_without_future():
    d=prices(85)
    a=signal_history(d.iloc[:80])
    b=signal_history(d)
    assert a.equals(b[b.bar_index<80].reset_index(drop=True))

def test_next_bar_execution_and_risk_cap():
    d=prices()
    d.iloc[66,d.columns.get_loc("Open")]=101
    eq,trades,met=backtest(d,history_at(),start_index=65,end_index=70,fee_bps=0,slippage_bps=0)
    assert len(trades)==1
    assert trades.iloc[0]["entry_date"]==str(d.index[66])
    assert trades.iloc[0]["entry"]==101
    assert trades.iloc[0]["shares"]*(101-95)<=10+1e-8
    assert met["trades"]==1

def test_gap_stop_executes_at_worse_open():
    d=prices()
    d.iloc[66,d.columns.get_loc("High")]=105
    d.iloc[66,d.columns.get_loc("Low")]=99
    d.iloc[67,d.columns.get_loc("Open")]=90
    d.iloc[67,d.columns.get_loc("Low")]=89
    _,trades,_=backtest(d,history_at(),start_index=65,end_index=72,fee_bps=0,slippage_bps=0)
    assert trades.iloc[0]["exit"]==90
    assert trades.iloc[0]["reason"]=="gap_stop"

def test_same_bar_stop_has_priority():
    d=prices()
    d.iloc[66,d.columns.get_loc("High")]=120
    d.iloc[66,d.columns.get_loc("Low")]=90
    _,trades,_=backtest(d,history_at(),start_index=65,end_index=72,fee_bps=0,slippage_bps=0)
    assert trades.iloc[0]["reason"]=="entry_bar_stop"

def test_fees_reduce_equity():
    d=prices()
    h=history_at()
    _,_,a=backtest(d,h,start_index=65,end_index=70,fee_bps=0,slippage_bps=0)
    _,_,b=backtest(d,h,start_index=65,end_index=70,fee_bps=20,slippage_bps=0)
    assert b["total_return_pct"]<a["total_return_pct"]

def test_walk_forward_nonoverlap():
    d=prices(210)
    folds=walk_forward(d,history_at(),train_bars=100,test_bars=40)
    assert len(folds)==3
    assert folds.iloc[0].test_start==str(d.index[100])
    assert folds.iloc[1].test_start==str(d.index[140])

def test_zero_signal_flat_equity():
    eq,trades,metrics=backtest(prices(),pd.DataFrame(),start_index=65,end_index=90)
    assert trades.empty
    assert (eq.equity==1000).all()
    assert metrics["trades"]==0
