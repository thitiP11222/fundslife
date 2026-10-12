import pandas as pd
from options_core.backtest_v3 import backtest_atr

def bars(side=1):
    dates=pd.date_range("2024-01-01",periods=90,freq="B")
    x=pd.DataFrame({"Open":100.,"High":101.,"Low":99.,"Close":100.,"atr14":2.,"signal":0},index=dates)
    x.iloc[60,x.columns.get_loc("signal")]=side
    return x

def test_stop_and_gap():
    x=bars()
    x.iloc[61,x.columns.get_loc("Open")]=95.
    x.iloc[61,x.columns.get_loc("Low")]=94.
    tr,stats=backtest_atr(x,hold_days=3,fee_bps=0,slippage_bps=0)
    assert stats["trades"]==1
    assert tr.iloc[0]["reason"]=="target"  # stop anchored to next open, no lookahead
    y=bars()
    y.iloc[62,y.columns.get_loc("Open")]=90.
    tr,_=backtest_atr(y,hold_days=3,fee_bps=0,slippage_bps=0)
    assert tr.iloc[0]["reason"]=="stop_gap"

def test_both_hit_conservative():
    x=bars()
    x.iloc[61,x.columns.get_loc("High")]=110.
    x.iloc[61,x.columns.get_loc("Low")]=90.
    tr,_=backtest_atr(x,stop_atr=1.,target_r=1.,hold_days=3)
    assert tr.iloc[0]["reason"]=="both_hit_stop_first"

def test_no_trades():
    x=bars()
    x["signal"]=0
    tr,stat=backtest_atr(x)
    assert tr.empty and stat["trades"]==0
