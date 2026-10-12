import numpy as np
import pandas as pd
from scanner import indicators,signals,structure,pattern,backtest

def bars(n=200):
    idx=pd.date_range("2023-01-01",periods=n,freq="B")
    close=np.linspace(100,130,n)+2*np.sin(np.arange(n)/6)
    return pd.DataFrame({"Open":close,"High":close+1,"Low":close-1,"Close":close,"Volume":10000},index=idx)

def test_causal_future_bar():
    a=bars()
    before=indicators(a).iloc[-2][["ph","pl","support","resistance"]]
    b=a.copy();b.iloc[-1,b.columns.get_loc("High")]=999
    after=indicators(b).iloc[-2][["ph","pl","support","resistance"]]
    assert before.equals(after)

def test_outputs():
    a=bars(); s=signals(a,a)
    assert set(s.signal.unique()).issubset({0,1,-1})
    assert isinstance(structure(s),str) and isinstance(pattern(s),str)
    t,metrics=backtest(s)
    assert metrics["trades"]==len(t)

def test_no_trade():
    a=signals(bars(),bars()); a["signal"]=0
    t,m=backtest(a)
    assert t.empty and m["trades"]==0
