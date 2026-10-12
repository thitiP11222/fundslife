import numpy as np
import pandas as pd
from src.strategy_v2 import strategy_v2,compare

def bars(n=320):
    ix=pd.date_range("2023-01-01",periods=n,freq="B")
    close=pd.Series(100+np.arange(n)*.15+2*np.sin(np.arange(n)/5),index=ix)
    return pd.DataFrame({"Open":close.shift().fillna(close.iloc[0]),"High":close+1,
      "Low":close-1,"Close":close,"Volume":10000},index=ix)

def test_causal_signal_and_comparison():
    a=bars();b=bars()
    initial=strategy_v2(a,b)["signal"].iloc[-2]
    a.loc[a.index[-1],"Close"]=999
    assert strategy_v2(a,b)["signal"].iloc[-2]==initial
    s,table=compare(bars(),bars())
    assert len(table)==4
    assert set(s.signal.unique()).issubset({-1,0,1})
    assert set(table.segment)=={"in_sample","out_of_sample"}
