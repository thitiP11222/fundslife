import numpy as np
import pandas as pd
import pytest
from src.patterns import scan,position_size,indicators,_confirmed_pivots

def sample(n=85):
    x=np.linspace(100,110,n)
    d=pd.DataFrame({"Open":x,"High":x+1,"Low":x-1,"Close":x,"Volume":np.ones(n)*100},index=pd.date_range("2025-01-01",periods=n))
    return d

def test_indicators_no_future_breakout():
    d=sample()
    result=indicators(d)
    assert result["Prior20High"].iloc[-1]==pytest.approx(d.High.iloc[-21:-1].max())

def test_no_signal_flat_market():
    d=sample()
    _,signals=scan(d)
    assert signals==[]

def test_breakout_latest_bar():
    d=sample()
    d.loc[d.index[-1],"Close"]=130
    d.loc[d.index[-1],"High"]=131
    d.loc[d.index[-1],"Volume"]=250
    _,signals=scan(d)
    s=next(s for s in signals if s.pattern=="20-bar Breakout")
    assert s.direction=="LONG"
    assert s.stop<s.entry<s.target

def test_breakout_rejects_low_volume():
    d=sample()
    d.loc[d.index[-1],"Close"]=130
    d.loc[d.index[-1],"High"]=131
    _,signals=scan(d)
    assert not any(s.pattern=="20-bar Breakout" for s in signals)

def test_confirmed_pivot_does_not_include_unconfirmed_last_bar():
    values=[5,4,3,4,5,6,4,2]
    assert _confirmed_pivots(values,"low",left=1,right=1)==[2]

def test_position_size_caps_allocation():
    out=position_size(entry=100,stop=95,capital=1000,risk_percent=1,max_allocation_percent=60)
    assert out["notional"]<=600
    assert out["planned_risk"]<=10

def test_insufficient_history():
    with pytest.raises(ValueError):
        scan(sample(20))
