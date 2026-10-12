import numpy as np
import pandas as pd
from options_core.engine import features,signals,backtest_underlying

def fixture():
    idx=pd.date_range("2024-01-01",periods=110,freq="B")
    close=pd.Series(np.linspace(100,140,len(idx)),index=idx)
    return pd.DataFrame({"Open":close.shift(1).fillna(close.iloc[0]),"High":close+1,
                         "Low":close-1,"Close":close,"Volume":10000},index=idx)

def test_features_and_backtest():
    d=fixture()
    f=features(d)
    assert len(f)==110 and "rsi" in f and "breakout20" in f
    x,t,stats=backtest_underlying(d)
    assert set(x["signal"].dropna().unique()).issubset({-1,0,1})
    assert stats["trades"]==len(t)

def test_signal_does_not_use_next_bar():
    d=fixture()
    s1=signals(features(d))["signal"].iloc[-2]
    d.loc[d.index[-1],"Close"]=500.
    s2=signals(features(d))["signal"].iloc[-2]
    assert s1==s2

def test_invalid_history():
    import pytest
    with pytest.raises(ValueError):
        features(fixture().head(10))
