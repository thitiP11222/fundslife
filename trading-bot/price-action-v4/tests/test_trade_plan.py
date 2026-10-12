import numpy as np
import pandas as pd
from scanner import signals
from trade_plan import describe

def test_plan_wait_contains_no_entry():
    dates=pd.date_range("2024-01-01",periods=120,freq="B")
    vals=100+np.arange(120)*0.03
    df=pd.DataFrame({"Open":vals,"High":vals+1,"Low":vals-1,"Close":vals,"Volume":10000},index=dates)
    x=signals(df,df)
    x["action"]="WAIT"
    out=describe(x)
    assert out["entry_trigger"] is None
    assert out["signal"]=="WAIT"
