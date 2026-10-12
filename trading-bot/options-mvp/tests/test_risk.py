import pytest
from options_core.risk import option_risk_check

def test_small_capital_rejects_contract():
    x=option_risk_check(equity_thb=1000,risk_pct=1,fx_thb_per_usd=35,
                        ask_usd=0.20,bid_usd=0.18)
    assert x["approved"] is False and x["contracts"]==0 and x["risk_budget_thb"]==10

def test_affordable_contract_passes():
    x=option_risk_check(equity_thb=100000,risk_pct=1,fx_thb_per_usd=35,
                        ask_usd=0.20,bid_usd=0.18)
    assert x["approved"] and x["contracts"]>=1

def test_bad_quote_rejected():
    with pytest.raises(ValueError):
        option_risk_check(equity_thb=1000,risk_pct=1,fx_thb_per_usd=35,
                          ask_usd=0.1,bid_usd=0.2)
