import pandas as pd

from src.options_advisor import (
    OptionAdvisorConfig,
    option_readiness,
    screen_long_calls,
    summarize_option_fit,
)


def test_breakout_waits_for_underlying_confirmation():
    result = option_readiness(
        market_risk_on=True,
        scanner_action="BUY_CANDIDATE",
        technical_setup="BUY_SETUP",
        current_price=100.0,
        entry_mode="Breakout",
        entry_price=105.0,
        atr_value=3.0,
    )

    assert result["ready"] is False
    assert result["status"] == "WAIT_BREAKOUT"


def test_breakout_can_review_chain_after_confirmation():
    result = option_readiness(
        market_risk_on=True,
        scanner_action="BUY_CANDIDATE",
        technical_setup="BUY_SETUP",
        current_price=106.0,
        entry_mode="Breakout",
        entry_price=105.0,
        atr_value=3.0,
    )

    assert result["ready"] is True
    assert result["status"] == "REVIEW_CALL_CHAIN"


def test_screen_rejects_contract_that_exceeds_small_budget():
    calls = pd.DataFrame(
        {
            "contractSymbol": ["TEST"],
            "strike": [100.0],
            "bid": [4.90],
            "ask": [5.00],
            "lastPrice": [4.95],
            "volume": [500],
            "openInterest": [2000],
            "impliedVolatility": [0.40],
        }
    )

    result = screen_long_calls(
        calls,
        spot_price=101.0,
        account_equity=30.0,
        option_budget_pct=10.0,
        cfg=OptionAdvisorConfig(),
    )

    assert not result.empty
    assert bool(result.iloc[0]["liquidity_ok"]) is True
    assert bool(result.iloc[0]["within_option_budget"]) is False

    summary = summarize_option_fit(result)
    assert summary["status"] == "OVER_BUDGET"
