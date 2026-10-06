from src.options_trade_plan import build_long_call_plan


def test_long_call_waits_for_breakout_trigger():
    plan = build_long_call_plan(
        market_risk_on=True,
        scanner_action="BUY_CANDIDATE",
        technical_setup="BUY_SETUP",
        technical_score=85,
        pattern_state={
            "name": "Bull Flag",
            "bias": "bullish",
            "trigger": 105.0,
            "invalidation": 97.0,
        },
        current_price=100.0,
        entry_mode="Breakout",
        entry_price=105.0,
        stop_price=95.0,
        target_1=115.0,
        target_2=125.0,
        rsi=62.0,
        volume_ratio=1.2,
    )

    assert plan["status"] == "WAIT_TRIGGER"


def test_long_call_can_review_after_trigger():
    plan = build_long_call_plan(
        market_risk_on=True,
        scanner_action="BUY_CANDIDATE",
        technical_setup="BUY_SETUP",
        technical_score=90,
        pattern_state={
            "name": "Breakout",
            "bias": "bullish",
            "trigger": 105.0,
            "invalidation": 99.0,
        },
        current_price=106.0,
        entry_mode="Breakout",
        entry_price=105.0,
        stop_price=98.0,
        target_1=112.0,
        target_2=119.0,
        rsi=65.0,
        volume_ratio=1.3,
    )

    assert plan["status"] == "REVIEW_CALL"
    assert plan["preferred_dte"] == "30-60 DTE"
