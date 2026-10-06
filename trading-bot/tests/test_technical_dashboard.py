import pandas as pd

from src.technical_dashboard import (
    TechnicalConfig,
    add_dashboard_indicators,
    build_trade_plan,
    detect_support_resistance,
)


def _sample_ohlcv(n: int = 180) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    close = [100 + i * 0.2 + (i % 7) * 0.1 for i in range(n)]
    return pd.DataFrame(
        {
            "Open": [x - 0.2 for x in close],
            "High": [x + 1.0 for x in close],
            "Low": [x - 1.0 for x in close],
            "Close": close,
            "Volume": [1_000_000 + i * 1000 for i in range(n)],
        },
        index=idx,
    )


def test_dashboard_indicators_are_created():
    cfg = TechnicalConfig()
    out = add_dashboard_indicators(_sample_ohlcv(), cfg)

    for column in ["EMA20", "EMA50", "RSI14", "ATR14", "VOL_MA20"]:
        assert column in out.columns


def test_support_resistance_returns_lists():
    levels = detect_support_resistance(_sample_ohlcv(), TechnicalConfig())

    assert isinstance(levels["supports"], list)
    assert isinstance(levels["resistances"], list)


def test_trade_plan_contains_risk_levels():
    plan = build_trade_plan(_sample_ohlcv(), TechnicalConfig())

    assert plan["setup"] in {"BUY_SETUP", "WATCH", "AVOID"}
    assert plan["stop"] < plan["target_1"] < plan["target_2"]
    assert 0 <= plan["score"] <= 100
