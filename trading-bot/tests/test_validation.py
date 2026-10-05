import pandas as pd

from config import StrategyConfig
from src.validation import (
    buy_and_hold_return_pct,
    parameter_sensitivity,
    rolling_out_of_sample,
)


def _sample_df(n: int = 1000) -> pd.DataFrame:
    close = pd.Series([100 + i * 0.05 for i in range(n)], dtype=float)
    return pd.DataFrame(
        {
            "Open": close - 0.02,
            "High": close + 0.20,
            "Low": close - 0.20,
            "Close": close,
            "Volume": [1000 + (i % 50) * 10 for i in range(n)],
        },
        index=pd.date_range("2026-01-01", periods=n, freq="15min"),
    )


def test_buy_and_hold_return_positive_for_rising_market():
    df = _sample_df(20)
    assert buy_and_hold_return_pct(df) > 0


def test_rolling_out_of_sample_returns_folds():
    df = _sample_df(1000)
    cfg = StrategyConfig()
    folds = rolling_out_of_sample(
        df,
        cfg,
        train_bars=600,
        test_bars=200,
    )

    assert len(folds) == 2
    assert {
        "fold",
        "strategy_return_pct",
        "buy_hold_return_pct",
        "trades",
    }.issubset(folds.columns)


def test_parameter_sensitivity_returns_expected_grid_size():
    df = _sample_df(500)
    cfg = StrategyConfig()

    result = parameter_sensitivity(
        df,
        cfg,
        breakout_periods=(15, 20),
        atr_stop_multipliers=(1.0, 1.5),
        risk_reward_ratios=(1.5, 2.0),
    )

    assert len(result) == 8
