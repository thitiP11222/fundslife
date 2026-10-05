import pandas as pd

from src.option_scanner import (
    OptionScanConfig,
    _grade_setup,
    _latest_closed_hourly_frame,
)


def test_grade_a_requires_top_rank_and_ready():
    row = pd.Series(
        {
            "momentum_rank": 1,
            "market_risk_on": True,
            "daily_trend_ready": True,
            "entry_ready": True,
            "entry_score": 100,
        }
    )

    assert _grade_setup(row, top_n=3) == "A"


def test_grade_b_for_top_rank_waiting_confirmation():
    row = pd.Series(
        {
            "momentum_rank": 2,
            "market_risk_on": True,
            "daily_trend_ready": True,
            "entry_ready": False,
            "entry_score": 75,
        }
    )

    assert _grade_setup(row, top_n=3) == "B"


def test_naive_hourly_data_drops_last_bar_conservatively():
    index = pd.date_range(
        "2026-01-01 09:30",
        periods=3,
        freq="1h",
    )
    df = pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [101, 102, 103],
            "Low": [99, 100, 101],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [1000, 1100, 1200],
        },
        index=index,
    )

    out = _latest_closed_hourly_frame(df)

    assert len(out) == 2
    assert out.index[-1] == index[-2]


def test_default_option_scan_config_top_n():
    cfg = OptionScanConfig()
    assert cfg.top_n == 3
