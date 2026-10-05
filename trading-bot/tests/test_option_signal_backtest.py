import pandas as pd

from src.option_scanner import OptionScanConfig
from src.option_signal_backtest import (
    SignalBacktestConfig,
    _prepare_daily,
    _prepare_hourly,
)


def _sample_daily(n: int = 260) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=n, freq="B")
    close = [100 + i * 0.2 for i in range(n)]
    return pd.DataFrame(
        {
            "Open": [x - 0.1 for x in close],
            "High": [x + 0.5 for x in close],
            "Low": [x - 0.5 for x in close],
            "Close": close,
            "Volume": [1_000_000 for _ in close],
        },
        index=index,
    )


def _sample_hourly(n: int = 120) -> pd.DataFrame:
    index = pd.date_range("2026-01-01 09:30", periods=n, freq="1h")
    close = [100 + i * 0.1 for i in range(n)]
    return pd.DataFrame(
        {
            "Open": [x - 0.05 for x in close],
            "High": [x + 0.20 for x in close],
            "Low": [x - 0.20 for x in close],
            "Close": close,
            "Volume": [1000 + i for i in range(n)],
        },
        index=index,
    )


def test_prepare_daily_builds_momentum_without_future_shift():
    cfg = OptionScanConfig()
    out = _prepare_daily(_sample_daily(), cfg)

    assert out["MOM_SCORE"].iloc[: cfg.momentum_slow_days].isna().all()
    assert pd.notna(out["MOM_SCORE"].iloc[-1])


def test_prepare_hourly_breakout_is_shifted():
    cfg = OptionScanConfig(hourly_breakout_period=10)
    out = _prepare_hourly(_sample_hourly(), cfg)

    i = 20
    expected = _sample_hourly()["High"].iloc[i - 10 : i].max()
    assert out["BREAKOUT"].iloc[i] == expected


def test_signal_backtest_config_defaults_are_conservative():
    cfg = SignalBacktestConfig()
    assert cfg.max_holding_bars == 35
    assert cfg.entry_slippage_pct > 0
