import pandas as pd

from src.indicators import atr, ema, rsi


def test_ema_returns_value_after_warmup():
    s = pd.Series(range(1, 30), dtype=float)
    assert ema(s, 5).iloc[-1] > 0


def test_rsi_rising_series_reaches_100():
    s = pd.Series(range(1, 40), dtype=float)
    assert rsi(s, 14).iloc[-1] == 100.0


def test_atr_is_positive():
    df = pd.DataFrame(
        {
            "High": [11, 12, 13, 14, 15, 16],
            "Low": [9, 10, 11, 12, 13, 14],
            "Close": [10, 11, 12, 13, 14, 15],
        },
        dtype=float,
    )
    assert atr(df, 3).iloc[-1] > 0
