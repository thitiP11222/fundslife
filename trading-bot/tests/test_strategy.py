import pandas as pd

from config import StrategyConfig
from src.strategy import generate_signals


def test_signal_values_are_binary():
    n = 120
    close = pd.Series([100 + i * 0.2 for i in range(n)], dtype=float)
    df = pd.DataFrame(
        {
            "Open": close - 0.1,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": [1000 + i for i in range(n)],
        },
        index=pd.date_range("2026-01-01", periods=n, freq="15min"),
    )

    out = generate_signals(df, StrategyConfig())
    assert set(out["SIGNAL"].unique()).issubset({0, 1})
