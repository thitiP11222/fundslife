from __future__ import annotations

import pandas as pd
import yfinance as yf


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download_ohlcv(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Download and normalize Yahoo Finance OHLCV data."""
    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise RuntimeError(
            f"No data returned for {symbol=} {interval=} {period=}."
        )

    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, axis=1, level=-1)
        else:
            df.columns = df.columns.get_level_values(0)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing OHLCV columns: {missing}")

    out = df[REQUIRED_COLUMNS].copy()
    out.index = pd.to_datetime(out.index)
    return out.dropna().sort_index()
