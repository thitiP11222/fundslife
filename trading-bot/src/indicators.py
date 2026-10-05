from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    value = 100 - (100 / (1 + rs))
    value = value.where(~((avg_gain > 0) & (avg_loss == 0)), 100.0)
    value = value.where(~((avg_gain == 0) & (avg_loss > 0)), 0.0)
    return value


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    ranges = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def add_indicators(
    df: pd.DataFrame,
    *,
    ema_fast: int,
    ema_slow: int,
    rsi_period: int,
    atr_period: int,
    breakout_period: int,
    volume_ma_period: int,
) -> pd.DataFrame:
    out = df.copy()
    out["EMA_FAST"] = ema(out["Close"], ema_fast)
    out["EMA_SLOW"] = ema(out["Close"], ema_slow)
    out["RSI"] = rsi(out["Close"], rsi_period)
    out["ATR"] = atr(out, atr_period)

    # Shift prevents the current candle from defining its own breakout.
    out["BREAKOUT_HIGH"] = (
        out["High"].rolling(breakout_period, min_periods=breakout_period).max().shift(1)
    )
    out["VOLUME_MA"] = (
        out["Volume"].rolling(volume_ma_period, min_periods=volume_ma_period).mean()
    )
    return out
