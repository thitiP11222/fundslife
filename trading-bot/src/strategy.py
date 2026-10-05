from __future__ import annotations

import pandas as pd

from config import StrategyConfig
from src.indicators import add_indicators


def build_features(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    return add_indicators(
        df,
        ema_fast=cfg.ema_fast,
        ema_slow=cfg.ema_slow,
        rsi_period=cfg.rsi_period,
        atr_period=cfg.atr_period,
        breakout_period=cfg.breakout_period,
        volume_ma_period=cfg.volume_ma_period,
    )


def generate_signals(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = build_features(df, cfg)

    long_condition = (
        (out["EMA_FAST"] > out["EMA_SLOW"])
        & (out["Close"] > out["BREAKOUT_HIGH"])
        & out["RSI"].between(cfg.rsi_min, cfg.rsi_max, inclusive="both")
        & (out["Volume"] >= out["VOLUME_MA"])
    )

    out["SIGNAL"] = 0
    out.loc[long_condition, "SIGNAL"] = 1
    return out


def latest_signal(df: pd.DataFrame, cfg: StrategyConfig) -> dict:
    signals = generate_signals(df, cfg)
    row = signals.iloc[-1]

    return {
        "time": str(signals.index[-1]),
        "symbol": cfg.symbol,
        "signal": "BUY" if int(row["SIGNAL"]) == 1 else "NO_TRADE",
        "close": float(row["Close"]),
        "ema_fast": float(row["EMA_FAST"]) if pd.notna(row["EMA_FAST"]) else None,
        "ema_slow": float(row["EMA_SLOW"]) if pd.notna(row["EMA_SLOW"]) else None,
        "rsi": float(row["RSI"]) if pd.notna(row["RSI"]) else None,
        "atr": float(row["ATR"]) if pd.notna(row["ATR"]) else None,
        "volume": float(row["Volume"]),
        "volume_ma": float(row["VOLUME_MA"]) if pd.notna(row["VOLUME_MA"]) else None,
    }
