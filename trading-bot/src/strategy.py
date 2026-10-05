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


def _breakout_signal(out: pd.DataFrame, cfg: StrategyConfig) -> pd.Series:
    volume_ok = out["Volume"] >= out["VOLUME_MA"] * cfg.volume_ratio_min

    return (
        (out["EMA_FAST"] > out["EMA_SLOW"])
        & (out["Close"] > out["BREAKOUT_HIGH"])
        & out["RSI"].between(cfg.rsi_min, cfg.rsi_max, inclusive="both")
        & volume_ok
    )


def _pullback_reclaim_signal(out: pd.DataFrame, cfg: StrategyConfig) -> pd.Series:
    """
    Trend-pullback setup.

    Wait for price to pull back to/below EMA20, then reclaim EMA20 with a
    bullish candle while EMA20 remains above EMA50. This avoids chasing a
    fresh multi-bar high, which was weak in the NVDA 1h validation sample.
    """
    volume_ok = out["Volume"] >= out["VOLUME_MA"] * cfg.volume_ratio_min

    reclaimed_fast_ema = (
        (out["Close"].shift(1) <= out["EMA_FAST"].shift(1))
        & (out["Close"] > out["EMA_FAST"])
    )

    bullish_candle = out["Close"] > out["Open"]

    return (
        (out["EMA_FAST"] > out["EMA_SLOW"])
        & reclaimed_fast_ema
        & bullish_candle
        & out["RSI"].between(cfg.rsi_min, cfg.rsi_max, inclusive="both")
        & volume_ok
    )


def generate_signals(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = build_features(df, cfg)

    if cfg.strategy_name == "breakout":
        long_condition = _breakout_signal(out, cfg)
    elif cfg.strategy_name == "pullback_reclaim":
        long_condition = _pullback_reclaim_signal(out, cfg)
    else:
        raise ValueError(
            f"Unknown strategy_name={cfg.strategy_name!r}. "
            "Use 'breakout' or 'pullback_reclaim'."
        )

    if cfg.use_market_regime:
        required = {
            "BENCH_CLOSE",
            "BENCH_EMA_FAST",
            "BENCH_EMA_SLOW",
            "RS_VS_BENCH",
        }
        missing = required.difference(out.columns)
        if missing:
            raise ValueError(
                "Market regime filter enabled but context columns are missing: "
                + ", ".join(sorted(missing))
            )

        market_ok = (
            (out["BENCH_CLOSE"] > out["BENCH_EMA_SLOW"])
            & (out["BENCH_EMA_FAST"] > out["BENCH_EMA_SLOW"])
        )
        relative_strength_ok = out["RS_VS_BENCH"] > cfg.rs_min
        long_condition = long_condition & market_ok & relative_strength_ok

    out["SIGNAL"] = 0
    out.loc[long_condition, "SIGNAL"] = 1
    return out


def latest_signal(df: pd.DataFrame, cfg: StrategyConfig) -> dict:
    signals = generate_signals(df, cfg)
    row = signals.iloc[-1]

    return {
        "time": str(signals.index[-1]),
        "symbol": cfg.symbol,
        "strategy": cfg.strategy_name,
        "signal": "BUY" if int(row["SIGNAL"]) == 1 else "NO_TRADE",
        "close": float(row["Close"]),
        "ema_fast": float(row["EMA_FAST"]) if pd.notna(row["EMA_FAST"]) else None,
        "ema_slow": float(row["EMA_SLOW"]) if pd.notna(row["EMA_SLOW"]) else None,
        "rsi": float(row["RSI"]) if pd.notna(row["RSI"]) else None,
        "atr": float(row["ATR"]) if pd.notna(row["ATR"]) else None,
        "volume": float(row["Volume"]),
        "volume_ma": float(row["VOLUME_MA"]) if pd.notna(row["VOLUME_MA"]) else None,
    }
