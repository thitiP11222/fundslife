from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.indicators import atr, ema


@dataclass(frozen=True)
class UnifiedMomentumConfig:
    # Core signal already supported by this project's robustness tests.
    momentum_fast_days: int = 63
    momentum_slow_days: int = 126
    momentum_fast_weight: float = 0.60
    momentum_slow_weight: float = 0.40

    benchmark_ma_days: int = 200
    stock_ema_fast: int = 50
    stock_ema_slow: int = 200
    atr_period: int = 14

    top_n: int = 3

    # Liquidity safeguards for US-stock/fractional execution.
    # These are execution heuristics, not alpha parameters.
    min_price_usd: float = 5.0
    min_avg_dollar_volume_20d: float = 20_000_000.0

    # Diagnostics inspired by published momentum literature.
    # They are NOT used to alter the validated 3M/6M ranking.
    high_lookback_days: int = 252
    long_momentum_days: int = 252
    near_52w_high_ratio: float = 0.90


def _prepare_stock(
    df: pd.DataFrame,
    cfg: UnifiedMomentumConfig,
) -> pd.DataFrame:
    x = df.copy().sort_index()
    close = x["Close"].astype(float)

    x["EMA50"] = ema(close, cfg.stock_ema_fast)
    x["EMA200"] = ema(close, cfg.stock_ema_slow)
    x["ATR14"] = atr(x, cfg.atr_period)

    x["MOM_3M"] = close / close.shift(cfg.momentum_fast_days) - 1
    x["MOM_6M"] = close / close.shift(cfg.momentum_slow_days) - 1
    x["MOM_12M"] = close / close.shift(cfg.long_momentum_days) - 1

    x["MOM_SCORE"] = (
        cfg.momentum_fast_weight * x["MOM_3M"]
        + cfg.momentum_slow_weight * x["MOM_6M"]
    )

    x["HIGH_52W"] = (
        x["High"]
        .rolling(
            cfg.high_lookback_days,
            min_periods=cfg.high_lookback_days,
        )
        .max()
    )
    x["PRICE_TO_52W_HIGH"] = close / x["HIGH_52W"]

    x["AVG_DOLLAR_VOLUME_20D"] = (
        (close * x["Volume"].astype(float))
        .rolling(20, min_periods=20)
        .mean()
    )

    x["TREND_OK"] = (
        (close > x["EMA50"])
        & (x["EMA50"] > x["EMA200"])
    )

    return x


def _market_regime(
    benchmark_df: pd.DataFrame,
    cfg: UnifiedMomentumConfig,
) -> dict[str, Any]:
    x = benchmark_df.copy().sort_index()
    close = x["Close"].astype(float)
    ma200 = close.rolling(
        cfg.benchmark_ma_days,
        min_periods=cfg.benchmark_ma_days,
    ).mean()

    latest_close = float(close.iloc[-1])
    latest_ma = float(ma200.iloc[-1]) if pd.notna(ma200.iloc[-1]) else float("nan")

    return {
        "risk_on": bool(
            pd.notna(ma200.iloc[-1])
            and latest_close > latest_ma
        ),
        "benchmark_close": latest_close,
        "benchmark_ma200": latest_ma,
    }


def scan_unified_momentum(
    stock_data: dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    cfg: UnifiedMomentumConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = cfg or UnifiedMomentumConfig()
    regime = _market_regime(benchmark_df, cfg)

    rows: list[dict[str, Any]] = []

    for symbol, df in stock_data.items():
        x = _prepare_stock(df, cfg)
        row = x.iloc[-1]

        required = [
            row["EMA50"],
            row["EMA200"],
            row["ATR14"],
            row["MOM_3M"],
            row["MOM_6M"],
            row["MOM_SCORE"],
            row["AVG_DOLLAR_VOLUME_20D"],
        ]
        if any(pd.isna(v) for v in required):
            continue

        close = float(row["Close"])
        atr_value = float(row["ATR14"])
        avg_dollar_volume = float(row["AVG_DOLLAR_VOLUME_20D"])
        mom_score = float(row["MOM_SCORE"])

        liquidity_ok = (
            close >= cfg.min_price_usd
            and avg_dollar_volume >= cfg.min_avg_dollar_volume_20d
        )

        rows.append(
            {
                "symbol": symbol,
                "close": close,
                "momentum_3m_pct": float(row["MOM_3M"]) * 100,
                "momentum_6m_pct": float(row["MOM_6M"]) * 100,
                "momentum_12m_pct": (
                    float(row["MOM_12M"]) * 100
                    if pd.notna(row["MOM_12M"])
                    else None
                ),
                "momentum_score": mom_score,
                "trend_ok": bool(row["TREND_OK"]),
                "price_to_52w_high": (
                    float(row["PRICE_TO_52W_HIGH"])
                    if pd.notna(row["PRICE_TO_52W_HIGH"])
                    else None
                ),
                "near_52w_high": bool(
                    pd.notna(row["PRICE_TO_52W_HIGH"])
                    and float(row["PRICE_TO_52W_HIGH"])
                    >= cfg.near_52w_high_ratio
                ),
                "atr_pct": (
                    atr_value / close * 100
                    if close > 0
                    else None
                ),
                "avg_dollar_volume_20d": avg_dollar_volume,
                "liquidity_ok": liquidity_ok,
            }
        )

    if not rows:
        return pd.DataFrame(), regime

    result = pd.DataFrame(rows).sort_values(
        "momentum_score",
        ascending=False,
    ).reset_index(drop=True)

    result["momentum_rank"] = range(1, len(result) + 1)

    core_ok = (
        regime["risk_on"]
        & result["trend_ok"]
        & result["liquidity_ok"]
        & (result["momentum_score"] > 0)
    )

    result["fractional_action"] = "NO_TRADE"
    result.loc[
        core_ok & (result["momentum_rank"] <= cfg.top_n),
        "fractional_action",
    ] = "BUY_CANDIDATE"
    result.loc[
        core_ok & (result["momentum_rank"] > cfg.top_n),
        "fractional_action",
    ] = "WATCH"

    # Options are deliberately stricter and remain manual.
    # 52-week-high proximity is only a confirmation flag here.
    result["option_action"] = "NO_OPTION"
    result.loc[
        core_ok
        & (result["momentum_rank"] == 1)
        & result["near_52w_high"],
        "option_action",
    ] = "REVIEW_CALL_CHAIN"

    # DR execution should follow the same underlying signal. The user must map
    # the foreign underlying to an actually listed, sufficiently liquid DR/DRx.
    result["dr_action"] = "NO_DR"
    result.loc[
        core_ok & (result["momentum_rank"] <= cfg.top_n),
        "dr_action",
    ] = "CHECK_DR_OR_DRX"

    return result, regime
