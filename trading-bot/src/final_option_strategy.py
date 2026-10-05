from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.indicators import atr, ema


@dataclass(frozen=True)
class FinalOptionConfig:
    momentum_fast_days: int = 63
    momentum_slow_days: int = 126
    benchmark_ma_days: int = 200

    stock_ema_fast: int = 50
    stock_ema_slow: int = 200

    top_n: int = 3
    hold_days: int = 20

    # Option contract guidance only. Contract selection remains manual.
    min_dte: int = 30
    max_dte: int = 60

    # Avoid chasing an unusually extended daily move.
    atr_period: int = 14
    max_extension_atr: float = 1.0


def _prepare_stock(
    df: pd.DataFrame,
    cfg: FinalOptionConfig,
) -> pd.DataFrame:
    x = df.copy().sort_index()
    close = x["Close"].astype(float)

    x["EMA50"] = ema(close, cfg.stock_ema_fast)
    x["EMA200"] = ema(close, cfg.stock_ema_slow)
    x["ATR"] = atr(x, cfg.atr_period)

    x["MOM_3M"] = close / close.shift(cfg.momentum_fast_days) - 1
    x["MOM_6M"] = close / close.shift(cfg.momentum_slow_days) - 1
    x["MOM_SCORE"] = 0.6 * x["MOM_3M"] + 0.4 * x["MOM_6M"]

    x["TREND_OK"] = (
        (close > x["EMA50"])
        & (x["EMA50"] > x["EMA200"])
    )

    return x


def _market_risk_on(
    benchmark_df: pd.DataFrame,
    cfg: FinalOptionConfig,
) -> tuple[bool, float, float]:
    x = benchmark_df.copy().sort_index()
    close = x["Close"].astype(float)
    ma = close.rolling(
        cfg.benchmark_ma_days,
        min_periods=cfg.benchmark_ma_days,
    ).mean()

    latest_close = float(close.iloc[-1])
    latest_ma = float(ma.iloc[-1]) if pd.notna(ma.iloc[-1]) else float("nan")

    risk_on = pd.notna(ma.iloc[-1]) and latest_close > latest_ma

    return bool(risk_on), latest_close, latest_ma


def scan_final_option_candidates(
    stock_data: dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    cfg: FinalOptionConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = cfg or FinalOptionConfig()

    risk_on, qqq_close, qqq_ma200 = _market_risk_on(
        benchmark_df,
        cfg,
    )

    rows: list[dict[str, Any]] = []

    for symbol, df in stock_data.items():
        x = _prepare_stock(df, cfg)
        row = x.iloc[-1]

        required = [
            row["EMA50"],
            row["EMA200"],
            row["ATR"],
            row["MOM_3M"],
            row["MOM_6M"],
            row["MOM_SCORE"],
        ]
        if any(pd.isna(value) for value in required):
            continue

        close = float(row["Close"])
        atr_value = float(row["ATR"])
        ema50 = float(row["EMA50"])
        ema200 = float(row["EMA200"])
        score = float(row["MOM_SCORE"])
        trend_ok = bool(row["TREND_OK"])

        # Extension is measured versus EMA50. This is a safety diagnostic, not
        # an entry edge. It only prevents the scanner from labeling a highly
        # stretched stock as immediately actionable.
        extension_atr = (
            (close - ema50) / atr_value
            if atr_value > 0
            else float("inf")
        )
        extension_ok = extension_atr <= cfg.max_extension_atr

        rows.append(
            {
                "symbol": symbol,
                "close": close,
                "ema50": ema50,
                "ema200": ema200,
                "atr14": atr_value,
                "momentum_3m_pct": float(row["MOM_3M"]) * 100,
                "momentum_6m_pct": float(row["MOM_6M"]) * 100,
                "momentum_score": score,
                "trend_ok": trend_ok,
                "extension_atr": extension_atr,
                "extension_ok": extension_ok,
            }
        )

    if not rows:
        return pd.DataFrame(), {
            "market_risk_on": risk_on,
            "qqq_close": qqq_close,
            "qqq_ma200": qqq_ma200,
        }

    result = pd.DataFrame(rows)
    result = result.sort_values(
        "momentum_score",
        ascending=False,
    ).reset_index(drop=True)

    result["momentum_rank"] = range(1, len(result) + 1)
    result["selected"] = (
        (result["momentum_rank"] <= cfg.top_n)
        & result["trend_ok"]
        & (result["momentum_score"] > 0)
    )

    result["action"] = "NO_TRADE"

    review_mask = (
        risk_on
        & result["selected"]
        & result["extension_ok"]
    )
    result.loc[review_mask, "action"] = "REVIEW_CALL"

    stretched_mask = (
        risk_on
        & result["selected"]
        & ~result["extension_ok"]
    )
    result.loc[stretched_mask, "action"] = "WAIT_NOT_CHASE"

    meta = {
        "market_risk_on": risk_on,
        "qqq_close": qqq_close,
        "qqq_ma200": qqq_ma200,
        "hold_days": cfg.hold_days,
        "min_dte": cfg.min_dte,
        "max_dte": cfg.max_dte,
        "top_n": cfg.top_n,
    }

    return result, meta
