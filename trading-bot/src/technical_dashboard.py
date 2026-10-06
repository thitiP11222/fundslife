from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.indicators import atr, ema, rsi


@dataclass(frozen=True)
class TechnicalConfig:
    ema_fast: int = 20
    ema_slow: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    volume_ma_period: int = 20
    swing_window: int = 3
    level_lookback: int = 120
    level_merge_atr: float = 0.60
    breakout_lookback: int = 20
    target_r_multiple_1: float = 1.0
    target_r_multiple_2: float = 2.0


def add_dashboard_indicators(df: pd.DataFrame, cfg: TechnicalConfig) -> pd.DataFrame:
    x = df.copy().sort_index()
    x["EMA20"] = ema(x["Close"], cfg.ema_fast)
    x["EMA50"] = ema(x["Close"], cfg.ema_slow)
    x["RSI14"] = rsi(x["Close"], cfg.rsi_period)
    x["ATR14"] = atr(x, cfg.atr_period)
    x["VOL_MA20"] = x["Volume"].rolling(cfg.volume_ma_period, min_periods=cfg.volume_ma_period).mean()
    x["PREV_HIGH_20"] = x["High"].rolling(cfg.breakout_lookback, min_periods=cfg.breakout_lookback).max().shift(1)
    return x


def _cluster_levels(values: list[float], tolerance: float) -> list[float]:
    if not values:
        return []
    values = sorted(values)
    clusters: list[list[float]] = [[values[0]]]
    for value in values[1:]:
        center = float(np.mean(clusters[-1]))
        if abs(value - center) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [float(np.mean(cluster)) for cluster in clusters]


def detect_support_resistance(df: pd.DataFrame, cfg: TechnicalConfig) -> dict[str, list[float]]:
    x = add_dashboard_indicators(df, cfg).tail(cfg.level_lookback)
    if len(x) < cfg.swing_window * 2 + 1:
        return {"supports": [], "resistances": []}

    last = x.iloc[-1]
    last_close = float(last["Close"])
    last_atr = float(last["ATR14"]) if pd.notna(last["ATR14"]) else last_close * 0.02
    tolerance = max(last_atr * cfg.level_merge_atr, last_close * 0.002)

    swing_highs: list[float] = []
    swing_lows: list[float] = []
    w = cfg.swing_window

    for i in range(w, len(x) - w):
        high = float(x["High"].iloc[i])
        low = float(x["Low"].iloc[i])
        high_window = x["High"].iloc[i - w : i + w + 1]
        low_window = x["Low"].iloc[i - w : i + w + 1]
        if high >= float(high_window.max()):
            swing_highs.append(high)
        if low <= float(low_window.min()):
            swing_lows.append(low)

    levels = _cluster_levels(swing_highs + swing_lows, tolerance)
    supports = sorted([level for level in levels if level < last_close], reverse=True)[:3]
    resistances = sorted([level for level in levels if level > last_close])[:3]
    return {"supports": supports, "resistances": resistances}


def detect_patterns(df: pd.DataFrame, cfg: TechnicalConfig) -> list[dict[str, Any]]:
    x = add_dashboard_indicators(df, cfg)
    if len(x) < 30:
        return []

    patterns: list[dict[str, Any]] = []
    cur = x.iloc[-1]
    prev = x.iloc[-2]

    co, cc, ch, cl = map(float, [cur["Open"], cur["Close"], cur["High"], cur["Low"]])
    po, pc = map(float, [prev["Open"], prev["Close"]])

    body = abs(cc - co)
    candle_range = max(ch - cl, 1e-9)
    lower_wick = min(co, cc) - cl
    upper_wick = ch - max(co, cc)

    if pc < po and cc > co and co <= pc and cc >= po:
        patterns.append({"name": "Bullish Engulfing", "bias": "bullish", "strength": 2})

    if pc > po and cc < co and co >= pc and cc <= po:
        patterns.append({"name": "Bearish Engulfing", "bias": "bearish", "strength": 2})

    if lower_wick >= body * 2 and upper_wick <= max(body, candle_range * 0.15) and body / candle_range <= 0.45:
        patterns.append({"name": "Hammer / Pin Bar", "bias": "bullish", "strength": 1})

    if upper_wick >= body * 2 and lower_wick <= max(body, candle_range * 0.15) and body / candle_range <= 0.45:
        patterns.append({"name": "Shooting Star / Bearish Pin", "bias": "bearish", "strength": 1})

    prev_high = cur["PREV_HIGH_20"]
    if pd.notna(prev_high) and cc > float(prev_high):
        patterns.append({"name": "20-Bar Breakout", "bias": "bullish", "strength": 2})

    recent = x.tail(25)
    highs = recent["High"].nlargest(4).sort_index()
    if len(highs) >= 2:
        last_two = highs.iloc[-2:]
        avg_high = float(last_two.mean())
        if avg_high > 0 and abs(float(last_two.iloc[-1]) - float(last_two.iloc[-2])) / avg_high <= 0.015:
            patterns.append({"name": "Possible Double Top", "bias": "bearish", "strength": 1})

    lows = recent["Low"].nsmallest(4).sort_index()
    if len(lows) >= 2:
        last_two = lows.iloc[-2:]
        avg_low = float(last_two.mean())
        if avg_low > 0 and abs(float(last_two.iloc[-1]) - float(last_two.iloc[-2])) / avg_low <= 0.015:
            patterns.append({"name": "Possible Double Bottom", "bias": "bullish", "strength": 1})

    return patterns


def build_trade_plan(df: pd.DataFrame, cfg: TechnicalConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TechnicalConfig()
    x = add_dashboard_indicators(df, cfg)
    if x.empty:
        raise ValueError("No price data available.")

    row = x.iloc[-1]
    required = [row["EMA20"], row["EMA50"], row["RSI14"], row["ATR14"], row["VOL_MA20"]]
    if any(pd.isna(v) for v in required):
        raise ValueError("Not enough bars to calculate technical indicators.")

    close = float(row["Close"])
    ema20_value = float(row["EMA20"])
    ema50_value = float(row["EMA50"])
    rsi_value = float(row["RSI14"])
    atr_value = float(row["ATR14"])
    volume_ma = float(row["VOL_MA20"])
    volume_ratio = float(row["Volume"]) / volume_ma if volume_ma > 0 else 0.0

    levels = detect_support_resistance(x, cfg)
    supports, resistances = levels["supports"], levels["resistances"]
    patterns = detect_patterns(x, cfg)

    bullish_score = sum(p["strength"] for p in patterns if p["bias"] == "bullish")
    bearish_score = sum(p["strength"] for p in patterns if p["bias"] == "bearish")

    trend_bullish = close > ema20_value > ema50_value
    trend_bearish = close < ema20_value < ema50_value

    nearest_support = supports[0] if supports else close - atr_value
    nearest_resistance = resistances[0] if resistances else close + atr_value

    score = 0
    reasons: list[str] = []

    if trend_bullish:
        score += 35
        reasons.append("Trend bullish: Close > EMA20 > EMA50")
    if 50 <= rsi_value <= 68:
        score += 20
        reasons.append("RSI in constructive momentum zone")
    if volume_ratio >= 1.0:
        score += 15
        reasons.append("Volume confirms the move")
    if bullish_score > bearish_score:
        score += min(15, bullish_score * 5)
        reasons.append("Bullish price pattern confirmation")

    distance_to_resistance = (nearest_resistance - close) / close if close > 0 else 0.0
    if distance_to_resistance >= atr_value / close:
        score += 15
        reasons.append("Adequate room before nearest resistance")

    if trend_bearish:
        score -= 25
        reasons.append("Bearish EMA structure")
    if rsi_value >= 75:
        score -= 10
        reasons.append("RSI is extended")

    score = int(max(0, min(100, score)))
    setup = "BUY_SETUP" if score >= 75 else "WATCH" if score >= 55 else "AVOID"

    pullback_entry = max(nearest_support, ema20_value)
    breakout_entry = nearest_resistance + 0.10 * atr_value
    stop = min(nearest_support - 0.50 * atr_value, ema20_value - atr_value)

    risk = max(pullback_entry - stop, atr_value * 0.5)
    target_1 = pullback_entry + cfg.target_r_multiple_1 * risk
    target_2 = pullback_entry + cfg.target_r_multiple_2 * risk

    return {
        "setup": setup,
        "score": score,
        "close": close,
        "ema20": ema20_value,
        "ema50": ema50_value,
        "rsi": rsi_value,
        "atr": atr_value,
        "volume_ratio": volume_ratio,
        "supports": supports,
        "resistances": resistances,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "pullback_entry": pullback_entry,
        "breakout_entry": breakout_entry,
        "stop": stop,
        "target_1": target_1,
        "target_2": target_2,
        "patterns": patterns,
        "reasons": reasons,
    }
