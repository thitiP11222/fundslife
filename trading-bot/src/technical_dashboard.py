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



def classify_market_pattern(
    df: pd.DataFrame,
    cfg: TechnicalConfig,
) -> dict[str, Any]:
    """
    Classify the latest chart into one primary, trader-friendly setup.

    This is heuristic pattern recognition, not computer-vision prediction.
    Priority is given to structures that are commonly used for trend-entry
    timing: breakout, breakout-retest, ascending triangle, bull flag,
    EMA pullback, double bottom, then range/no-clear-pattern.
    """
    x = add_dashboard_indicators(df, cfg)
    if len(x) < 35:
        return {
            "name": "Insufficient data",
            "bias": "neutral",
            "confidence": 0,
            "trigger": None,
            "invalidation": None,
            "note": "Need more bars to classify the setup.",
        }

    row = x.iloc[-1]
    close = float(row["Close"])
    ema20_value = float(row["EMA20"])
    ema50_value = float(row["EMA50"])
    atr_value = float(row["ATR14"])
    volume_ma = float(row["VOL_MA20"])
    volume_ratio = float(row["Volume"]) / volume_ma if volume_ma > 0 else 0.0

    recent20 = x.tail(20)
    recent10 = x.tail(10)
    recent6 = x.tail(6)

    prev_20_high = float(x["High"].iloc[-21:-1].max())
    prev_20_low = float(x["Low"].iloc[-21:-1].min())

    trend_up = close > ema20_value > ema50_value
    trend_down = close < ema20_value < ema50_value

    # 1) Fresh breakout
    if close > prev_20_high and volume_ratio >= 1.0:
        return {
            "name": "Breakout",
            "bias": "bullish",
            "confidence": 90,
            "trigger": prev_20_high + 0.10 * atr_value,
            "invalidation": prev_20_high - 0.75 * atr_value,
            "note": "Price closed above the prior 20-bar high with volume confirmation.",
        }

    # 2) Breakout retest: a recent close was above resistance, current price
    # revisits that area but remains above it.
    prior_closes = recent6["Close"].iloc[:-1]
    had_breakout = bool((prior_closes > prev_20_high).any())
    retest_distance = abs(close - prev_20_high)

    if (
        trend_up
        and had_breakout
        and retest_distance <= 0.60 * atr_value
        and close >= prev_20_high - 0.20 * atr_value
    ):
        return {
            "name": "Breakout Retest",
            "bias": "bullish",
            "confidence": 88,
            "trigger": prev_20_high + 0.20 * atr_value,
            "invalidation": prev_20_high - 0.80 * atr_value,
            "note": "Recent breakout is being retested near former resistance.",
        }

    # 3) Ascending triangle: highs cluster while lows rise.
    highs = recent10["High"].astype(float)
    lows = recent10["Low"].astype(float)
    high_band = float(highs.max() - highs.min())
    low_slope = float(np.polyfit(np.arange(len(lows)), lows.values, 1)[0])

    if (
        trend_up
        and high_band <= 1.25 * atr_value
        and low_slope > 0
        and close >= float(highs.max()) - 1.0 * atr_value
    ):
        resistance = float(highs.max())
        return {
            "name": "Ascending Triangle",
            "bias": "bullish",
            "confidence": 82,
            "trigger": resistance + 0.10 * atr_value,
            "invalidation": float(lows.tail(5).min()) - 0.25 * atr_value,
            "note": "Highs are compressing near resistance while recent lows are rising.",
        }

    # 4) Bull flag: strong impulse followed by a shallow consolidation.
    impulse = x.iloc[-16:-6]
    flag = x.iloc[-6:]
    impulse_gain = (
        float(impulse["Close"].iloc[-1]) - float(impulse["Close"].iloc[0])
    )
    flag_pullback = (
        float(flag["Close"].iloc[-1]) - float(flag["Close"].iloc[0])
    )

    if (
        trend_up
        and impulse_gain >= 2.0 * atr_value
        and flag_pullback <= 0.5 * atr_value
        and float(flag["Low"].min()) >= ema20_value - 0.75 * atr_value
    ):
        flag_high = float(flag["High"].max())
        flag_low = float(flag["Low"].min())
        return {
            "name": "Bull Flag",
            "bias": "bullish",
            "confidence": 80,
            "trigger": flag_high + 0.10 * atr_value,
            "invalidation": flag_low - 0.25 * atr_value,
            "note": "Strong upward impulse followed by a shallow consolidation above the trend.",
        }

    # 5) EMA pullback in an uptrend.
    distance_ema20 = abs(close - ema20_value)

    if (
        trend_up
        and distance_ema20 <= 0.75 * atr_value
        and float(row["RSI14"]) >= 45
    ):
        return {
            "name": "EMA20 Pullback",
            "bias": "bullish",
            "confidence": 75,
            "trigger": max(close, ema20_value + 0.20 * atr_value),
            "invalidation": ema20_value - 1.0 * atr_value,
            "note": "Price is pulling back toward a rising EMA20 while the broader trend remains bullish.",
        }

    # 6) Double bottom/top confirmation from existing detector.
    patterns = detect_patterns(x, cfg)
    pattern_state = classify_market_pattern(x, cfg)
    names = {p["name"] for p in patterns}

    if "Possible Double Bottom" in names:
        return {
            "name": "Possible Double Bottom",
            "bias": "bullish",
            "confidence": 65,
            "trigger": float(recent20["High"].tail(8).max()),
            "invalidation": float(recent20["Low"].min()) - 0.25 * atr_value,
            "note": "Two recent lows formed near the same zone; wait for neckline confirmation.",
        }

    if "Possible Double Top" in names:
        return {
            "name": "Possible Double Top",
            "bias": "bearish",
            "confidence": 65,
            "trigger": float(recent20["Low"].tail(8).min()),
            "invalidation": float(recent20["High"].max()) + 0.25 * atr_value,
            "note": "Two recent highs formed near the same zone; bullish entries need caution.",
        }

    # 7) Range / no clear setup.
    range_width = float(recent20["High"].max() - recent20["Low"].min())
    if range_width <= 4.0 * atr_value:
        return {
            "name": "Range / Consolidation",
            "bias": "neutral",
            "confidence": 60,
            "trigger": float(recent20["High"].max()) + 0.10 * atr_value,
            "invalidation": float(recent20["Low"].min()) - 0.10 * atr_value,
            "note": "Price is consolidating; wait for a confirmed break or support reaction.",
        }

    if trend_down:
        return {
            "name": "Downtrend",
            "bias": "bearish",
            "confidence": 80,
            "trigger": None,
            "invalidation": ema20_value + atr_value,
            "note": "Close is below EMA20 and EMA50; avoid forcing bullish trades.",
        }

    return {
        "name": "No Clear Pattern",
        "bias": "neutral",
        "confidence": 40,
        "trigger": None,
        "invalidation": None,
        "note": "Trend exists but no high-quality entry pattern is confirmed.",
    }


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
        "pattern_state": pattern_state,
        "reasons": reasons,
    }
