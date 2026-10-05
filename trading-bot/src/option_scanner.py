from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.indicators import atr, ema, rsi


@dataclass(frozen=True)
class OptionScanConfig:
    momentum_fast_days: int = 63
    momentum_slow_days: int = 126
    benchmark_ma_days: int = 200

    hourly_ema_fast: int = 20
    hourly_ema_slow: int = 50
    hourly_rsi_period: int = 14
    hourly_atr_period: int = 14
    hourly_breakout_period: int = 10
    hourly_volume_ma: int = 20

    min_rsi: float = 50.0
    max_rsi: float = 68.0
    min_volume_ratio: float = 1.0

    top_n: int = 3


def _latest_closed_hourly_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return data only through the most recent fully closed 1-hour candle.

    Yahoo can include the currently forming hourly candle. Using it distorts
    volume confirmation, so we exclude any bar whose one-hour window has not
    finished yet. For naive timestamps, we conservatively drop the last row.
    """
    x = df.copy().sort_index()

    if x.empty:
        return x

    last_ts = pd.Timestamp(x.index[-1])

    if last_ts.tzinfo is None:
        return x.iloc[:-1] if len(x) > 1 else x.iloc[0:0]

    now = pd.Timestamp.now(tz=last_ts.tz)
    if last_ts + pd.Timedelta(hours=1) > now:
        return x.iloc[:-1]

    return x


def _momentum_score(close: pd.Series, fast: int, slow: int) -> float | None:
    if len(close) <= slow:
        return None

    fast_return = float(close.iloc[-1] / close.iloc[-1 - fast] - 1)
    slow_return = float(close.iloc[-1] / close.iloc[-1 - slow] - 1)

    return 0.6 * fast_return + 0.4 * slow_return


def _market_risk_on(
    benchmark_daily: pd.DataFrame,
    ma_days: int,
) -> bool:
    close = benchmark_daily["Close"].astype(float)
    ma = close.rolling(ma_days, min_periods=ma_days).mean()

    if ma.empty or pd.isna(ma.iloc[-1]):
        return False

    return bool(float(close.iloc[-1]) > float(ma.iloc[-1]))


def _hourly_entry_snapshot(
    df: pd.DataFrame,
    cfg: OptionScanConfig,
) -> dict:
    x = _latest_closed_hourly_frame(df)

    if x.empty:
        return {
            "ready": False,
            "score": 0,
            "reason": "NO_CLOSED_1H_CANDLE",
        }

    x["EMA_FAST"] = ema(x["Close"], cfg.hourly_ema_fast)
    x["EMA_SLOW"] = ema(x["Close"], cfg.hourly_ema_slow)
    x["RSI"] = rsi(x["Close"], cfg.hourly_rsi_period)
    x["ATR"] = atr(x, cfg.hourly_atr_period)
    x["VOL_MA"] = (
        x["Volume"]
        .rolling(cfg.hourly_volume_ma, min_periods=cfg.hourly_volume_ma)
        .mean()
    )
    x["BREAKOUT"] = (
        x["High"]
        .rolling(
            cfg.hourly_breakout_period,
            min_periods=cfg.hourly_breakout_period,
        )
        .max()
        .shift(1)
    )

    row = x.iloc[-1]

    required = [
        row["EMA_FAST"],
        row["EMA_SLOW"],
        row["RSI"],
        row["ATR"],
        row["VOL_MA"],
        row["BREAKOUT"],
    ]
    if any(pd.isna(value) for value in required):
        return {
            "ready": False,
            "score": 0,
            "closed_bar_time": str(x.index[-1]),
            "reason": "NOT_ENOUGH_DATA",
        }

    close = float(row["Close"])
    ema_fast_value = float(row["EMA_FAST"])
    ema_slow_value = float(row["EMA_SLOW"])
    rsi_value = float(row["RSI"])
    atr_value = float(row["ATR"])
    breakout = float(row["BREAKOUT"])
    volume_ratio = (
        float(row["Volume"]) / float(row["VOL_MA"])
        if float(row["VOL_MA"]) > 0
        else 0.0
    )

    trend_ok = close > ema_fast_value > ema_slow_value
    momentum_ok = cfg.min_rsi <= rsi_value <= cfg.max_rsi
    volume_ok = volume_ratio >= cfg.min_volume_ratio
    breakout_ok = close > breakout

    score = (
        int(trend_ok) * 35
        + int(momentum_ok) * 25
        + int(volume_ok) * 15
        + int(breakout_ok) * 25
    )

    trigger_price = breakout
    invalidation_price = ema_fast_value - atr_value
    target_1 = trigger_price + atr_value
    target_2 = trigger_price + 2.0 * atr_value

    ready = trend_ok and momentum_ok and breakout_ok and volume_ok

    reasons = []
    if not trend_ok:
        reasons.append("1H_TREND_NOT_READY")
    if not momentum_ok:
        reasons.append("RSI_NOT_READY")
    if not volume_ok:
        reasons.append("VOLUME_NOT_CONFIRMED")
    if not breakout_ok:
        reasons.append("WAIT_BREAKOUT")

    return {
        "ready": ready,
        "score": score,
        "closed_bar_time": str(x.index[-1]),
        "close": close,
        "ema20": ema_fast_value,
        "ema50": ema_slow_value,
        "rsi": rsi_value,
        "atr": atr_value,
        "volume_ratio": volume_ratio,
        "trigger_price": trigger_price,
        "invalidation_price": invalidation_price,
        "target_1": target_1,
        "target_2": target_2,
        "trend_ok": trend_ok,
        "momentum_ok": momentum_ok,
        "volume_ok": volume_ok,
        "breakout_ok": breakout_ok,
        "reason": "READY" if ready else ",".join(reasons),
    }


def _grade_setup(row: pd.Series, top_n: int) -> str:
    top_rank = int(row["momentum_rank"]) <= top_n

    if (
        top_rank
        and bool(row["market_risk_on"])
        and bool(row["daily_trend_ready"])
        and bool(row["entry_ready"])
    ):
        return "A"

    if (
        top_rank
        and bool(row["market_risk_on"])
        and bool(row["daily_trend_ready"])
        and int(row["entry_score"]) >= 60
    ):
        return "B"

    if (
        int(row["momentum_rank"]) <= 5
        and bool(row["daily_trend_ready"])
        and int(row["entry_score"]) >= 35
    ):
        return "C"

    return "WAIT"


def scan_option_candidates(
    daily_data: dict[str, pd.DataFrame],
    hourly_data: dict[str, pd.DataFrame],
    benchmark_daily: pd.DataFrame,
    cfg: OptionScanConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or OptionScanConfig()
    risk_on = _market_risk_on(
        benchmark_daily,
        cfg.benchmark_ma_days,
    )

    rows = []

    for symbol, daily in daily_data.items():
        if symbol not in hourly_data:
            continue

        score = _momentum_score(
            daily["Close"].astype(float),
            cfg.momentum_fast_days,
            cfg.momentum_slow_days,
        )

        if score is None:
            continue

        daily_close = daily["Close"].astype(float)
        daily_ema50 = ema(daily_close, 50)
        daily_ema200 = ema(daily_close, 200)

        trend_ready = (
            pd.notna(daily_ema50.iloc[-1])
            and pd.notna(daily_ema200.iloc[-1])
            and float(daily_close.iloc[-1]) > float(daily_ema50.iloc[-1])
            and float(daily_ema50.iloc[-1]) > float(daily_ema200.iloc[-1])
        )

        hourly = _hourly_entry_snapshot(hourly_data[symbol], cfg)

        rows.append(
            {
                "symbol": symbol,
                "market_risk_on": risk_on,
                "daily_momentum_score": score,
                "daily_trend_ready": trend_ready,
                "entry_score": hourly.get("score", 0),
                "closed_bar_time": hourly.get("closed_bar_time"),
                "underlying_close": hourly.get("close"),
                "entry_trigger": hourly.get("trigger_price"),
                "invalidation": hourly.get("invalidation_price"),
                "target_1": hourly.get("target_1"),
                "target_2": hourly.get("target_2"),
                "rsi_1h": hourly.get("rsi"),
                "volume_ratio_1h": hourly.get("volume_ratio"),
                "trend_1h_ok": hourly.get("trend_ok", False),
                "rsi_1h_ok": hourly.get("momentum_ok", False),
                "volume_1h_ok": hourly.get("volume_ok", False),
                "breakout_1h_ok": hourly.get("breakout_ok", False),
                "entry_reason": hourly.get("reason"),
                "entry_ready": bool(hourly.get("ready", False)),
            }
        )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["daily_momentum_score", "entry_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    result["momentum_rank"] = range(1, len(result) + 1)
    result["setup_grade"] = result.apply(
        lambda row: _grade_setup(row, cfg.top_n),
        axis=1,
    )
    result["option_signal"] = "WAIT"
    result.loc[result["setup_grade"] == "A", "option_signal"] = "CALL_CANDIDATE"

    return result
