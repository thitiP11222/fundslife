from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.indicators import atr, ema, rsi
from src.option_scanner import OptionScanConfig


def _date_key(value) -> pd.Timestamp:
    """
    Convert tz-aware or tz-naive timestamps to a timezone-free calendar date.

    Daily ranking and hourly signals only need a comparable trading-date key;
    keeping timezone offsets here creates invalid tz-aware/tz-naive comparisons.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


@dataclass(frozen=True)
class SignalBacktestConfig:
    max_holding_bars: int = 35
    entry_slippage_pct: float = 0.0002


def _prepare_daily(
    df: pd.DataFrame,
    cfg: OptionScanConfig,
) -> pd.DataFrame:
    x = df.copy().sort_index()
    close = x["Close"].astype(float)

    x["DAILY_EMA50"] = ema(close, 50)
    x["DAILY_EMA200"] = ema(close, 200)
    x["MOM_FAST"] = close / close.shift(cfg.momentum_fast_days) - 1
    x["MOM_SLOW"] = close / close.shift(cfg.momentum_slow_days) - 1
    x["MOM_SCORE"] = 0.6 * x["MOM_FAST"] + 0.4 * x["MOM_SLOW"]
    x["DAILY_TREND_READY"] = (
        (close > x["DAILY_EMA50"])
        & (x["DAILY_EMA50"] > x["DAILY_EMA200"])
    )
    return x


def _prepare_hourly(
    df: pd.DataFrame,
    cfg: OptionScanConfig,
) -> pd.DataFrame:
    x = df.copy().sort_index()
    x["EMA20"] = ema(x["Close"], cfg.hourly_ema_fast)
    x["EMA50"] = ema(x["Close"], cfg.hourly_ema_slow)
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
    return x


def _benchmark_regime(
    benchmark_daily: pd.DataFrame,
    cfg: OptionScanConfig,
) -> pd.Series:
    close = benchmark_daily["Close"].astype(float)
    ma = close.rolling(
        cfg.benchmark_ma_days,
        min_periods=cfg.benchmark_ma_days,
    ).mean()
    return (close > ma).rename("MARKET_RISK_ON")


def _daily_rank_table(
    daily_data: dict[str, pd.DataFrame],
    benchmark_daily: pd.DataFrame,
    cfg: OptionScanConfig,
) -> pd.DataFrame:
    prepared = {
        symbol: _prepare_daily(df, cfg)
        for symbol, df in daily_data.items()
    }

    regime = _benchmark_regime(benchmark_daily, cfg)

    all_dates = sorted(
        {
            _date_key(ts)
            for df in prepared.values()
            for ts in df.index
        }
    )

    rows: list[dict[str, Any]] = []

    for date in all_dates:
        scored: list[tuple[float, str, bool]] = []

        for symbol, df in prepared.items():
            date_keys = pd.Index([_date_key(ts) for ts in df.index])
            day_rows = df[date_keys == date]
            if day_rows.empty:
                continue

            row = day_rows.iloc[-1]
            score = row["MOM_SCORE"]

            if pd.isna(score):
                continue

            scored.append(
                (
                    float(score),
                    symbol,
                    bool(row["DAILY_TREND_READY"]),
                )
            )

        scored.sort(reverse=True)

        benchmark_date_keys = pd.Index(
            [_date_key(ts) for ts in regime.index]
        )
        benchmark_rows = regime[benchmark_date_keys == date]
        risk_on = bool(benchmark_rows.iloc[-1]) if not benchmark_rows.empty else False

        for rank, (score, symbol, trend_ready) in enumerate(scored, start=1):
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "symbol": symbol,
                    "momentum_rank": rank,
                    "daily_momentum_score": score,
                    "daily_trend_ready": trend_ready,
                    "market_risk_on": risk_on,
                }
            )

    return pd.DataFrame(rows)


def backtest_grade_a_signals(
    daily_data: dict[str, pd.DataFrame],
    hourly_data: dict[str, pd.DataFrame],
    benchmark_daily: pd.DataFrame,
    scan_cfg: OptionScanConfig | None = None,
    bt_cfg: SignalBacktestConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scan_cfg = scan_cfg or OptionScanConfig()
    bt_cfg = bt_cfg or SignalBacktestConfig()

    rank_table = _daily_rank_table(
        daily_data,
        benchmark_daily,
        scan_cfg,
    )

    hourly_prepared = {
        symbol: _prepare_hourly(df, scan_cfg)
        for symbol, df in hourly_data.items()
    }

    events: list[dict[str, Any]] = []

    for symbol, hourly in hourly_prepared.items():
        if symbol not in daily_data or hourly.empty:
            continue

        symbol_rank = rank_table[rank_table["symbol"] == symbol].copy()
        if symbol_rank.empty:
            continue

        symbol_rank = symbol_rank.set_index("date").sort_index()
        active_until = None

        for i in range(len(hourly) - 1):
            signal_time = hourly.index[i]

            if active_until is not None and signal_time <= active_until:
                continue

            row = hourly.iloc[i]

            required = [
                row["EMA20"],
                row["EMA50"],
                row["RSI"],
                row["ATR"],
                row["VOL_MA"],
                row["BREAKOUT"],
            ]
            if any(pd.isna(value) for value in required):
                continue

            signal_date = _date_key(signal_time)

            # Use the most recently completed daily bar strictly before the
            # hourly signal date to avoid same-day daily look-ahead.
            eligible = symbol_rank[symbol_rank.index < signal_date]
            if eligible.empty:
                continue

            daily_row = eligible.iloc[-1]

            close = float(row["Close"])
            ema20 = float(row["EMA20"])
            ema50 = float(row["EMA50"])
            rsi_value = float(row["RSI"])
            atr_value = float(row["ATR"])
            breakout = float(row["BREAKOUT"])
            vol_ma = float(row["VOL_MA"])
            volume_ratio = (
                float(row["Volume"]) / vol_ma if vol_ma > 0 else 0.0
            )

            trend_ok = close > ema20 > ema50
            rsi_ok = scan_cfg.min_rsi <= rsi_value <= scan_cfg.max_rsi
            volume_ok = volume_ratio >= scan_cfg.min_volume_ratio
            breakout_ok = close > breakout

            grade_a = (
                int(daily_row["momentum_rank"]) <= scan_cfg.top_n
                and bool(daily_row["market_risk_on"])
                and bool(daily_row["daily_trend_ready"])
                and trend_ok
                and rsi_ok
                and volume_ok
                and breakout_ok
            )

            if not grade_a:
                continue

            entry_row = hourly.iloc[i + 1]
            entry_time = hourly.index[i + 1]
            entry = float(entry_row["Open"]) * (
                1 + bt_cfg.entry_slippage_pct
            )

            invalidation = ema20 - atr_value

            # Targets must be anchored to the actual next-bar entry, not the
            # earlier breakout trigger. Otherwise a gap-up entry can leave very
            # little reward while still risking a full stop.
            risk_per_share = entry - invalidation
            if risk_per_share <= 0:
                continue

            target_1 = entry + risk_per_share
            target_2 = entry + 2.0 * risk_per_share

            # Reject excessive gap-up entries. If the next open is already more
            # than 0.5 ATR above the breakout trigger, the setup is considered
            # chased rather than a clean breakout entry.
            max_entry = breakout + 0.5 * atr_value
            if entry > max_entry:
                continue

            outcome = "TIME_EXIT"
            exit_time = None
            exit_price = None
            bars_held = 0
            t1_hit = False
            t2_hit = False

            end_i = min(
                i + 1 + bt_cfg.max_holding_bars,
                len(hourly) - 1,
            )

            for j in range(i + 1, end_i + 1):
                future = hourly.iloc[j]
                future_time = hourly.index[j]
                low = float(future["Low"])
                high = float(future["High"])
                open_price = float(future["Open"])
                bars_held = j - i

                # Conservative ordering: invalidation wins ties.
                if open_price <= invalidation:
                    outcome = "INVALIDATION"
                    exit_time = future_time
                    exit_price = open_price
                    break

                if low <= invalidation:
                    outcome = "INVALIDATION"
                    exit_time = future_time
                    exit_price = invalidation
                    break

                if high >= target_1:
                    t1_hit = True

                if high >= target_2:
                    t2_hit = True
                    outcome = "T2"
                    exit_time = future_time
                    exit_price = target_2
                    break

            if exit_time is None:
                final_row = hourly.iloc[end_i]
                exit_time = hourly.index[end_i]
                exit_price = float(final_row["Close"])
                if t1_hit:
                    outcome = "T1_ONLY"

            underlying_return_pct = (
                float(exit_price) / entry - 1
            ) * 100

            r_multiple = (
                (float(exit_price) - entry) / risk_per_share
                if risk_per_share > 0
                else None
            )

            events.append(
                {
                    "symbol": symbol,
                    "signal_time": signal_time,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "momentum_rank": int(daily_row["momentum_rank"]),
                    "daily_momentum_score": float(
                        daily_row["daily_momentum_score"]
                    ),
                    "entry": entry,
                    "trigger": breakout,
                    "invalidation": invalidation,
                    "target_1": target_1,
                    "target_2": target_2,
                    "outcome": outcome,
                    "t1_hit": t1_hit,
                    "t2_hit": t2_hit,
                    "bars_held": bars_held,
                    "underlying_return_pct": underlying_return_pct,
                    "r_multiple": r_multiple,
                }
            )

            active_until = exit_time

    events_df = pd.DataFrame(events)

    if events_df.empty:
        return events_df, {
            "signals": 0,
            "t1_hit_rate_pct": 0.0,
            "t2_hit_rate_pct": 0.0,
            "invalidation_rate_pct": 0.0,
            "avg_underlying_return_pct": 0.0,
            "avg_r_multiple": 0.0,
            "avg_bars_held": 0.0,
        }

    metrics = {
        "signals": int(len(events_df)),
        "t1_hit_rate_pct": float(events_df["t1_hit"].mean() * 100),
        "t2_hit_rate_pct": float(events_df["t2_hit"].mean() * 100),
        "invalidation_rate_pct": float(
            (events_df["outcome"] == "INVALIDATION").mean() * 100
        ),
        "avg_underlying_return_pct": float(
            events_df["underlying_return_pct"].mean()
        ),
        "avg_r_multiple": float(
            events_df["r_multiple"].dropna().mean()
        ),
        "avg_bars_held": float(events_df["bars_held"].mean()),
    }

    return events_df, metrics
