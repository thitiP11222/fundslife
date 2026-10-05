from __future__ import annotations

from typing import Any

import pandas as pd

from src.indicators import ema


def _prepare_daily(
    df: pd.DataFrame,
    *,
    momentum_fast: int = 63,
    momentum_slow: int = 126,
) -> pd.DataFrame:
    x = df.copy().sort_index()
    close = x["Close"].astype(float)

    x["EMA50"] = ema(close, 50)
    x["EMA200"] = ema(close, 200)
    x["MOM_FAST"] = close / close.shift(momentum_fast) - 1
    x["MOM_SLOW"] = close / close.shift(momentum_slow) - 1
    x["MOM_SCORE"] = 0.6 * x["MOM_FAST"] + 0.4 * x["MOM_SLOW"]
    x["TREND_OK"] = (
        (close > x["EMA50"])
        & (x["EMA50"] > x["EMA200"])
    )

    return x


def momentum_forward_event_study(
    stock_data: dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    *,
    top_n: int = 3,
    benchmark_ma_days: int = 200,
    forward_days: tuple[int, ...] = (5, 10, 20),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cross-sectional momentum event study.

    On each trading day:
    - QQQ must be above its 200-day moving average.
    - Rank stocks by blended 3M/6M momentum.
    - Keep only symbols in an uptrend (Close > EMA50 > EMA200).
    - Measure forward close-to-close returns over 5/10/20 trading days.

    This is an event study, not a portfolio backtest. It is intended to answer
    whether momentum ranking has predictive value over horizons relevant to
    call-option holding periods.
    """

    prepared = {
        symbol: _prepare_daily(df)
        for symbol, df in stock_data.items()
    }

    benchmark = benchmark_df.copy().sort_index()
    bench_close = benchmark["Close"].astype(float)
    benchmark["RISK_ON"] = (
        bench_close
        > bench_close.rolling(
            benchmark_ma_days,
            min_periods=benchmark_ma_days,
        ).mean()
    )

    common_dates = benchmark.index
    for df in prepared.values():
        common_dates = common_dates.intersection(df.index)
    common_dates = common_dates.sort_values()

    events: list[dict[str, Any]] = []

    for i, ts in enumerate(common_dates):
        if not bool(benchmark.loc[ts, "RISK_ON"]):
            continue

        ranked: list[tuple[float, str]] = []

        for symbol, df in prepared.items():
            row = df.loc[ts]
            score = row["MOM_SCORE"]

            if pd.isna(score) or not bool(row["TREND_OK"]):
                continue

            ranked.append((float(score), symbol))

        ranked.sort(reverse=True)

        for rank, (score, symbol) in enumerate(
            ranked[:top_n],
            start=1,
        ):
            df = prepared[symbol]
            loc = df.index.get_loc(ts)

            if not isinstance(loc, int):
                continue

            entry_close = float(df.iloc[loc]["Close"])

            record: dict[str, Any] = {
                "date": ts,
                "symbol": symbol,
                "rank": rank,
                "momentum_score": score,
                "entry_close": entry_close,
            }

            complete = True

            for days in forward_days:
                future_loc = loc + days
                if future_loc >= len(df):
                    complete = False
                    break

                future_close = float(df.iloc[future_loc]["Close"])
                ret = (future_close / entry_close - 1) * 100

                record[f"return_{days}d_pct"] = ret
                record[f"win_{days}d"] = ret > 0

            if complete:
                events.append(record)

    events_df = pd.DataFrame(events)

    if events_df.empty:
        return events_df, pd.DataFrame()

    summary_rows = []

    for rank in sorted(events_df["rank"].unique()):
        subset = events_df[events_df["rank"] == rank]

        row: dict[str, Any] = {
            "rank": int(rank),
            "events": int(len(subset)),
        }

        for days in forward_days:
            values = subset[f"return_{days}d_pct"]
            row[f"avg_return_{days}d_pct"] = float(values.mean())
            row[f"median_return_{days}d_pct"] = float(values.median())
            row[f"win_rate_{days}d_pct"] = float(
                (values > 0).mean() * 100
            )

        summary_rows.append(row)

    # Aggregate all Top-N events as well.
    all_row: dict[str, Any] = {
        "rank": 0,
        "events": int(len(events_df)),
    }

    for days in forward_days:
        values = events_df[f"return_{days}d_pct"]
        all_row[f"avg_return_{days}d_pct"] = float(values.mean())
        all_row[f"median_return_{days}d_pct"] = float(values.median())
        all_row[f"win_rate_{days}d_pct"] = float(
            (values > 0).mean() * 100
        )

    summary_rows.append(all_row)

    return events_df, pd.DataFrame(summary_rows)
