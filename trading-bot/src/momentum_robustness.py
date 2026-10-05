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


def non_overlapping_momentum_study(
    stock_data: dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    *,
    top_n: int = 3,
    benchmark_ma_days: int = 200,
    hold_days: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Robustness study for momentum selection.

    Uses non-overlapping observation windows per symbol/rank and compares
    stock forward return against QQQ over the same holding horizon.
    """

    prepared = {
        symbol: _prepare_daily(df)
        for symbol, df in stock_data.items()
    }

    benchmark = benchmark_df.copy().sort_index()
    bench_close = benchmark["Close"].astype(float)
    benchmark["BENCH_MA"] = (
        bench_close
        .rolling(benchmark_ma_days, min_periods=benchmark_ma_days)
        .mean()
    )
    benchmark["RISK_ON"] = bench_close > benchmark["BENCH_MA"]

    common_dates = benchmark.index
    for df in prepared.values():
        common_dates = common_dates.intersection(df.index)
    common_dates = common_dates.sort_values()

    rows: list[dict[str, Any]] = []
    next_allowed: dict[tuple[str, int], pd.Timestamp] = {}

    for ts in common_dates:
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

        bench_loc = benchmark.index.get_loc(ts)
        if not isinstance(bench_loc, int):
            continue
        if bench_loc + hold_days >= len(benchmark):
            continue

        bench_entry = float(benchmark.iloc[bench_loc]["Close"])
        bench_exit = float(benchmark.iloc[bench_loc + hold_days]["Close"])
        bench_ret = (bench_exit / bench_entry - 1) * 100

        for rank, (score, symbol) in enumerate(
            ranked[:top_n],
            start=1,
        ):
            key = (symbol, rank)

            if key in next_allowed and ts < next_allowed[key]:
                continue

            df = prepared[symbol]
            loc = df.index.get_loc(ts)

            if not isinstance(loc, int):
                continue
            if loc + hold_days >= len(df):
                continue

            entry = float(df.iloc[loc]["Close"])
            exit_ = float(df.iloc[loc + hold_days]["Close"])
            stock_ret = (exit_ / entry - 1) * 100

            rows.append(
                {
                    "date": ts,
                    "symbol": symbol,
                    "rank": rank,
                    "momentum_score": score,
                    "stock_return_pct": stock_ret,
                    "qqq_return_pct": bench_ret,
                    "excess_return_pct": stock_ret - bench_ret,
                    "win": stock_ret > 0,
                    "beat_qqq": stock_ret > bench_ret,
                }
            )

            next_allowed[key] = df.index[loc + hold_days]

    events = pd.DataFrame(rows)

    if events.empty:
        return events, pd.DataFrame()

    summary_rows = []

    def summarize(label: str, subset: pd.DataFrame) -> None:
        summary_rows.append(
            {
                "group": label,
                "events": int(len(subset)),
                "avg_return_pct": float(subset["stock_return_pct"].mean()),
                "median_return_pct": float(subset["stock_return_pct"].median()),
                "win_rate_pct": float(subset["win"].mean() * 100),
                "avg_excess_vs_qqq_pct": float(
                    subset["excess_return_pct"].mean()
                ),
                "median_excess_vs_qqq_pct": float(
                    subset["excess_return_pct"].median()
                ),
                "beat_qqq_rate_pct": float(
                    subset["beat_qqq"].mean() * 100
                ),
            }
        )

    for rank in sorted(events["rank"].unique()):
        summarize(
            f"Rank {rank}",
            events[events["rank"] == rank],
        )

    summarize("Top 3", events)

    for symbol in sorted(events["symbol"].unique()):
        summarize(
            f"Symbol {symbol}",
            events[events["symbol"] == symbol],
        )

    return events, pd.DataFrame(summary_rows)
