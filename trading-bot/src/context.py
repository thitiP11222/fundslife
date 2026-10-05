from __future__ import annotations

import pandas as pd

from src.indicators import ema


def add_market_context(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    *,
    rs_lookback: int = 20,
    benchmark_fast_ema: int = 50,
    benchmark_slow_ema: int = 200,
) -> pd.DataFrame:
    """
    Align benchmark data to stock bars and add:
    - benchmark trend regime
    - stock relative strength vs benchmark over a lookback window

    Relative strength is defined as:
        stock_return - benchmark_return
    """
    stock = stock_df.copy().sort_index()
    benchmark = benchmark_df.copy().sort_index()

    aligned = benchmark[["Close"]].rename(
        columns={"Close": "BENCH_CLOSE"}
    ).reindex(stock.index).ffill()

    stock["BENCH_CLOSE"] = aligned["BENCH_CLOSE"]
    stock["BENCH_EMA_FAST"] = ema(
        stock["BENCH_CLOSE"], benchmark_fast_ema
    )
    stock["BENCH_EMA_SLOW"] = ema(
        stock["BENCH_CLOSE"], benchmark_slow_ema
    )

    stock_return = stock["Close"].pct_change(rs_lookback)
    benchmark_return = stock["BENCH_CLOSE"].pct_change(rs_lookback)
    stock["RS_VS_BENCH"] = stock_return - benchmark_return

    return stock
