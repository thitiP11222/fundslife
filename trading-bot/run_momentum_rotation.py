from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import download_ohlcv
from src.momentum_rotation import backtest_momentum_rotation


SYMBOLS = ["NVDA", "AMD", "AAPL", "MSFT", "META"]
BENCHMARK = "QQQ"


def max_drawdown_pct(series: pd.Series) -> float:
    drawdown = series / series.cummax() - 1
    return float(drawdown.min() * 100)


def annual_returns(equity_df: pd.DataFrame) -> pd.DataFrame:
    if equity_df.empty:
        return pd.DataFrame(columns=["year", "return_pct"])

    x = equity_df.copy()
    x["time"] = pd.to_datetime(x["time"])
    x = x.set_index("time")

    yearly = x["equity"].resample("YE").last().pct_change()
    first_year = x["equity"].resample("YE").last().iloc[:1] / x["equity"].iloc[0] - 1
    if len(first_year):
        yearly.iloc[0] = first_year.iloc[0]

    return pd.DataFrame(
        {
            "year": yearly.index.year,
            "return_pct": yearly.values * 100,
        }
    )


def main() -> None:
    interval = "1d"
    period = "5y"

    print(f"Downloading benchmark {BENCHMARK}...")
    benchmark = download_ohlcv(BENCHMARK, interval, period)

    stocks = {}
    for symbol in SYMBOLS:
        print(f"Downloading {symbol}...")
        stocks[symbol] = download_ohlcv(symbol, interval, period)

    trades, metrics = backtest_momentum_rotation(
        stocks,
        benchmark,
        starting_cash=30.0,
        rebalance_every=5,
        momentum_fast=63,
        momentum_slow=126,
        benchmark_ma=200,
        max_position_value_pct=0.95,
        commission_pct=0.0005,
        slippage_pct=0.0002,
    )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    trades_path = results_dir / "momentum_rotation_trades_1d_5y.csv"
    equity_path = results_dir / "momentum_rotation_equity_1d_5y.csv"
    annual_path = results_dir / "momentum_rotation_annual_1d_5y.csv"

    trades.to_csv(trades_path, index=False)

    equity_df = metrics["equity_curve"]
    equity_df.to_csv(equity_path, index=False)

    annual = annual_returns(equity_df)
    annual.to_csv(annual_path, index=False)

    # Benchmark measured across the same available daily history.
    qqq_start = float(benchmark["Close"].iloc[0])
    qqq_end = float(benchmark["Close"].iloc[-1])
    qqq_return = (qqq_end / qqq_start - 1) * 100
    qqq_dd = max_drawdown_pct(benchmark["Close"].astype(float))

    print()
    print("=" * 72)
    print("DAILY MOMENTUM ROTATION BACKTEST - NEXT OPEN EXECUTION")
    print("=" * 72)
    print(f"Starting cash : ${metrics['starting_cash']:.2f}")
    print(f"Ending equity : ${metrics['ending_equity']:.2f}")
    print(f"Return        : {metrics['total_return_pct']:.2f}%")
    print(f"Trades        : {metrics['trades']}")
    print(f"Win rate      : {metrics['win_rate_pct']:.2f}%")
    print(f"Profit factor : {metrics['profit_factor']}")
    print(f"Expectancy    : ${metrics['expectancy']:.4f}/trade")
    print(f"Max drawdown  : {metrics['max_drawdown_pct']:.2f}%")

    print()
    print("=" * 72)
    print("QQQ BUY & HOLD BENCHMARK")
    print("=" * 72)
    print(f"QQQ return    : {qqq_return:.2f}%")
    print(f"QQQ max DD    : {qqq_dd:.2f}%")

    if not annual.empty:
        print()
        print("=" * 72)
        print("STRATEGY RETURNS BY CALENDAR YEAR")
        print("=" * 72)
        print(
            annual.to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}",
            )
        )

    if not trades.empty:
        print()
        print("Trades by symbol:")
        print(trades["symbol"].value_counts().to_string())

    print()
    print(f"Saved trades : {trades_path}")
    print(f"Saved equity : {equity_path}")
    print(f"Saved annual : {annual_path}")


if __name__ == "__main__":
    main()
