from __future__ import annotations

from pathlib import Path

from src.data import download_ohlcv
from src.momentum_rotation import backtest_momentum_rotation


SYMBOLS = ["NVDA", "AMD", "AAPL", "MSFT", "META"]
BENCHMARK = "QQQ"


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
    trades.to_csv(trades_path, index=False)

    print()
    print("=" * 72)
    print("DAILY MOMENTUM ROTATION BACKTEST")
    print("=" * 72)
    print(f"Starting cash : ${metrics['starting_cash']:.2f}")
    print(f"Ending equity : ${metrics['ending_equity']:.2f}")
    print(f"Return        : {metrics['total_return_pct']:.2f}%")
    print(f"Trades        : {metrics['trades']}")
    print(f"Win rate      : {metrics['win_rate_pct']:.2f}%")
    print(f"Profit factor : {metrics['profit_factor']}")
    print(f"Expectancy    : ${metrics['expectancy']:.4f}/trade")
    print(f"Max drawdown  : {metrics['max_drawdown_pct']:.2f}%")

    if not trades.empty:
        print()
        print("Trades by symbol:")
        print(trades["symbol"].value_counts().to_string())

    print()
    print(f"Saved trades: {trades_path}")


if __name__ == "__main__":
    main()
