from __future__ import annotations

from pathlib import Path

from src.data import download_ohlcv
from src.option_scanner import OptionScanConfig
from src.option_signal_backtest import (
    SignalBacktestConfig,
    backtest_grade_a_signals,
)


SYMBOLS = [
    "NVDA",
    "AMD",
    "AAPL",
    "MSFT",
    "META",
    "GOOGL",
    "AMZN",
    "AVGO",
    "TSLA",
    "NFLX",
]
BENCHMARK = "QQQ"


def main() -> None:
    scan_cfg = OptionScanConfig(
        top_n=3,
        hourly_breakout_period=10,
        min_rsi=50,
        max_rsi=68,
        min_volume_ratio=1.0,
    )
    bt_cfg = SignalBacktestConfig(
        max_holding_bars=35,
        entry_slippage_pct=0.0002,
    )

    print(f"Downloading daily benchmark {BENCHMARK}...")
    benchmark_daily = download_ohlcv(BENCHMARK, "1d", "2y")

    daily_data = {}
    hourly_data = {}

    for symbol in SYMBOLS:
        print(f"Loading {symbol}...")
        daily_data[symbol] = download_ohlcv(symbol, "1d", "2y")
        hourly_data[symbol] = download_ohlcv(symbol, "1h", "1y")

    events, metrics = backtest_grade_a_signals(
        daily_data,
        hourly_data,
        benchmark_daily,
        scan_cfg,
        bt_cfg,
    )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    events_path = results_dir / "option_grade_a_backtest.csv"
    events.to_csv(events_path, index=False)

    print()
    print("=" * 80)
    print("OPTION SCANNER V3 - GRADE A HISTORICAL EVALUATION")
    print("=" * 80)
    print(f"Grade A signals       : {metrics['signals']}")
    print(f"T1 hit rate           : {metrics['t1_hit_rate_pct']:.2f}%")
    print(f"T2 hit rate           : {metrics['t2_hit_rate_pct']:.2f}%")
    print(f"Invalidation rate     : {metrics['invalidation_rate_pct']:.2f}%")
    print(
        f"Avg underlying return : "
        f"{metrics['avg_underlying_return_pct']:.2f}%"
    )
    print(f"Avg R multiple        : {metrics['avg_r_multiple']:.3f}R")
    print(f"Avg holding bars      : {metrics['avg_bars_held']:.2f}")

    if not events.empty:
        print()
        print("Outcome counts:")
        print(events["outcome"].value_counts().to_string())

        print()
        print("Signals by symbol:")
        print(events["symbol"].value_counts().to_string())

    print()
    print(f"Saved: {events_path}")


if __name__ == "__main__":
    main()
