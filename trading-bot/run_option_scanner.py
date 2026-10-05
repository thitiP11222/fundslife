from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import download_ohlcv
from src.option_scanner import OptionScanConfig, scan_option_candidates


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


def append_signal_log(result: pd.DataFrame, path: Path) -> None:
    if result.empty:
        return

    snapshot = result.copy()
    snapshot.insert(
        0,
        "scan_time_utc",
        pd.Timestamp.now(tz="UTC").isoformat(),
    )

    write_header = not path.exists()
    snapshot.to_csv(
        path,
        mode="a",
        header=write_header,
        index=False,
    )


def main() -> None:
    cfg = OptionScanConfig(
        top_n=3,
        hourly_breakout_period=10,
        min_rsi=50,
        max_rsi=68,
        min_volume_ratio=1.0,
    )

    print(f"Downloading daily benchmark {BENCHMARK}...")
    benchmark_daily = download_ohlcv(BENCHMARK, "1d", "2y")

    daily_data = {}
    hourly_data = {}

    for symbol in SYMBOLS:
        print(f"Scanning {symbol}...")
        daily_data[symbol] = download_ohlcv(symbol, "1d", "2y")
        hourly_data[symbol] = download_ohlcv(symbol, "1h", "60d")

    result = scan_option_candidates(
        daily_data,
        hourly_data,
        benchmark_daily,
        cfg,
    )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output = results_dir / "option_candidates.csv"
    signal_log = results_dir / "option_signal_log.csv"

    result.to_csv(output, index=False)
    append_signal_log(result, signal_log)

    print()
    print("=" * 132)
    print("DIME OPTIONS UNDERLYING SCANNER V2")
    print("=" * 132)

    if result.empty:
        print("No data available.")
        return

    display = [
        "momentum_rank",
        "symbol",
        "setup_grade",
        "option_signal",
        "daily_momentum_score",
        "entry_score",
        "underlying_close",
        "entry_trigger",
        "invalidation",
        "target_1",
        "target_2",
        "rsi_1h",
        "volume_ratio_1h",
        "entry_reason",
    ]

    print(
        result[display].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print()
    print("=" * 132)
    print("SETUP GUIDE")
    print("=" * 132)
    print("A    = CALL candidate; open Dime and inspect the option chain.")
    print("B    = Strong watchlist; wait for the missing 1H confirmation.")
    print("C    = Early watchlist; no options entry yet.")
    print("WAIT = Ignore for now.")

    calls = result[result["setup_grade"] == "A"]
    watch = result[result["setup_grade"].isin(["B", "C"])]

    print()
    print("=" * 132)
    print("ACTIONABLE / WATCHLIST")
    print("=" * 132)

    if calls.empty:
        print("No Grade-A CALL candidate now.")
    else:
        for _, row in calls.iterrows():
            print(
                f"{row['symbol']} | Grade A CALL | "
                f"1H close > {row['entry_trigger']:.2f} | "
                f"invalidation < {row['invalidation']:.2f} | "
                f"T1 {row['target_1']:.2f} | T2 {row['target_2']:.2f}"
            )

    if not watch.empty:
        print()
        print("Watch:")
        for _, row in watch.head(5).iterrows():
            print(
                f"{row['symbol']} | Grade {row['setup_grade']} | "
                f"{row['entry_reason']}"
            )

    print()
    print("Important: targets and invalidation are UNDERLYING stock prices,")
    print("not option-premium targets. PUT signals are not enabled yet.")

    print()
    print(f"Saved current scan: {output}")
    print(f"Appended signal log: {signal_log}")


if __name__ == "__main__":
    main()
