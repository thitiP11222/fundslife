from __future__ import annotations

from pathlib import Path

from src.data import download_ohlcv
from src.momentum_robustness import non_overlapping_momentum_study


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
    print(f"Downloading benchmark {BENCHMARK}...")
    benchmark = download_ohlcv(BENCHMARK, "1d", "5y")

    stocks = {}
    for symbol in SYMBOLS:
        print(f"Loading {symbol}...")
        stocks[symbol] = download_ohlcv(symbol, "1d", "5y")

    events, summary = non_overlapping_momentum_study(
        stocks,
        benchmark,
        top_n=3,
        benchmark_ma_days=200,
        hold_days=20,
    )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    events_path = results_dir / "momentum_robustness_events.csv"
    summary_path = results_dir / "momentum_robustness_summary.csv"

    events.to_csv(events_path, index=False)
    summary.to_csv(summary_path, index=False)

    print()
    print("=" * 104)
    print("MOMENTUM ROBUSTNESS - NON-OVERLAPPING 20D EVENTS")
    print("=" * 104)

    if summary.empty:
        print("No valid events.")
    else:
        print(
            summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}",
            )
        )

    print()
    print(f"Saved events : {events_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
