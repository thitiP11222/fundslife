from __future__ import annotations

from pathlib import Path

from src.data import download_ohlcv
from src.momentum_event_study import momentum_forward_event_study


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

    events, summary = momentum_forward_event_study(
        stocks,
        benchmark,
        top_n=3,
        benchmark_ma_days=200,
        forward_days=(5, 10, 20),
    )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    events_path = results_dir / "momentum_event_study_events.csv"
    summary_path = results_dir / "momentum_event_study_summary.csv"

    events.to_csv(events_path, index=False)
    summary.to_csv(summary_path, index=False)

    print()
    print("=" * 100)
    print("MOMENTUM EVENT STUDY - FORWARD RETURNS")
    print("=" * 100)

    if summary.empty:
        print("No valid events.")
    else:
        print("Rank 0 = aggregate Top-3 events")
        print(
            summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}",
            )
        )

    if not events.empty:
        print()
        print("Events by symbol:")
        print(events["symbol"].value_counts().to_string())

    print()
    print(f"Saved events : {events_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
