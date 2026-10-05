from __future__ import annotations

from pathlib import Path

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
    result.to_csv(output, index=False)

    print()
    print("=" * 118)
    print("DIME OPTIONS UNDERLYING SCANNER")
    print("=" * 118)

    if result.empty:
        print("No data available.")
        return

    display = [
        "momentum_rank",
        "symbol",
        "option_signal",
        "daily_momentum_score",
        "entry_score",
        "underlying_close",
        "entry_trigger",
        "invalidation",
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

    calls = result[result["option_signal"] == "CALL_CANDIDATE"]

    print()
    print("=" * 118)
    print("ACTIONABLE CANDIDATES")
    print("=" * 118)

    if calls.empty:
        print("No CALL candidate now. WAIT.")
    else:
        for _, row in calls.iterrows():
            print(
                f"{row['symbol']}: CALL CANDIDATE | "
                f"underlying > {row['entry_trigger']:.2f} | "
                f"invalidation < {row['invalidation']:.2f} | "
                f"score {int(row['entry_score'])}/100"
            )

    print()
    print("Dime contract checklist (manual option-chain step):")
    print("- Prefer 21-45 DTE rather than very short-dated contracts.")
    print("- Prefer liquid contracts with a tight bid/ask spread.")
    print("- Avoid using the whole account premium on one trade.")
    print("- Use the underlying invalidation level to exit the thesis.")
    print("- This scanner validates CALL direction only; PUT is not enabled yet.")

    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
