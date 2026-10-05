from __future__ import annotations

from pathlib import Path

from src.data import download_ohlcv
from src.unified_momentum_scanner import (
    UnifiedMomentumConfig,
    scan_unified_momentum,
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
    cfg = UnifiedMomentumConfig(
        top_n=3,
        min_price_usd=5.0,
        min_avg_dollar_volume_20d=20_000_000.0,
        near_52w_high_ratio=0.90,
    )

    print(f"Downloading benchmark {BENCHMARK}...")
    benchmark = download_ohlcv(BENCHMARK, "1d", "5y")

    stocks = {}
    for symbol in SYMBOLS:
        print(f"Loading {symbol}...")
        stocks[symbol] = download_ohlcv(symbol, "1d", "5y")

    result, regime = scan_unified_momentum(
        stocks,
        benchmark,
        cfg,
    )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output = results_dir / "unified_momentum_candidates.csv"
    result.to_csv(output, index=False)

    print()
    print("=" * 132)
    print("UNIFIED MOMENTUM SCANNER")
    print("=" * 132)
    print(
        f"Market regime: "
        f"{'RISK-ON' if regime['risk_on'] else 'RISK-OFF'} | "
        f"QQQ {regime['benchmark_close']:.2f} vs "
        f"MA200 {regime['benchmark_ma200']:.2f}"
    )

    if result.empty:
        print("No valid candidates.")
        return

    display = [
        "momentum_rank",
        "symbol",
        "fractional_action",
        "dr_action",
        "option_action",
        "momentum_3m_pct",
        "momentum_6m_pct",
        "momentum_12m_pct",
        "price_to_52w_high",
        "trend_ok",
        "atr_pct",
        "avg_dollar_volume_20d",
    ]

    print()
    print(
        result[display].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print()
    print("=" * 132)
    print("HOW TO USE")
    print("=" * 132)
    print("Fractional:")
    print("- BUY_CANDIDATE = core momentum candidate; execute only with a predefined allocation/risk rule.")
    print("- WATCH = positive trend/momentum but outside Top 3.")
    print()
    print("DR / DRx:")
    print("- CHECK_DR_OR_DRX = first confirm a listed DR/DRx tracks this underlying and has acceptable liquidity/spread.")
    print()
    print("Options:")
    print("- REVIEW_CALL_CHAIN = research candidate only; inspect DTE, strike, premium, spread and liquidity manually.")
    print("- Never interpret REVIEW_CALL_CHAIN as an automatic buy order.")
    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
