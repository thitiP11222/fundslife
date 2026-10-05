from __future__ import annotations

from pathlib import Path

from src.data import download_ohlcv
from src.final_option_strategy import (
    FinalOptionConfig,
    scan_final_option_candidates,
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
    cfg = FinalOptionConfig(
        top_n=3,
        hold_days=20,
        min_dte=30,
        max_dte=60,
        max_extension_atr=1.0,
    )

    print(f"Downloading benchmark {BENCHMARK}...")
    benchmark = download_ohlcv(
        BENCHMARK,
        interval="1d",
        period="5y",
    )

    stocks = {}
    for symbol in SYMBOLS:
        print(f"Loading {symbol}...")
        stocks[symbol] = download_ohlcv(
            symbol,
            interval="1d",
            period="5y",
        )

    result, meta = scan_final_option_candidates(
        stocks,
        benchmark,
        cfg,
    )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output = results_dir / "final_option_candidates.csv"
    result.to_csv(output, index=False)

    print()
    print("=" * 108)
    print("FINAL DIME OPTIONS STOCK SELECTION")
    print("=" * 108)

    print(
        f"QQQ regime      : "
        f"{'RISK-ON' if meta['market_risk_on'] else 'RISK-OFF'}"
    )
    print(
        f"QQQ Close/MA200 : "
        f"{meta['qqq_close']:.2f} / {meta['qqq_ma200']:.2f}"
    )
    print(
        f"Research horizon: ~{meta['hold_days']} trading days"
    )
    print(
        f"Option DTE guide: {meta['min_dte']}-{meta['max_dte']} DTE"
    )

    print()
    display = [
        "momentum_rank",
        "symbol",
        "action",
        "momentum_3m_pct",
        "momentum_6m_pct",
        "momentum_score",
        "trend_ok",
        "close",
        "extension_atr",
    ]

    if result.empty:
        print("No valid candidates.")
        return

    print(
        result[display].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    review = result[result["action"] == "REVIEW_CALL"]
    wait = result[result["action"] == "WAIT_NOT_CHASE"]

    print()
    print("=" * 108)
    print("FINAL ACTION")
    print("=" * 108)

    if not meta["market_risk_on"]:
        print("NO TRADE: QQQ is below its 200-day moving average.")
    elif review.empty:
        print("NO CALL ENTRY: no Top-3 candidate is both trend-qualified and unextended.")
    else:
        print("Open Dime and review CALL option chains only for:")
        for _, row in review.iterrows():
            print(
                f"- {row['symbol']} | Rank #{int(row['momentum_rank'])} | "
                f"3M {row['momentum_3m_pct']:.2f}% | "
                f"6M {row['momentum_6m_pct']:.2f}%"
            )

    if not wait.empty:
        print()
        print("Do not chase:")
        for _, row in wait.iterrows():
            print(
                f"- {row['symbol']} | extended "
                f"{row['extension_atr']:.2f} ATR above EMA50"
            )

    print()
    print("Manual Dime option-chain checklist:")
    print("- Prefer CALL only while market regime is RISK-ON.")
    print("- Use roughly 30-60 DTE for a ~20-trading-day research horizon.")
    print("- Prefer ATM or slightly ITM over far OTM lottery contracts.")
    print("- Avoid wide bid/ask spreads and illiquid contracts.")
    print("- Do not use the entire account premium on one contract.")
    print("- No auto-ordering is enabled.")

    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
