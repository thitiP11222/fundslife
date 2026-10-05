from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import DEFAULT_CONFIG
from src.backtest import backtest
from src.data import download_ohlcv
from src.validation import buy_and_hold_return_pct


SYMBOLS = ["NVDA", "AMD", "AAPL", "MSFT", "META"]


def evaluate_symbol(symbol: str) -> list[dict]:
    df = download_ohlcv(symbol, "1h", "1y")

    candidates = [
        replace(
            DEFAULT_CONFIG,
            symbol=symbol,
            strategy_name="breakout",
            breakout_period=25,
            rsi_min=50,
            rsi_max=70,
            volume_ratio_min=1.0,
            atr_stop_multiplier=1.0,
            risk_reward_ratio=2.0,
        ),
        replace(
            DEFAULT_CONFIG,
            symbol=symbol,
            strategy_name="pullback_reclaim",
            rsi_min=45,
            rsi_max=65,
            volume_ratio_min=0.8,
            atr_stop_multiplier=1.5,
            risk_reward_ratio=2.0,
        ),
    ]

    rows = []
    benchmark = buy_and_hold_return_pct(df)

    for cfg in candidates:
        _, metrics = backtest(df, cfg)
        rows.append(
            {
                "symbol": symbol,
                "strategy": cfg.strategy_name,
                "return_pct": metrics["total_return_pct"],
                "buy_hold_pct": benchmark,
                "excess_vs_buy_hold_pct": (
                    metrics["total_return_pct"] - benchmark
                ),
                "trades": metrics["trades"],
                "win_rate_pct": metrics["win_rate_pct"],
                "profit_factor": metrics["profit_factor"],
                "expectancy_usd": metrics["expectancy"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
            }
        )

    return rows


def main() -> None:
    rows = []

    for symbol in SYMBOLS:
        print(f"Evaluating {symbol}...")
        try:
            rows.extend(evaluate_symbol(symbol))
        except Exception as exc:
            rows.append(
                {
                    "symbol": symbol,
                    "strategy": "ERROR",
                    "error": str(exc),
                }
            )

    result = pd.DataFrame(rows)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output = results_dir / "strategy_comparison_1h_1y.csv"
    result.to_csv(output, index=False)

    print()
    print("=" * 96)
    print("MULTI-SYMBOL STRATEGY COMPARISON")
    print("=" * 96)

    if not result.empty:
        display_cols = [
            "symbol",
            "strategy",
            "return_pct",
            "buy_hold_pct",
            "excess_vs_buy_hold_pct",
            "trades",
            "win_rate_pct",
            "profit_factor",
            "expectancy_usd",
            "max_drawdown_pct",
        ]
        available = [c for c in display_cols if c in result.columns]
        print(
            result[available].to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )

        valid = result[result["strategy"] != "ERROR"].copy()

        if not valid.empty:
            summary = (
                valid.groupby("strategy")
                .agg(
                    symbols=("symbol", "count"),
                    profitable_symbols=("return_pct", lambda s: int((s > 0).sum())),
                    total_trades=("trades", "sum"),
                    avg_return_pct=("return_pct", "mean"),
                    avg_excess_vs_buy_hold_pct=("excess_vs_buy_hold_pct", "mean"),
                    avg_profit_factor=("profit_factor", "mean"),
                    avg_max_drawdown_pct=("max_drawdown_pct", "mean"),
                )
                .reset_index()
            )

            print()
            print("=" * 96)
            print("AGGREGATE")
            print("=" * 96)
            print(
                summary.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.3f}",
                )
            )

    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
