from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import DEFAULT_CONFIG
from src.backtest import backtest
from src.context import add_market_context
from src.data import download_ohlcv
from src.validation import buy_and_hold_return_pct


SYMBOLS = ["NVDA", "AMD", "AAPL", "MSFT", "META"]
BENCHMARK = "QQQ"


def evaluate_symbol(symbol: str, benchmark_df: pd.DataFrame) -> list[dict]:
    df = download_ohlcv(symbol, "1h", "1y")
    context_df = add_market_context(df, benchmark_df)

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
            exit_mode="fixed_rr",
            use_market_regime=False,
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
            exit_mode="fixed_rr",
            use_market_regime=False,
        ),
        replace(
            DEFAULT_CONFIG,
            symbol=symbol,
            strategy_name="pullback_reclaim",
            rsi_min=45,
            rsi_max=65,
            volume_ratio_min=0.8,
            atr_stop_multiplier=2.0,
            exit_mode="trend_trailing",
            trailing_atr_multiplier=1.0,
            use_market_regime=False,
        ),
        replace(
            DEFAULT_CONFIG,
            symbol=symbol,
            strategy_name="pullback_reclaim",
            rsi_min=45,
            rsi_max=65,
            volume_ratio_min=0.8,
            atr_stop_multiplier=2.0,
            exit_mode="trend_trailing",
            trailing_atr_multiplier=1.0,
            use_market_regime=True,
            rs_min=0.0,
        ),
    ]

    rows = []
    benchmark = buy_and_hold_return_pct(df)

    for cfg in candidates:
        test_df = context_df if cfg.use_market_regime else df
        _, metrics = backtest(test_df, cfg)

        label = cfg.strategy_name
        if cfg.exit_mode == "trend_trailing":
            label += "_trailing"
        if cfg.use_market_regime:
            label += "_regime_rs"

        rows.append(
            {
                "symbol": symbol,
                "strategy": label,
                "return_pct": metrics["total_return_pct"],
                "buy_hold_pct": benchmark,
                "trades": metrics["trades"],
                "win_rate_pct": metrics["win_rate_pct"],
                "profit_factor": metrics["profit_factor"],
                "payoff_ratio": metrics["payoff_ratio"],
                "expectancy_usd": metrics["expectancy"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "return_to_drawdown": metrics["return_to_drawdown"],
            }
        )

    return rows


def main() -> None:
    rows = []

    print(f"Downloading benchmark {BENCHMARK}...")
    benchmark_df = download_ohlcv(BENCHMARK, "1h", "1y")

    for symbol in SYMBOLS:
        print(f"Evaluating {symbol}...")
        try:
            rows.extend(evaluate_symbol(symbol, benchmark_df))
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
    print("=" * 120)
    print("MULTI-SYMBOL STRATEGY COMPARISON")
    print("=" * 120)

    if not result.empty:
        display_cols = [
            "symbol",
            "strategy",
            "return_pct",
            "buy_hold_pct",
            "trades",
            "win_rate_pct",
            "profit_factor",
            "payoff_ratio",
            "expectancy_usd",
            "max_drawdown_pct",
            "return_to_drawdown",
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
                    avg_profit_factor=("profit_factor", "mean"),
                    avg_payoff_ratio=("payoff_ratio", "mean"),
                    avg_max_drawdown_pct=("max_drawdown_pct", "mean"),
                    avg_return_to_drawdown=("return_to_drawdown", "mean"),
                )
                .reset_index()
            )

            print()
            print("=" * 120)
            print("AGGREGATE")
            print("=" * 120)
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
