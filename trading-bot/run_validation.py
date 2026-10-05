from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from config import DEFAULT_CONFIG
from src.backtest import backtest
from src.data import download_ohlcv
from src.validation import (
    buy_and_hold_return_pct,
    parameter_sensitivity,
    rolling_out_of_sample,
)


def main() -> None:
    # Yahoo Finance intraday history is limited, so use the widest practical
    # window for 15-minute research while keeping the strategy unchanged.
    cfg = replace(DEFAULT_CONFIG, period="60d")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print(f"Downloading {cfg.symbol} {cfg.interval} data for {cfg.period}...")
    df = download_ohlcv(cfg.symbol, cfg.interval, cfg.period)

    _, full_metrics = backtest(df, cfg)
    benchmark = buy_and_hold_return_pct(df)

    print()
    print("=" * 64)
    print("FULL-SAMPLE CHECK")
    print("=" * 64)
    print(f"Strategy return : {full_metrics['total_return_pct']:.2f}%")
    print(f"Buy & Hold      : {benchmark:.2f}%")
    print(f"Trades          : {full_metrics['trades']}")
    print(f"Win rate        : {full_metrics['win_rate_pct']:.2f}%")
    print(f"Profit factor   : {full_metrics['profit_factor']}")
    print(f"Expectancy      : ${full_metrics['expectancy']:.2f}/trade")
    print(f"Max drawdown    : {full_metrics['max_drawdown_pct']:.2f}%")

    folds = rolling_out_of_sample(
        df,
        cfg,
        train_bars=600,
        test_bars=200,
    )

    folds_path = results_dir / f"walk_forward_{cfg.symbol}_{cfg.interval}.csv"
    folds.to_csv(folds_path, index=False)

    print()
    print("=" * 64)
    print("ROLLING OUT-OF-SAMPLE")
    print("=" * 64)

    if folds.empty:
        print("Not enough bars for the selected train/test windows.")
    else:
        print(f"Folds           : {len(folds)}")
        print(f"Positive folds  : {(folds['strategy_return_pct'] > 0).sum()}")
        print(
            "Avg OOS return  : "
            f"{folds['strategy_return_pct'].mean():.2f}% per fold"
        )
        print(
            "Avg Buy & Hold  : "
            f"{folds['buy_hold_return_pct'].mean():.2f}% per fold"
        )
        print(f"Total OOS trades: {int(folds['trades'].sum())}")

    sensitivity = parameter_sensitivity(df, cfg)
    sensitivity_path = (
        results_dir / f"parameter_sensitivity_{cfg.symbol}_{cfg.interval}.csv"
    )
    sensitivity.to_csv(sensitivity_path, index=False)

    print()
    print("=" * 64)
    print("PARAMETER SENSITIVITY - TOP 10")
    print("=" * 64)
    if sensitivity.empty:
        print("No sensitivity results.")
    else:
        print(
            sensitivity.head(10).to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )

    print()
    print(f"Saved OOS folds : {folds_path}")
    print(f"Saved sensitivity: {sensitivity_path}")


if __name__ == "__main__":
    main()
