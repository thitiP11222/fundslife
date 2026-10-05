from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from config import DEFAULT_CONFIG
from src.data import download_ohlcv
from src.rotation import backtest_top1_rotation


SYMBOLS = ["NVDA", "AMD", "AAPL", "MSFT", "META"]
BENCHMARK = "QQQ"


def main() -> None:
    cfg = replace(
        DEFAULT_CONFIG,
        strategy_name="pullback_reclaim",
        interval="1h",
        period="1y",
        rsi_min=45,
        rsi_max=65,
        volume_ratio_min=0.8,
        atr_stop_multiplier=2.0,
        trailing_atr_multiplier=1.0,
        exit_mode="trend_trailing",
        risk_per_trade=0.005,
        max_position_value_pct=0.90,
    )

    print(f"Downloading benchmark {BENCHMARK}...")
    benchmark = download_ohlcv(BENCHMARK, cfg.interval, cfg.period)

    stock_data = {}
    for symbol in SYMBOLS:
        print(f"Downloading {symbol}...")
        stock_data[symbol] = download_ohlcv(
            symbol,
            cfg.interval,
            cfg.period,
        )

    trades, metrics = backtest_top1_rotation(
        stock_data,
        benchmark,
        cfg,
        rs_lookback=20,
    )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    trades_path = results_dir / "rotation_trades_1h_1y.csv"
    trades.to_csv(trades_path, index=False)

    print()
    print("=" * 72)
    print("TOP-1 ROTATION BACKTEST")
    print("=" * 72)
    print(f"Starting cash : ${metrics['starting_cash']:.2f}")
    print(f"Ending equity : ${metrics['ending_equity']:.2f}")
    print(f"Return        : {metrics['total_return_pct']:.2f}%")
    print(f"Trades        : {metrics['trades']}")
    print(f"Win rate      : {metrics['win_rate_pct']:.2f}%")
    print(f"Profit factor : {metrics['profit_factor']}")
    print(f"Expectancy    : ${metrics['expectancy']:.4f}/trade")
    print(f"Max drawdown  : {metrics['max_drawdown_pct']:.2f}%")

    if not trades.empty:
        print()
        print("Trades by symbol:")
        print(trades["symbol"].value_counts().to_string())

    print()
    print(f"Saved trades: {trades_path}")


if __name__ == "__main__":
    main()
