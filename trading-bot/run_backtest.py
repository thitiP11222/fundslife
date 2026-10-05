import json
from pathlib import Path

from config import DEFAULT_CONFIG
from src.backtest import backtest
from src.data import download_ohlcv


def main() -> None:
    cfg = DEFAULT_CONFIG
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print(f"Downloading {cfg.symbol} {cfg.interval} data...")
    df = download_ohlcv(cfg.symbol, cfg.interval, cfg.period)
    trades, metrics = backtest(df, cfg)

    trades_path = results_dir / f"backtest_{cfg.symbol}_{cfg.interval}.csv"
    metrics_path = results_dir / f"metrics_{cfg.symbol}_{cfg.interval}.json"

    trades.to_csv(trades_path, index=False)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print("=" * 56)
    print("BACKTEST RESULT")
    print("=" * 56)
    print(f"Starting cash : ${metrics['starting_cash']:.2f}")
    print(f"Ending equity : ${metrics['ending_equity']:.2f}")
    print(f"Return        : {metrics['total_return_pct']:.2f}%")
    print(f"Trades        : {metrics['trades']}")
    print(f"Win rate      : {metrics['win_rate_pct']:.2f}%")
    print(f"Profit factor : {metrics['profit_factor']}")
    print(f"Expectancy    : ${metrics['expectancy']:.2f}/trade")
    print(f"Max drawdown  : {metrics['max_drawdown_pct']:.2f}%")
    print()
    print(f"Trades saved : {trades_path}")
    print(f"Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
