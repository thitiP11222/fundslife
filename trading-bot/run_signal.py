from config import DEFAULT_CONFIG
from src.data import download_ohlcv
from src.strategy import latest_signal


def main() -> None:
    cfg = DEFAULT_CONFIG
    df = download_ohlcv(cfg.symbol, cfg.interval, cfg.period)
    result = latest_signal(df, cfg)

    print("=" * 56)
    print("FUNDSLIFE STOCK TRADING BOT - SIGNAL")
    print("=" * 56)
    for key, value in result.items():
        print(f"{key:>12}: {value}")


if __name__ == "__main__":
    main()
