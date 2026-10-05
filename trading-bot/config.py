from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "NVDA"

    # Small-account profile: slower timeframe to reduce noise and turnover.
    interval: str = "1h"
    period: str = "1y"

    ema_fast: int = 20
    ema_slow: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    breakout_period: int = 25
    volume_ma_period: int = 20

    rsi_min: float = 50.0
    rsi_max: float = 70.0

    atr_stop_multiplier: float = 2.0
    risk_reward_ratio: float = 2.0

    # Backtests are priced in USD because Yahoo Finance US-stock data is in USD.
    # Set this to the USD-equivalent of your THB budget after broker FX conversion.
    starting_cash: float = 30.0

    # Preserve capital first: 0.5% risk per trade.
    risk_per_trade: float = 0.005
    max_position_value_pct: float = 0.90

    # Fractional-share support for small accounts.
    allow_fractional_shares: bool = True
    share_step: float = 0.001

    commission_pct: float = 0.0005
    slippage_pct: float = 0.0002


DEFAULT_CONFIG = StrategyConfig()
