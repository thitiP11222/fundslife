from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "NVDA"
    interval: str = "15m"
    period: str = "30d"

    ema_fast: int = 20
    ema_slow: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    breakout_period: int = 20
    volume_ma_period: int = 20

    rsi_min: float = 50.0
    rsi_max: float = 70.0

    atr_stop_multiplier: float = 1.5
    risk_reward_ratio: float = 2.0

    starting_cash: float = 10_000.0
    risk_per_trade: float = 0.01
    max_position_value_pct: float = 0.25

    commission_pct: float = 0.0005
    slippage_pct: float = 0.0002


DEFAULT_CONFIG = StrategyConfig()
