from __future__ import annotations

from dataclasses import replace
from itertools import product

import pandas as pd

from config import StrategyConfig
from src.backtest import backtest


def buy_and_hold_return_pct(df: pd.DataFrame) -> float:
    """Simple close-to-close benchmark return."""
    if len(df) < 2:
        return 0.0

    first = float(df["Close"].iloc[0])
    last = float(df["Close"].iloc[-1])

    if first <= 0:
        return 0.0

    return (last / first - 1) * 100


def rolling_out_of_sample(
    df: pd.DataFrame,
    cfg: StrategyConfig,
    *,
    train_bars: int = 600,
    test_bars: int = 200,
) -> pd.DataFrame:
    """
    Evaluate a fixed strategy on sequential unseen test windows.

    The train window is used only to establish temporal separation and provide
    enough historical context for indicators. No parameters are optimized on
    the test window.

    Each fold:
        [train bars][test bars]
                     ^ metrics measured here

    Warm-up bars from the end of the train window are included when computing
    indicators, but trades opened before the test window are discarded.
    """
    if train_bars <= 0 or test_bars <= 0:
        raise ValueError("train_bars and test_bars must be positive.")

    minimum = train_bars + test_bars
    if len(df) < minimum:
        return pd.DataFrame()

    warmup = max(
        cfg.ema_slow,
        cfg.rsi_period,
        cfg.atr_period,
        cfg.breakout_period + 1,
        cfg.volume_ma_period,
    ) + 5

    rows = []
    fold = 1

    test_start = train_bars
    while test_start + test_bars <= len(df):
        test_end = test_start + test_bars

        context_start = max(0, test_start - warmup)
        context = df.iloc[context_start:test_end].copy()
        test_index = df.index[test_start:test_end]

        trades, metrics = backtest(context, cfg)

        if not trades.empty:
            trades = trades[trades["entry_time"].isin(test_index)].copy()

            if trades.empty:
                ending_equity = cfg.starting_cash
                total_return_pct = 0.0
                trade_count = 0
                win_rate_pct = 0.0
                expectancy = 0.0
                profit_factor = None
            else:
                net_pnl = float(trades["net_pnl"].sum())
                ending_equity = cfg.starting_cash + net_pnl
                total_return_pct = net_pnl / cfg.starting_cash * 100
                trade_count = int(len(trades))
                win_rate_pct = float((trades["net_pnl"] > 0).mean() * 100)
                expectancy = float(trades["net_pnl"].mean())

                wins = float(trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum())
                losses = abs(
                    float(trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum())
                )
                profit_factor = wins / losses if losses > 0 else None
        else:
            ending_equity = cfg.starting_cash
            total_return_pct = 0.0
            trade_count = 0
            win_rate_pct = 0.0
            expectancy = 0.0
            profit_factor = None

        test_df = df.iloc[test_start:test_end]

        rows.append(
            {
                "fold": fold,
                "train_start": str(df.index[test_start - train_bars]),
                "train_end": str(df.index[test_start - 1]),
                "test_start": str(test_df.index[0]),
                "test_end": str(test_df.index[-1]),
                "bars": len(test_df),
                "strategy_return_pct": total_return_pct,
                "buy_hold_return_pct": buy_and_hold_return_pct(test_df),
                "trades": trade_count,
                "win_rate_pct": win_rate_pct,
                "profit_factor": profit_factor,
                "expectancy": expectancy,
                "ending_equity": ending_equity,
            }
        )

        fold += 1
        test_start += test_bars

    return pd.DataFrame(rows)


def parameter_sensitivity(
    df: pd.DataFrame,
    cfg: StrategyConfig,
    *,
    breakout_periods: tuple[int, ...] = (15, 20, 25),
    atr_stop_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0),
    risk_reward_ratios: tuple[float, ...] = (1.5, 2.0, 2.5),
) -> pd.DataFrame:
    """
    Compare nearby parameter values.

    This is a robustness diagnostic, not an optimizer. A strategy is more
    credible when neighboring parameter combinations also behave reasonably.
    """
    rows = []

    for breakout, atr_mult, rr in product(
        breakout_periods,
        atr_stop_multipliers,
        risk_reward_ratios,
    ):
        candidate = replace(
            cfg,
            breakout_period=breakout,
            atr_stop_multiplier=atr_mult,
            risk_reward_ratio=rr,
        )

        _, metrics = backtest(df, candidate)

        rows.append(
            {
                "breakout_period": breakout,
                "atr_stop_multiplier": atr_mult,
                "risk_reward_ratio": rr,
                "return_pct": metrics["total_return_pct"],
                "trades": metrics["trades"],
                "win_rate_pct": metrics["win_rate_pct"],
                "profit_factor": metrics["profit_factor"],
                "expectancy": metrics["expectancy"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values(
        ["return_pct", "expectancy"],
        ascending=[False, False],
    ).reset_index(drop=True)
