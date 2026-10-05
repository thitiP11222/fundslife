from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from config import StrategyConfig
from src.risk import position_size_from_risk
from src.strategy import generate_signals


def backtest(
    df: pd.DataFrame,
    cfg: StrategyConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = generate_signals(df, cfg)

    cash = cfg.starting_cash
    position = None
    trades = []

    for i in range(1, len(data)):
        ts = data.index[i]
        row = data.iloc[i]
        prev = data.iloc[i - 1]

        if position is None and int(prev["SIGNAL"]) == 1 and pd.notna(prev["ATR"]):
            entry = float(row["Open"]) * (1 + cfg.slippage_pct)
            stop_distance = float(prev["ATR"]) * cfg.atr_stop_multiplier
            stop = entry - stop_distance
            target = entry + stop_distance * cfg.risk_reward_ratio

            shares = position_size_from_risk(
                equity=cash,
                risk_fraction=cfg.risk_per_trade,
                entry_price=entry,
                stop_price=stop,
                max_position_value_pct=cfg.max_position_value_pct,
            )

            if shares > 0:
                position = {
                    "entry_time": ts,
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "shares": shares,
                    "entry_commission": entry * shares * cfg.commission_pct,
                }

        if position is None:
            continue

        low = float(row["Low"])
        high = float(row["High"])
        raw_exit = None
        exit_reason = None

        # Conservative intrabar assumption: stop is checked before target.
        if low <= position["stop"]:
            raw_exit = position["stop"]
            exit_reason = "STOP"
        elif high >= position["target"]:
            raw_exit = position["target"]
            exit_reason = "TARGET"

        if raw_exit is None:
            continue

        exit_price = float(raw_exit) * (1 - cfg.slippage_pct)
        shares = int(position["shares"])
        gross_pnl = (exit_price - position["entry"]) * shares
        exit_commission = exit_price * shares * cfg.commission_pct
        net_pnl = gross_pnl - position["entry_commission"] - exit_commission
        cash += net_pnl

        trades.append(
            {
                "entry_time": position["entry_time"],
                "exit_time": ts,
                "shares": shares,
                "entry": position["entry"],
                "exit": exit_price,
                "stop": position["stop"],
                "target": position["target"],
                "exit_reason": exit_reason,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "equity_after": cash,
            }
        )
        position = None

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        metrics = {
            "starting_cash": cfg.starting_cash,
            "ending_equity": cash,
            "total_return_pct": 0.0,
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "expectancy": 0.0,
            "max_drawdown_pct": 0.0,
            "config": asdict(cfg),
        }
        return trades_df, metrics

    wins = trades_df.loc[trades_df["net_pnl"] > 0, "net_pnl"]
    losses = trades_df.loc[trades_df["net_pnl"] < 0, "net_pnl"]

    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    equity = pd.Series(
        [cfg.starting_cash, *trades_df["equity_after"].tolist()],
        dtype=float,
    )
    drawdown = equity / equity.cummax() - 1

    metrics = {
        "starting_cash": cfg.starting_cash,
        "ending_equity": float(cash),
        "total_return_pct": float((cash / cfg.starting_cash - 1) * 100),
        "trades": int(len(trades_df)),
        "win_rate_pct": float((trades_df["net_pnl"] > 0).mean() * 100),
        "profit_factor": float(profit_factor) if profit_factor is not None else None,
        "expectancy": float(trades_df["net_pnl"].mean()),
        "max_drawdown_pct": float(drawdown.min() * 100),
        "config": asdict(cfg),
    }
    return trades_df, metrics
