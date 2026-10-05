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
    equity_curve = [cfg.starting_cash]

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

        if position is not None:
            open_price = float(row["Open"])
            low = float(row["Low"])
            high = float(row["High"])
            raw_exit = None
            exit_reason = None

            # A stop order can gap through its trigger. In that case the first
            # executable approximation is the opening price, not the stop level.
            if open_price <= position["stop"]:
                raw_exit = open_price
                exit_reason = "STOP_GAP"
            elif low <= position["stop"]:
                raw_exit = position["stop"]
                exit_reason = "STOP"
            elif high >= position["target"]:
                raw_exit = position["target"]
                exit_reason = "TARGET"

            if raw_exit is not None:
                exit_price = float(raw_exit) * (1 - cfg.slippage_pct)
                shares = int(position["shares"])
                gross_pnl = (exit_price - position["entry"]) * shares
                exit_commission = exit_price * shares * cfg.commission_pct
                net_pnl = (
                    gross_pnl
                    - position["entry_commission"]
                    - exit_commission
                )
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

        # Mark open positions to market on every candle so drawdown captures
        # adverse movement before a trade is closed.
        if position is None:
            marked_equity = cash
        else:
            shares = int(position["shares"])
            unrealized = (float(row["Close"]) - position["entry"]) * shares
            marked_equity = cash + unrealized - position["entry_commission"]

        equity_curve.append(marked_equity)

    # Close any still-open position at the last available close so ending
    # equity and return are not based on an ignored position.
    if position is not None and len(data) > 0:
        ts = data.index[-1]
        raw_exit = float(data.iloc[-1]["Close"])
        exit_price = raw_exit * (1 - cfg.slippage_pct)
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
                "exit_reason": "END_OF_DATA",
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "equity_after": cash,
            }
        )
        equity_curve[-1] = cash
        position = None

    trades_df = pd.DataFrame(trades)

    equity = pd.Series(equity_curve, dtype=float)
    drawdown = equity / equity.cummax() - 1
    max_drawdown_pct = float(drawdown.min() * 100)

    if trades_df.empty:
        metrics = {
            "starting_cash": cfg.starting_cash,
            "ending_equity": cash,
            "total_return_pct": float((cash / cfg.starting_cash - 1) * 100),
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "expectancy": 0.0,
            "max_drawdown_pct": max_drawdown_pct,
            "config": asdict(cfg),
        }
        return trades_df, metrics

    wins = trades_df.loc[trades_df["net_pnl"] > 0, "net_pnl"]
    losses = trades_df.loc[trades_df["net_pnl"] < 0, "net_pnl"]

    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    metrics = {
        "starting_cash": cfg.starting_cash,
        "ending_equity": float(cash),
        "total_return_pct": float((cash / cfg.starting_cash - 1) * 100),
        "trades": int(len(trades_df)),
        "win_rate_pct": float((trades_df["net_pnl"] > 0).mean() * 100),
        "profit_factor": float(profit_factor) if profit_factor is not None else None,
        "expectancy": float(trades_df["net_pnl"].mean()),
        "max_drawdown_pct": max_drawdown_pct,
        "config": asdict(cfg),
    }
    return trades_df, metrics
