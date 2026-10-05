from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from config import StrategyConfig
from src.context import add_market_context
from src.indicators import add_indicators
from src.risk import position_size_from_risk


def _prepare_symbol(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    cfg: StrategyConfig,
    *,
    rs_lookback: int,
) -> pd.DataFrame:
    out = add_market_context(
        df,
        benchmark_df,
        rs_lookback=rs_lookback,
    )
    out = add_indicators(
        out,
        ema_fast=cfg.ema_fast,
        ema_slow=cfg.ema_slow,
        rsi_period=cfg.rsi_period,
        atr_period=cfg.atr_period,
        breakout_period=cfg.breakout_period,
        volume_ma_period=cfg.volume_ma_period,
    )

    reclaimed = (
        (out["Close"].shift(1) <= out["EMA_FAST"].shift(1))
        & (out["Close"] > out["EMA_FAST"])
    )
    bullish = out["Close"] > out["Open"]
    volume_ok = out["Volume"] >= out["VOLUME_MA"] * cfg.volume_ratio_min
    market_ok = (
        (out["BENCH_CLOSE"] > out["BENCH_EMA_SLOW"])
        & (out["BENCH_EMA_FAST"] > out["BENCH_EMA_SLOW"])
    )

    out["SETUP_OK"] = (
        (out["EMA_FAST"] > out["EMA_SLOW"])
        & reclaimed
        & bullish
        & out["RSI"].between(cfg.rsi_min, cfg.rsi_max, inclusive="both")
        & volume_ok
        & market_ok
        & (out["RS_VS_BENCH"] > 0)
    )

    return out


def backtest_top1_rotation(
    stock_data: dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    cfg: StrategyConfig,
    *,
    rs_lookback: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Small-account portfolio backtest.

    At most one position is open. On each bar, among symbols with a valid
    pullback-reclaim setup, choose the symbol with the highest RS_VS_BENCH.

    Entry uses next-bar open. Exit uses a rising EMA20 - ATR trailing stop.
    """
    prepared: dict[str, pd.DataFrame] = {}
    for symbol, df in stock_data.items():
        prepared[symbol] = _prepare_symbol(
            df,
            benchmark_df,
            replace(cfg, symbol=symbol),
            rs_lookback=rs_lookback,
        )

    common_index = None
    for df in prepared.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)

    if common_index is None or len(common_index) < 3:
        return pd.DataFrame(), {
            "starting_cash": cfg.starting_cash,
            "ending_equity": cfg.starting_cash,
            "total_return_pct": 0.0,
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "expectancy": 0.0,
            "max_drawdown_pct": 0.0,
        }

    common_index = common_index.sort_values()

    cash = cfg.starting_cash
    position = None
    pending_symbol = None
    trades: list[dict[str, Any]] = []
    equity_curve = [cash]

    for i in range(1, len(common_index)):
        ts = common_index[i]
        prev_ts = common_index[i - 1]

        # Execute a selection made on the previous candle at current open.
        if position is None and pending_symbol is not None:
            row = prepared[pending_symbol].loc[ts]
            prev = prepared[pending_symbol].loc[prev_ts]

            if pd.notna(prev["ATR"]):
                entry = float(row["Open"]) * (1 + cfg.slippage_pct)
                stop_distance = float(prev["ATR"]) * cfg.atr_stop_multiplier
                stop = entry - stop_distance

                shares = position_size_from_risk(
                    equity=cash,
                    risk_fraction=cfg.risk_per_trade,
                    entry_price=entry,
                    stop_price=stop,
                    max_position_value_pct=cfg.max_position_value_pct,
                    allow_fractional=cfg.allow_fractional_shares,
                    share_step=cfg.share_step,
                )

                if shares > 0:
                    position = {
                        "symbol": pending_symbol,
                        "entry_time": ts,
                        "entry": entry,
                        "stop": stop,
                        "initial_stop": stop,
                        "shares": shares,
                        "entry_commission": entry * shares * cfg.commission_pct,
                    }

            pending_symbol = None

        # Manage open position.
        if position is not None:
            symbol = position["symbol"]
            row = prepared[symbol].loc[ts]
            prev = prepared[symbol].loc[prev_ts]

            if pd.notna(prev["EMA_FAST"]) and pd.notna(prev["ATR"]):
                candidate_stop = (
                    float(prev["EMA_FAST"])
                    - float(prev["ATR"]) * cfg.trailing_atr_multiplier
                )
                position["stop"] = max(position["stop"], candidate_stop)

            open_price = float(row["Open"])
            low = float(row["Low"])
            raw_exit = None
            exit_reason = None

            if open_price <= position["stop"]:
                raw_exit = open_price
                exit_reason = "TRAILING_GAP"
            elif low <= position["stop"]:
                raw_exit = position["stop"]
                exit_reason = "TRAILING_STOP"

            if raw_exit is not None:
                exit_price = raw_exit * (1 - cfg.slippage_pct)
                shares = float(position["shares"])
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
                        "symbol": symbol,
                        "entry_time": position["entry_time"],
                        "exit_time": ts,
                        "shares": shares,
                        "entry": position["entry"],
                        "exit": exit_price,
                        "initial_stop": position["initial_stop"],
                        "final_stop": position["stop"],
                        "exit_reason": exit_reason,
                        "net_pnl": net_pnl,
                        "equity_after": cash,
                    }
                )
                position = None

        # Select only one best setup after current candle closes.
        if position is None:
            candidates = []
            for symbol, df in prepared.items():
                row = df.loc[ts]
                if bool(row["SETUP_OK"]) and pd.notna(row["RS_VS_BENCH"]):
                    candidates.append((float(row["RS_VS_BENCH"]), symbol))

            if candidates:
                candidates.sort(reverse=True)
                pending_symbol = candidates[0][1]

        if position is None:
            marked_equity = cash
        else:
            row = prepared[position["symbol"]].loc[ts]
            shares = float(position["shares"])
            unrealized = (float(row["Close"]) - position["entry"]) * shares
            marked_equity = cash + unrealized - position["entry_commission"]

        equity_curve.append(marked_equity)

    # Close remaining position at end of data.
    if position is not None:
        ts = common_index[-1]
        row = prepared[position["symbol"]].loc[ts]
        exit_price = float(row["Close"]) * (1 - cfg.slippage_pct)
        shares = float(position["shares"])
        gross_pnl = (exit_price - position["entry"]) * shares
        exit_commission = exit_price * shares * cfg.commission_pct
        net_pnl = gross_pnl - position["entry_commission"] - exit_commission
        cash += net_pnl

        trades.append(
            {
                "symbol": position["symbol"],
                "entry_time": position["entry_time"],
                "exit_time": ts,
                "shares": shares,
                "entry": position["entry"],
                "exit": exit_price,
                "initial_stop": position["initial_stop"],
                "final_stop": position["stop"],
                "exit_reason": "END_OF_DATA",
                "net_pnl": net_pnl,
                "equity_after": cash,
            }
        )
        equity_curve[-1] = cash

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
        }
        return trades_df, metrics

    wins = trades_df.loc[trades_df["net_pnl"] > 0, "net_pnl"]
    losses = trades_df.loc[trades_df["net_pnl"] < 0, "net_pnl"]
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))

    metrics = {
        "starting_cash": cfg.starting_cash,
        "ending_equity": float(cash),
        "total_return_pct": float((cash / cfg.starting_cash - 1) * 100),
        "trades": int(len(trades_df)),
        "win_rate_pct": float((trades_df["net_pnl"] > 0).mean() * 100),
        "profit_factor": (
            gross_profit / gross_loss if gross_loss > 0 else None
        ),
        "expectancy": float(trades_df["net_pnl"].mean()),
        "max_drawdown_pct": max_drawdown_pct,
    }

    return trades_df, metrics
