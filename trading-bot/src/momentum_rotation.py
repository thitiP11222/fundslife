from __future__ import annotations

from typing import Any

import pandas as pd


def _pct_change(series: pd.Series, periods: int) -> pd.Series:
    return series / series.shift(periods) - 1


def backtest_momentum_rotation(
    stock_data: dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    *,
    starting_cash: float = 30.0,
    rebalance_every: int = 5,
    momentum_fast: int = 63,
    momentum_slow: int = 126,
    benchmark_ma: int = 200,
    max_position_value_pct: float = 0.95,
    commission_pct: float = 0.0005,
    slippage_pct: float = 0.0002,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Long-only Top-1 momentum rotation.

    Rules:
    - Daily data.
    - Risk-on only when benchmark close > benchmark 200-day MA.
    - Every rebalance period, rank stocks by blended 3M/6M momentum.
    - Hold only the strongest stock if its momentum is positive.
    - Otherwise remain in cash.
    """

    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be positive.")

    frames: dict[str, pd.DataFrame] = {}

    for symbol, df in stock_data.items():
        x = df.copy().sort_index()
        x["MOM_FAST"] = _pct_change(x["Close"], momentum_fast)
        x["MOM_SLOW"] = _pct_change(x["Close"], momentum_slow)
        x["MOM_SCORE"] = 0.6 * x["MOM_FAST"] + 0.4 * x["MOM_SLOW"]
        frames[symbol] = x

    benchmark = benchmark_df.copy().sort_index()
    benchmark["BENCH_MA"] = (
        benchmark["Close"].rolling(benchmark_ma, min_periods=benchmark_ma).mean()
    )

    common_index = benchmark.index
    for df in frames.values():
        common_index = common_index.intersection(df.index)

    common_index = common_index.sort_values()

    cash = starting_cash
    current_symbol: str | None = None
    shares = 0.0
    entry_price = 0.0
    entry_time = None
    entry_commission = 0.0

    trades: list[dict[str, Any]] = []
    equity_curve = [starting_cash]

    for i, ts in enumerate(common_index):
        bench_row = benchmark.loc[ts]

        if i == 0:
            continue

        should_rebalance = i % rebalance_every == 0

        if should_rebalance:
            risk_on = (
                pd.notna(bench_row["BENCH_MA"])
                and float(bench_row["Close"]) > float(bench_row["BENCH_MA"])
            )

            selected_symbol = None

            if risk_on:
                ranked = []
                for symbol, df in frames.items():
                    row = df.loc[ts]
                    score = row["MOM_SCORE"]
                    if pd.notna(score) and float(score) > 0:
                        ranked.append((float(score), symbol))

                if ranked:
                    ranked.sort(reverse=True)
                    selected_symbol = ranked[0][1]

            if selected_symbol != current_symbol:
                # Exit existing position at current close.
                if current_symbol is not None:
                    raw_exit = float(frames[current_symbol].loc[ts, "Close"])
                    exit_price = raw_exit * (1 - slippage_pct)
                    exit_commission = exit_price * shares * commission_pct
                    gross_pnl = (exit_price - entry_price) * shares
                    net_pnl = gross_pnl - entry_commission - exit_commission

                    cash += entry_price * shares + gross_pnl - exit_commission

                    trades.append(
                        {
                            "symbol": current_symbol,
                            "entry_time": entry_time,
                            "exit_time": ts,
                            "shares": shares,
                            "entry": entry_price,
                            "exit": exit_price,
                            "net_pnl": net_pnl,
                            "exit_reason": "ROTATE_OR_CASH",
                        }
                    )

                    current_symbol = None
                    shares = 0.0
                    entry_price = 0.0
                    entry_time = None
                    entry_commission = 0.0

                # Enter new selection.
                if selected_symbol is not None:
                    raw_entry = float(frames[selected_symbol].loc[ts, "Close"])
                    entry_price = raw_entry * (1 + slippage_pct)

                    budget = cash * max_position_value_pct
                    shares = budget / entry_price

                    if shares > 0:
                        entry_commission = (
                            entry_price * shares * commission_pct
                        )
                        cash -= entry_price * shares + entry_commission
                        current_symbol = selected_symbol
                        entry_time = ts

        if current_symbol is None:
            marked_equity = cash
        else:
            close = float(frames[current_symbol].loc[ts, "Close"])
            marked_equity = cash + close * shares

        equity_curve.append(marked_equity)

    # Liquidate at end.
    if current_symbol is not None and len(common_index) > 0:
        ts = common_index[-1]
        raw_exit = float(frames[current_symbol].loc[ts, "Close"])
        exit_price = raw_exit * (1 - slippage_pct)
        exit_commission = exit_price * shares * commission_pct
        gross_pnl = (exit_price - entry_price) * shares
        net_pnl = gross_pnl - entry_commission - exit_commission

        cash += entry_price * shares + gross_pnl - exit_commission

        trades.append(
            {
                "symbol": current_symbol,
                "entry_time": entry_time,
                "exit_time": ts,
                "shares": shares,
                "entry": entry_price,
                "exit": exit_price,
                "net_pnl": net_pnl,
                "exit_reason": "END_OF_DATA",
            }
        )

        if equity_curve:
            equity_curve[-1] = cash

    trades_df = pd.DataFrame(trades)

    equity = pd.Series(equity_curve, dtype=float)
    drawdown = equity / equity.cummax() - 1
    max_drawdown_pct = float(drawdown.min() * 100)

    ending_equity = float(cash)

    if trades_df.empty:
        return trades_df, {
            "starting_cash": starting_cash,
            "ending_equity": ending_equity,
            "total_return_pct": (ending_equity / starting_cash - 1) * 100,
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "expectancy": 0.0,
            "max_drawdown_pct": max_drawdown_pct,
        }

    wins = trades_df.loc[trades_df["net_pnl"] > 0, "net_pnl"]
    losses = trades_df.loc[trades_df["net_pnl"] < 0, "net_pnl"]

    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))

    return trades_df, {
        "starting_cash": starting_cash,
        "ending_equity": ending_equity,
        "total_return_pct": (ending_equity / starting_cash - 1) * 100,
        "trades": int(len(trades_df)),
        "win_rate_pct": float((trades_df["net_pnl"] > 0).mean() * 100),
        "profit_factor": (
            gross_profit / gross_loss if gross_loss > 0 else None
        ),
        "expectancy": float(trades_df["net_pnl"].mean()),
        "max_drawdown_pct": max_drawdown_pct,
    }
