from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_oos_events(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize already non-overlapping momentum events by calendar year and
    with leave-one-symbol-out robustness.

    This does not optimize parameters. It checks whether the observed edge is
    stable across time and whether it survives removing any one stock.
    """
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()

    x = events.copy()
    x["date"] = pd.to_datetime(x["date"])
    x["year"] = x["date"].dt.year

    yearly_rows: list[dict[str, Any]] = []

    for year, subset in x.groupby("year"):
        yearly_rows.append(
            {
                "year": int(year),
                "events": int(len(subset)),
                "avg_return_pct": float(
                    subset["stock_return_pct"].mean()
                ),
                "median_return_pct": float(
                    subset["stock_return_pct"].median()
                ),
                "win_rate_pct": float(
                    (subset["stock_return_pct"] > 0).mean() * 100
                ),
                "avg_excess_vs_qqq_pct": float(
                    subset["excess_return_pct"].mean()
                ),
                "median_excess_vs_qqq_pct": float(
                    subset["excess_return_pct"].median()
                ),
                "beat_qqq_rate_pct": float(
                    (subset["excess_return_pct"] > 0).mean() * 100
                ),
            }
        )

    yearly = pd.DataFrame(yearly_rows).sort_values("year")

    loo_rows: list[dict[str, Any]] = []

    symbols = sorted(x["symbol"].unique())

    for excluded in symbols:
        subset = x[x["symbol"] != excluded]

        loo_rows.append(
            {
                "excluded_symbol": excluded,
                "events": int(len(subset)),
                "avg_return_pct": float(
                    subset["stock_return_pct"].mean()
                ),
                "median_return_pct": float(
                    subset["stock_return_pct"].median()
                ),
                "win_rate_pct": float(
                    (subset["stock_return_pct"] > 0).mean() * 100
                ),
                "avg_excess_vs_qqq_pct": float(
                    subset["excess_return_pct"].mean()
                ),
                "median_excess_vs_qqq_pct": float(
                    subset["excess_return_pct"].median()
                ),
                "beat_qqq_rate_pct": float(
                    (subset["excess_return_pct"] > 0).mean() * 100
                ),
            }
        )

    leave_one_out = pd.DataFrame(loo_rows)

    return yearly, leave_one_out
