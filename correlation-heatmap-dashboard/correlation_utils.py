from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd


def clean_tickers(raw_tickers: str | Iterable[str], max_assets: int = 20) -> list[str]:
    """Normalize ticker input and remove duplicates while preserving order."""
    if isinstance(raw_tickers, str):
        tokens = raw_tickers.replace("\n", ",").split(",")
    else:
        tokens = list(raw_tickers)

    cleaned: list[str] = []
    seen: set[str] = set()

    for token in tokens:
        ticker = str(token).strip().upper()
        if ticker and ticker not in seen:
            cleaned.append(ticker)
            seen.add(ticker)

    if len(cleaned) < 2:
        raise ValueError("กรุณาระบุอย่างน้อย 2 ticker")
    if len(cleaned) > max_assets:
        raise ValueError(f"รองรับได้สูงสุด {max_assets} ticker ต่อครั้ง")

    return cleaned


def compute_returns(
    prices: pd.DataFrame,
    return_type: str = "simple",
) -> pd.DataFrame:
    """Convert aligned close prices to simple or logarithmic returns."""
    if prices.empty:
        raise ValueError("Price DataFrame ว่าง")

    numeric_prices = prices.apply(pd.to_numeric, errors="coerce").sort_index()

    if return_type == "simple":
        returns = numeric_prices.pct_change(fill_method=None)
    elif return_type == "log":
        if (numeric_prices <= 0).any().any():
            raise ValueError("Log return ใช้ไม่ได้เมื่อมีราคาน้อยกว่าหรือเท่ากับศูนย์")
        returns = np.log(numeric_prices / numeric_prices.shift(1))
    else:
        raise ValueError("return_type ต้องเป็น 'simple' หรือ 'log'")

    return returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def compute_pairwise_observations(returns: pd.DataFrame) -> pd.DataFrame:
    """Count overlapping non-null observations for every asset pair."""
    availability = returns.notna().astype(int)
    counts = availability.T.dot(availability)
    counts.index.name = "Ticker"
    counts.columns.name = "Ticker"
    return counts


def calculate_correlation(
    returns: pd.DataFrame,
    method: str = "pearson",
    min_observations: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate a return-correlation matrix and its pairwise sample counts."""
    if method not in {"pearson", "spearman"}:
        raise ValueError("method ต้องเป็น 'pearson' หรือ 'spearman'")
    if min_observations < 2:
        raise ValueError("min_observations ต้องไม่น้อยกว่า 2")

    counts = compute_pairwise_observations(returns)
    correlation = returns.corr(method=method, min_periods=min_observations)

    # Explicitly mask coefficients with insufficient overlapping observations.
    correlation = correlation.mask(counts < min_observations)
    return correlation, counts


def classify_correlation(value: float) -> str:
    """Return an interpretation label based on correlation magnitude/direction."""
    if pd.isna(value):
        return "ข้อมูลไม่พอ"
    if value >= 0.80:
        return "บวกสูงมาก"
    if value >= 0.60:
        return "บวกสูง"
    if value >= 0.30:
        return "บวกปานกลาง"
    if value > -0.30:
        return "ต่ำ/ใกล้ศูนย์"
    if value > -0.60:
        return "ลบปานกลาง"
    if value > -0.80:
        return "ลบสูง"
    return "ลบสูงมาก"


def build_pair_table(
    correlation: pd.DataFrame,
    observation_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Convert a symmetric matrix into one row per unique asset pair."""
    rows: list[dict[str, object]] = []

    common_assets = [
        asset
        for asset in correlation.columns
        if asset in correlation.index
        and asset in observation_counts.index
        and asset in observation_counts.columns
    ]

    for asset_1, asset_2 in combinations(common_assets, 2):
        value = correlation.loc[asset_1, asset_2]
        rows.append(
            {
                "Asset 1": asset_1,
                "Asset 2": asset_2,
                "Correlation": float(value) if pd.notna(value) else np.nan,
                "Absolute Correlation": (
                    float(abs(value)) if pd.notna(value) else np.nan
                ),
                "Overlapping Observations": int(
                    observation_counts.loc[asset_1, asset_2]
                ),
                "Interpretation": classify_correlation(value),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Asset 1",
                "Asset 2",
                "Correlation",
                "Absolute Correlation",
                "Overlapping Observations",
                "Interpretation",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        ["Correlation", "Asset 1", "Asset 2"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def average_correlation_by_asset(correlation: pd.DataFrame) -> pd.DataFrame:
    """Calculate each asset's average correlation with all other assets."""
    rows: list[dict[str, object]] = []

    for asset in correlation.columns:
        peers = correlation.loc[asset].drop(labels=[asset], errors="ignore").dropna()
        rows.append(
            {
                "Ticker": asset,
                "Average Correlation": peers.mean() if not peers.empty else np.nan,
                "Average Absolute Correlation": (
                    peers.abs().mean() if not peers.empty else np.nan
                ),
                "Valid Pairs": int(peers.count()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "Average Correlation",
        ascending=True,
        na_position="last",
    ).reset_index(drop=True)


def summarize_correlations(pair_table: pd.DataFrame) -> dict[str, object]:
    """Return descriptive portfolio-level correlation statistics."""
    valid = pair_table.dropna(subset=["Correlation"])

    if valid.empty:
        return {
            "average_pair_correlation": np.nan,
            "median_pair_correlation": np.nan,
            "average_absolute_correlation": np.nan,
            "share_below_0_50": np.nan,
            "highest_pair": None,
            "lowest_pair": None,
            "valid_pairs": 0,
        }

    highest = valid.loc[valid["Correlation"].idxmax()]
    lowest = valid.loc[valid["Correlation"].idxmin()]

    return {
        "average_pair_correlation": float(valid["Correlation"].mean()),
        "median_pair_correlation": float(valid["Correlation"].median()),
        "average_absolute_correlation": float(
            valid["Absolute Correlation"].mean()
        ),
        # A transparent heuristic, not a portfolio-risk metric.
        "share_below_0_50": float((valid["Correlation"] < 0.50).mean()),
        "highest_pair": (
            str(highest["Asset 1"]),
            str(highest["Asset 2"]),
            float(highest["Correlation"]),
        ),
        "lowest_pair": (
            str(lowest["Asset 1"]),
            str(lowest["Asset 2"]),
            float(lowest["Correlation"]),
        ),
        "valid_pairs": int(len(valid)),
    }


def rolling_pair_correlation(
    returns: pd.DataFrame,
    asset_1: str,
    asset_2: str,
    window: int = 60,
    min_period_ratio: float = 0.8,
) -> pd.Series:
    """Calculate rolling correlation between two return series."""
    if asset_1 not in returns or asset_2 not in returns:
        raise KeyError("ไม่พบ ticker ที่เลือกใน Return DataFrame")
    if window < 5:
        raise ValueError("Rolling window ต้องไม่น้อยกว่า 5")

    min_periods = max(2, int(window * min_period_ratio))
    pair = returns[[asset_1, asset_2]].dropna()
    result = pair[asset_1].rolling(
        window=window,
        min_periods=min_periods,
    ).corr(pair[asset_2])
    result.name = f"{asset_1}-{asset_2} Rolling Correlation"
    return result
