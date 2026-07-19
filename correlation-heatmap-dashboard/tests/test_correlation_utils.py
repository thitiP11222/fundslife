import numpy as np
import pandas as pd
import pytest

from correlation_utils import (
    build_pair_table,
    calculate_correlation,
    clean_tickers,
    compute_returns,
    rolling_pair_correlation,
    summarize_correlations,
)


def test_clean_tickers_removes_duplicates_and_normalizes():
    result = clean_tickers(" aapl, MSFT\nAAPL, voo ")
    assert result == ["AAPL", "MSFT", "VOO"]


def test_clean_tickers_requires_two_assets():
    with pytest.raises(ValueError):
        clean_tickers("AAPL")


def test_compute_simple_returns():
    prices = pd.DataFrame(
        {
            "A": [100.0, 110.0, 121.0],
            "B": [50.0, 45.0, 49.5],
        },
        index=pd.date_range("2026-01-01", periods=3),
    )
    returns = compute_returns(prices, "simple")
    assert returns.loc[returns.index[0], "A"] == pytest.approx(0.10)
    assert returns.loc[returns.index[1], "A"] == pytest.approx(0.10)
    assert returns.loc[returns.index[0], "B"] == pytest.approx(-0.10)


def test_calculate_correlation_and_pair_table():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.03, 0.00],
            "B": [0.02, 0.04, -0.02, 0.06, 0.00],
            "C": [-0.01, -0.02, 0.01, -0.03, 0.00],
        }
    )
    correlation, counts = calculate_correlation(
        returns,
        method="pearson",
        min_observations=5,
    )
    pairs = build_pair_table(correlation, counts)

    ab = pairs.query("`Asset 1` == 'A' and `Asset 2` == 'B'").iloc[0]
    ac = pairs.query("`Asset 1` == 'A' and `Asset 2` == 'C'").iloc[0]

    assert ab["Correlation"] == pytest.approx(1.0)
    assert ac["Correlation"] == pytest.approx(-1.0)
    assert ab["Overlapping Observations"] == 5


def test_minimum_observations_masks_small_samples():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, np.nan],
            "B": [0.02, 0.04, 0.01],
        }
    )
    correlation, _ = calculate_correlation(
        returns,
        min_observations=3,
    )
    assert np.isnan(correlation.loc["A", "B"])


def test_summary_finds_highest_and_lowest_pairs():
    pair_table = pd.DataFrame(
        [
            {
                "Asset 1": "A",
                "Asset 2": "B",
                "Correlation": 0.9,
                "Absolute Correlation": 0.9,
            },
            {
                "Asset 1": "A",
                "Asset 2": "C",
                "Correlation": -0.2,
                "Absolute Correlation": 0.2,
            },
        ]
    )
    summary = summarize_correlations(pair_table)
    assert summary["highest_pair"] == ("A", "B", 0.9)
    assert summary["lowest_pair"] == ("A", "C", -0.2)
    assert summary["valid_pairs"] == 2


def test_rolling_pair_correlation_has_expected_name():
    returns = pd.DataFrame(
        {
            "A": np.linspace(-0.02, 0.02, 20),
            "B": np.linspace(-0.01, 0.03, 20),
        },
        index=pd.date_range("2026-01-01", periods=20),
    )
    result = rolling_pair_correlation(returns, "A", "B", window=5)
    assert result.name == "A-B Rolling Correlation"
    assert result.dropna().iloc[-1] == pytest.approx(1.0)
