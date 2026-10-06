from __future__ import annotations

import math


def _round_down_to_step(value: float, step: float) -> float:
    if step <= 0:
        raise ValueError("share_step must be positive.")
    return math.floor(value / step) * step


def position_size_from_risk(
    *,
    equity: float,
    risk_fraction: float,
    entry_price: float,
    stop_price: float,
    max_position_value_pct: float,
    allow_fractional: bool = False,
    share_step: float = 1.0,
) -> float:
    if equity <= 0:
        return 0.0
    if not 0 < risk_fraction <= 1:
        raise ValueError("risk_fraction must be in (0, 1].")
    if not 0 < max_position_value_pct <= 1:
        raise ValueError("max_position_value_pct must be in (0, 1].")
    if entry_price <= 0:
        raise ValueError("entry_price must be positive.")

    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return 0.0

    risk_budget = equity * risk_fraction
    shares_by_risk = risk_budget / stop_distance

    max_position_value = equity * max_position_value_pct
    shares_by_value = max_position_value / entry_price

    shares = min(shares_by_risk, shares_by_value)

    if allow_fractional:
        rounded = _round_down_to_step(shares, share_step)
        return max(0.0, round(rounded, 6))

    return float(max(0, math.floor(shares)))
