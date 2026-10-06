from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class OptionAdvisorConfig:
    min_dte: int = 30
    max_dte: int = 60
    min_open_interest: int = 100
    min_volume: int = 10
    max_spread_pct: float = 15.0

    # Long-call strike window:
    # slightly ITM through approximately ATM / very slightly OTM.
    min_strike_to_spot: float = 0.95
    max_strike_to_spot: float = 1.01

    contract_multiplier: int = 100
    estimated_commission_per_contract: float = 0.55


def option_readiness(
    *,
    market_risk_on: bool,
    scanner_action: str | None,
    technical_setup: str,
    current_price: float,
    entry_mode: str,
    entry_price: float,
    atr_value: float,
) -> dict[str, Any]:
    reasons: list[str] = []

    if not market_risk_on:
        return {
            "status": "NO_OPTION",
            "ready": False,
            "reason": "Market regime is RISK-OFF.",
        }

    if scanner_action != "BUY_CANDIDATE":
        return {
            "status": "NO_OPTION",
            "ready": False,
            "reason": "Underlying is not a Top-3 momentum candidate.",
        }

    if technical_setup != "BUY_SETUP":
        return {
            "status": "WAIT",
            "ready": False,
            "reason": "Technical setup is not strong enough yet.",
        }

    if entry_mode == "Breakout":
        if current_price < entry_price:
            return {
                "status": "WAIT_BREAKOUT",
                "ready": False,
                "reason": (
                    f"Wait for the underlying to confirm above "
                    f"{entry_price:.2f}."
                ),
            }
        reasons.append("Breakout trigger confirmed.")

    else:
        tolerance = max(0.5 * atr_value, current_price * 0.005)
        if abs(current_price - entry_price) > tolerance:
            return {
                "status": "WAIT_PULLBACK",
                "ready": False,
                "reason": (
                    f"Current price is not near the pullback zone "
                    f"around {entry_price:.2f}."
                ),
            }
        reasons.append("Price is near the pullback entry zone.")

    return {
        "status": "REVIEW_CALL_CHAIN",
        "ready": True,
        "reason": " ".join(reasons),
    }


def expiration_candidates(
    expirations: list[str],
    *,
    today: date | None = None,
    cfg: OptionAdvisorConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or OptionAdvisorConfig()
    today = today or date.today()

    rows: list[dict[str, Any]] = []

    for value in expirations:
        expiry = datetime.strptime(value, "%Y-%m-%d").date()
        dte = (expiry - today).days

        if cfg.min_dte <= dte <= cfg.max_dte:
            rows.append(
                {
                    "expiration": value,
                    "dte": dte,
                }
            )

    return pd.DataFrame(rows)


def screen_long_calls(
    calls: pd.DataFrame,
    *,
    spot_price: float,
    account_equity: float,
    option_budget_pct: float,
    cfg: OptionAdvisorConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or OptionAdvisorConfig()

    if calls.empty:
        return pd.DataFrame()

    x = calls.copy()

    numeric = [
        "strike",
        "bid",
        "ask",
        "lastPrice",
        "volume",
        "openInterest",
        "impliedVolatility",
    ]

    for column in numeric:
        if column in x.columns:
            x[column] = pd.to_numeric(
                x[column],
                errors="coerce",
            )

    required = ["strike", "bid", "ask"]
    if any(column not in x.columns for column in required):
        return pd.DataFrame()

    x = x[
        x["strike"].between(
            spot_price * cfg.min_strike_to_spot,
            spot_price * cfg.max_strike_to_spot,
        )
    ].copy()

    x = x[
        (x["bid"] >= 0)
        & (x["ask"] > 0)
        & (x["ask"] >= x["bid"])
    ].copy()

    if x.empty:
        return x

    x["mid"] = (x["bid"] + x["ask"]) / 2.0
    x["spread"] = x["ask"] - x["bid"]
    x["spread_pct"] = (
        x["spread"] / x["mid"].replace(0, pd.NA) * 100
    )

    x["premium_cost"] = (
        x["ask"] * cfg.contract_multiplier
    )
    x["estimated_entry_cost"] = (
        x["premium_cost"]
        + cfg.estimated_commission_per_contract
    )

    budget = account_equity * option_budget_pct / 100.0
    x["within_option_budget"] = (
        x["estimated_entry_cost"] <= budget
    )

    if "openInterest" not in x.columns:
        x["openInterest"] = 0
    if "volume" not in x.columns:
        x["volume"] = 0
    if "impliedVolatility" not in x.columns:
        x["impliedVolatility"] = pd.NA

    x["openInterest"] = x["openInterest"].fillna(0)
    x["volume"] = x["volume"].fillna(0)

    x["liquidity_ok"] = (
        (x["spread_pct"] <= cfg.max_spread_pct)
        & (x["openInterest"] >= cfg.min_open_interest)
        & (x["volume"] >= cfg.min_volume)
    )

    x["moneyness"] = "ATM"
    x.loc[
        x["strike"] < spot_price * 0.995,
        "moneyness",
    ] = "ITM"
    x.loc[
        x["strike"] > spot_price * 1.005,
        "moneyness",
    ] = "OTM"

    x["distance_from_spot_pct"] = (
        (x["strike"] / spot_price - 1.0) * 100
    )

    # Prioritize liquidity first, then closeness to spot, then tighter spread.
    x["rank_score"] = (
        x["liquidity_ok"].astype(int) * 100
        - x["distance_from_spot_pct"].abs() * 2
        - x["spread_pct"].fillna(100)
    )

    return x.sort_values(
        ["liquidity_ok", "rank_score"],
        ascending=[False, False],
    ).reset_index(drop=True)


def summarize_option_fit(
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    if candidates.empty:
        return {
            "status": "NO_CONTRACT",
            "message": (
                "No ATM/slightly-ITM call contracts passed the basic "
                "market-data screen."
            ),
        }

    liquid = candidates[candidates["liquidity_ok"]]

    if liquid.empty:
        return {
            "status": "POOR_LIQUIDITY",
            "message": (
                "Contracts near the preferred strike area have weak "
                "liquidity or a wide bid/ask spread."
            ),
        }

    affordable = liquid[liquid["within_option_budget"]]

    if affordable.empty:
        cheapest = float(liquid["estimated_entry_cost"].min())
        return {
            "status": "OVER_BUDGET",
            "message": (
                "The liquid ATM/slightly-ITM contracts exceed the option "
                f"risk budget. Cheapest screened contract is about "
                f"USD {cheapest:.2f} for 1 contract."
            ),
        }

    top = affordable.iloc[0]

    return {
        "status": "REVIEW_CONTRACT",
        "message": (
            "A contract passed the screen. Verify the same contract and "
            "live quote in Dime before placing any order."
        ),
        "contractSymbol": top.get("contractSymbol"),
        "strike": float(top["strike"]),
        "ask": float(top["ask"]),
        "estimated_entry_cost": float(top["estimated_entry_cost"]),
        "spread_pct": float(top["spread_pct"]),
        "open_interest": int(top["openInterest"]),
        "volume": int(top["volume"]),
        "iv_pct": (
            float(top["impliedVolatility"]) * 100
            if pd.notna(top["impliedVolatility"])
            else None
        ),
        "moneyness": str(top["moneyness"]),
    }
