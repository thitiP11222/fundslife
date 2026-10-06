from __future__ import annotations

from typing import Any


BULLISH_OPTION_PATTERNS = {
    "Breakout",
    "Breakout Retest",
    "Ascending Triangle",
    "Bull Flag",
    "EMA20 Pullback",
    "Possible Double Bottom",
}


def build_long_call_plan(
    *,
    market_risk_on: bool,
    scanner_action: str | None,
    technical_setup: str,
    technical_score: int,
    pattern_state: dict[str, Any],
    current_price: float,
    entry_mode: str,
    entry_price: float,
    stop_price: float,
    target_1: float,
    target_2: float,
    rsi: float,
    volume_ratio: float,
) -> dict[str, Any]:
    """
    Build an underlying-driven long-call plan.

    This function does not price an option contract. It decides whether the
    underlying setup is strong enough to justify reviewing a long Call and
    provides contract-selection guidance.
    """

    pattern_name = str(pattern_state.get("name", "No Clear Pattern"))
    pattern_bias = str(pattern_state.get("bias", "neutral"))
    pattern_trigger = pattern_state.get("trigger")
    pattern_invalidation = pattern_state.get("invalidation")

    checks = {
        "market_risk_on": bool(market_risk_on),
        "momentum_candidate": scanner_action == "BUY_CANDIDATE",
        "technical_setup": technical_setup == "BUY_SETUP",
        "technical_score": technical_score >= 75,
        "bullish_pattern": (
            pattern_bias == "bullish"
            and pattern_name in BULLISH_OPTION_PATTERNS
        ),
        "rsi_ok": 50 <= rsi <= 70,
        "volume_ok": volume_ratio >= 1.0,
    }

    passed = sum(checks.values())

    if not checks["market_risk_on"]:
        status = "NO_CALL"
        message = "ตลาดไม่อยู่ใน Risk-On จึงไม่เน้น Long Call"
    elif not checks["momentum_candidate"]:
        status = "NO_CALL"
        message = "หุ้นยังไม่ใช่ Top momentum candidate"
    elif not checks["technical_setup"]:
        status = "WAIT"
        message = "Technical setup ยังไม่แข็งแรงพอ"
    elif not checks["bullish_pattern"]:
        status = "WAIT_PATTERN"
        message = "ยังไม่มี bullish pattern ที่ชัดพอสำหรับ Long Call"
    elif entry_mode == "Breakout" and current_price < entry_price:
        status = "WAIT_TRIGGER"
        message = (
            f"รอราคาหุ้นยืนยันเหนือ {entry_price:.2f} "
            "ก่อนพิจารณา Call"
        )
    elif entry_mode == "Pullback" and abs(current_price - entry_price) > max(
        current_price * 0.015,
        0.01,
    ):
        status = "WAIT_PULLBACK"
        message = (
            f"รอราคาหุ้นกลับเข้าใกล้โซน {entry_price:.2f} "
            "ก่อนพิจารณา Call"
        )
    else:
        status = "REVIEW_CALL"
        message = "Underlying ผ่านเงื่อนไขหลัก สามารถเปิด Option Chain เพื่อตรวจ Call ต่อได้"

    invalidation = (
        float(pattern_invalidation)
        if pattern_invalidation is not None
        else float(stop_price)
    )

    trigger = (
        float(pattern_trigger)
        if pattern_trigger is not None
        else float(entry_price)
    )

    return {
        "status": status,
        "message": message,
        "checks": checks,
        "checks_passed": passed,
        "checks_total": len(checks),
        "pattern": pattern_name,
        "pattern_bias": pattern_bias,
        "trigger": trigger,
        "invalidation": invalidation,
        "target_1": float(target_1),
        "target_2": float(target_2),
        "preferred_dte": "30-60 DTE",
        "preferred_strike": "ATM or slightly ITM",
        "avoid": "Far OTM / wide spread / low liquidity",
        "exit_logic": (
            "Exit or reduce if the underlying invalidates the setup; "
            "consider taking profit near underlying T1/T2 instead of "
            "holding blindly to expiration."
        ),
    }
