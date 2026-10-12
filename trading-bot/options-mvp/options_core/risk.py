"""Contract size / risk gates for long single-leg US equity options only."""
from __future__ import annotations
import math

def option_risk_check(*, equity_thb:float, risk_pct:float, fx_thb_per_usd:float,
                      ask_usd:float, bid_usd:float, multiplier:int=100,
                      commission_usd:float=0.55, other_fees_usd:float=0.05,
                      stop_loss_fraction:float=1.0)->dict:
    """Fees modeled as round-trip per-side commission+fees; use a conservative manual estimate.
    stop_loss_fraction=1 assumes total premium loss. Stops may slip or fail.
    """
    vals=[equity_thb,risk_pct,fx_thb_per_usd,ask_usd,bid_usd,commission_usd,other_fees_usd,stop_loss_fraction]
    if not all(math.isfinite(v) for v in vals) or equity_thb<=0 or not (0<risk_pct<=100) or fx_thb_per_usd<=0 or ask_usd<=0 or bid_usd<0 or bid_usd>ask_usd or multiplier<=0 or commission_usd<0 or other_fees_usd<0 or not (0<stop_loss_fraction<=1):
        raise ValueError("Invalid contract inputs")
    budget_thb=equity_thb*risk_pct/100
    one_way_fee=commission_usd+other_fees_usd
    premium_usd=ask_usd*multiplier
    cost_thb=(premium_usd+one_way_fee)*fx_thb_per_usd
    max_loss_thb=(premium_usd+one_way_fee)*fx_thb_per_usd
    spread_usd=(ask_usd-bid_usd)*multiplier
    planned_loss_thb=(ask_usd*multiplier*stop_loss_fraction+2*one_way_fee)*fx_thb_per_usd
    # Conservative hard cap: loss may be entire paid premium.
    quantity=math.floor(min(equity_thb/cost_thb,budget_thb/max_loss_thb))
    return {"approved":quantity>=1,"contracts":max(0,quantity),"risk_budget_thb":budget_thb,
            "cost_thb":cost_thb,"max_loss_thb":max_loss_thb,
            "planned_loss_thb":planned_loss_thb,
            "spread_pct":100*(ask_usd-bid_usd)/ask_usd,
            "reason":"PASS" if quantity>=1 else "NO_TRADE: one contract can exceed capital or 1% worst-case loss budget",
            "assumption":"Long option only; multiplier as entered; excludes changing broker charges, FX spread and exercise risk"}
