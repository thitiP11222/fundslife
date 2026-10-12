"""Event-driven underlying OHLC backtest; NOT historical options P&L."""
from __future__ import annotations
import math
import pandas as pd

def backtest_atr(frame: pd.DataFrame, *, start: int=60, end: int|None=None,
                 hold_days: int=10, stop_atr: float=1.5, target_r: float=2.0,
                 trailing_atr: float=0.0, fee_bps: float=10.0, slippage_bps: float=5.0):
    if hold_days<1 or stop_atr<=0 or target_r<=0 or trailing_atr<0 or fee_bps<0 or slippage_bps<0:
        raise ValueError("Invalid backtest parameters")
    needed={"Open","High","Low","Close","atr14","signal"}
    if not needed.issubset(frame.columns): raise ValueError("Missing OHLC, ATR or signal columns")
    end=min(len(frame),end if end is not None else len(frame))
    rows=[]; i=max(1,start)
    while i+1<end:
        side=int(frame["signal"].iloc[i])
        atr=float(frame["atr14"].iloc[i])
        if side==0 or not math.isfinite(atr) or atr<=0:
            i+=1; continue
        entry_idx=i+1
        entry_open=float(frame["Open"].iloc[entry_idx])
        if not math.isfinite(entry_open) or entry_open<=0:
            i+=1; continue
        slip=slippage_bps/10000
        entry=entry_open*(1+side*slip)
        distance=stop_atr*atr
        stop=entry-side*distance
        target=entry+side*target_r*distance
        extreme=entry
        exit_px=None; reason=None; exit_idx=None
        last_idx=min(end-1,entry_idx+hold_days-1)
        for j in range(entry_idx,last_idx+1):
            bar=frame.iloc[j]
            op,hi,lo,close=(float(bar[k]) for k in ["Open","High","Low","Close"])
            if not all(math.isfinite(v) for v in (op,hi,lo,close)) or min(op,hi,lo,close)<=0:
                raise ValueError("Invalid OHLC bar")
            # Before intraday range: gap through stop/target fills at opening price.
            if (side==1 and op<=stop) or (side==-1 and op>=stop):
                exit_px=op; reason="stop_gap"
            elif (side==1 and op>=target) or (side==-1 and op<=target):
                exit_px=op; reason="target_gap"
            else:
                stop_hit=(lo<=stop) if side==1 else (hi>=stop)
                target_hit=(hi>=target) if side==1 else (lo<=target)
                if stop_hit:  # conservative if both touched in same bar
                    exit_px=stop; reason="stop" if not target_hit else "both_hit_stop_first"
                elif target_hit:
                    exit_px=target; reason="target"
            if exit_px is not None:
                exit_idx=j; break
            # Update trailing stop only after the daily bar: activates next bar.
            if trailing_atr>0:
                extreme=max(extreme,hi) if side==1 else min(extreme,lo)
                candidate=extreme-side*trailing_atr*atr
                stop=max(stop,candidate) if side==1 else min(stop,candidate)
            if j==last_idx:
                exit_px=close; reason="time_exit";exit_idx=j
        if exit_idx is None: break
        # adverse execution slippage and round-trip commissions
        exit_fill=exit_px*(1-side*slip)
        net=side*(exit_fill/entry-1)-2*fee_bps/10000
        rows.append({"signal_date":str(frame.index[i].date()),"entry_date":str(frame.index[entry_idx].date()),
                     "exit_date":str(frame.index[exit_idx].date()),"side":"CALL_BIAS" if side==1 else "PUT_BIAS",
                     "entry":entry,"exit":exit_fill,"reason":reason,"net_return":net})
        i=exit_idx+1
    trades=pd.DataFrame(rows)
    if trades.empty:
        return trades,{"trades":0,"win_rate":0.,"profit_factor":None,"expectancy":0.,"max_drawdown":0.,"compounded_return":0.}
    returns=trades["net_return"]
    wins=returns[returns>0].sum();loss=-returns[returns<0].sum()
    equity=(1+returns.clip(lower=-1)).cumprod()
    peak=pd.concat([pd.Series([1.]),equity.reset_index(drop=True)],ignore_index=True).cummax().iloc[1:].reset_index(drop=True)
    stats={"trades":len(trades),"win_rate":float((returns>0).mean()),
           "profit_factor":float(wins/loss) if loss>0 else None,
           "expectancy":float(returns.mean()),"max_drawdown":float((equity.reset_index(drop=True)/peak-1).min()),
           "compounded_return":float(equity.iloc[-1]-1)}
    return trades,stats
