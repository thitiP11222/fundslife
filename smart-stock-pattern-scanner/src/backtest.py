"""Historical event-driven, long-only backtest for completed OHLCV bars.

Orders are placed after close at t and filled at next bar open t+1.
Stops/targets are checked conservatively: stop first if both touched.
All prices must use the same adjustment convention.
"""
from dataclasses import asdict
import numpy as np
import pandas as pd
from .patterns import scan

def signal_history(data: pd.DataFrame, rr: float=2.0, min_bars: int=65) -> pd.DataFrame:
    """Recompute on prefix only: no future bars at signal formation."""
    items=[]
    for end in range(min_bars,len(data)+1):
        _,signals=scan(data.iloc[:end],rr=rr)
        for s in signals:
            record=s.as_dict()
            record["bar_index"]=end-1
            items.append(record)
    return pd.DataFrame(items,columns=["pattern","direction","status","entry","stop","target",
                                       "score","explanation","detected_at","bar_index"])

def _metrics(equity: pd.DataFrame, trades: pd.DataFrame, initial: float, periods_per_year: int):
    if equity.empty:
        return {"total_return_pct":0., "max_drawdown_pct":0., "sharpe":0., "win_rate_pct":0.,
                "profit_factor":0., "trades":0}
    e=equity["equity"].astype(float)
    ret=e.pct_change().replace([np.inf,-np.inf],np.nan).dropna()
    std=ret.std(ddof=1)
    sharpe=float(ret.mean()/std*np.sqrt(periods_per_year)) if len(ret)>1 and std>0 else 0.
    max_dd=float((e/e.cummax()-1).min()*100)
    if trades.empty:
        wr=0.;pf=0.
    else:
        pl=trades["pnl"].astype(float)
        wr=float((pl>0).mean()*100)
        gross_win=float(pl[pl>0].sum())
        gross_loss=float(-pl[pl<0].sum())
        pf=round(gross_win/gross_loss,3) if gross_loss>0 else (float("inf") if gross_win>0 else 0.)
    return {"total_return_pct":round(float((e.iloc[-1]/initial-1)*100),2),
            "max_drawdown_pct":round(max_dd,2),"sharpe":round(sharpe,3),
            "win_rate_pct":round(wr,2),"profit_factor":pf,"trades":len(trades)}

def backtest(data: pd.DataFrame, history: pd.DataFrame | None=None,
             initial: float=1000., risk_percent: float=1., max_allocation_percent:float=60.,
             fee_bps: float=5., slippage_bps:float=5., rr:float=2.,
             start_index:int=65, end_index:int|None=None, periods_per_year:int=252):
    """A single active position, long-only, no leverage or pyramiding.

    Trades may originate only from signals at/after start_index, and are
    force-closed at the segment end. Split boundaries never carry positions.
    """
    if initial<=0 or not 0<risk_percent<=100 or not 0<max_allocation_percent<=100:
        raise ValueError("Invalid capital or risk settings")
    if fee_bps<0 or slippage_bps<0 or periods_per_year<=0:
        raise ValueError("Invalid execution settings")
    end=len(data) if end_index is None else min(end_index,len(data))
    if start_index<65 or end<=start_index:raise ValueError("Insufficient evaluation history")
    if history is None: history=signal_history(data.iloc[:end],rr=rr)
    grouped={}
    for rec in history.to_dict("records"):
        j=int(rec["bar_index"])
        if start_index<=j<end-1 and rec["direction"]=="LONG":
            grouped.setdefault(j,[]).append(rec)
    cash=float(initial)
    qty=0.;stop=target=entry=entry_cost=0.
    entry_bar=None
    trade_rows=[];equity_rows=[]
    slip=slippage_bps/10000.
    fee=fee_bps/10000.
    # Starting state before first evaluated bar; no historical position leakage.
    for i in range(start_index,end):
        bar=data.iloc[i]
        op,hi,lo,cl=(float(bar[k]) for k in ("Open","High","Low","Close"))
        if not all(np.isfinite(v) and v>0 for v in [op,hi,lo,cl]):raise ValueError("Invalid price bar")
        if qty>0:
            exit_price=None;reason=None
            if op<=stop:
                exit_price=op*(1-slip);reason="gap_stop"
            elif op>=target:
                exit_price=op*(1-slip);reason="gap_target"
            elif lo<=stop:
                exit_price=stop*(1-slip);reason="stop"
            elif hi>=target:
                exit_price=target*(1-slip);reason="target"
            if exit_price is not None:
                proceeds=qty*exit_price*(1-fee)
                cash+=proceeds
                trade_rows.append({"entry_date":str(data.index[entry_bar]),"exit_date":str(data.index[i]),
                    "entry":entry,"exit":exit_price,"shares":qty,"pnl":proceeds-entry_cost,
                    "return_pct":(proceeds/entry_cost-1)*100,"reason":reason})
                qty=0.
        # Use only previous close's signal; entry occurs at THIS bar open.
        # A position closed today is not re-opened until a later bar.
        if qty==0 and i>start_index and i-1 in grouped and not (trade_rows and trade_rows[-1]["exit_date"]==str(data.index[i])):
            s=sorted(grouped[i-1],key=lambda r:r["score"],reverse=True)[0]
            entry_price=op*(1+slip)
            stop_candidate=float(s["stop"])
            # Skip trades where gap invalidates protective stop or target.
            if 0<stop_candidate<entry_price:
                risk_per_share=entry_price-stop_candidate
                risk_budget=cash*risk_percent/100.
                allocation=cash*max_allocation_percent/100.
                quantity=min(risk_budget/risk_per_share,allocation/(entry_price*(1+fee)))
                cost=quantity*entry_price*(1+fee)
                if quantity>0 and cost<=cash:
                    qty=quantity;entry=entry_price;entry_cost=cost
                    stop=stop_candidate;target=entry_price+rr*risk_per_share
                    entry_bar=i;cash-=cost
                    # Conservative open entry: stop/target can be hit intrabar.
                    exit_price=None;reason=None
                    if lo<=stop:
                        exit_price=stop*(1-slip);reason="entry_bar_stop"
                    elif hi>=target:
                        exit_price=target*(1-slip);reason="entry_bar_target"
                    if exit_price is not None:
                        proceeds=qty*exit_price*(1-fee);cash+=proceeds
                        trade_rows.append({"entry_date":str(data.index[entry_bar]),"exit_date":str(data.index[i]),
                            "entry":entry,"exit":exit_price,"shares":qty,"pnl":proceeds-entry_cost,
                            "return_pct":(proceeds/entry_cost-1)*100,"reason":reason})
                        qty=0.
        # End-of-period liquidation, pessimistic slippage and fee.
        if qty>0 and i==end-1:
            price=cl*(1-slip);proceeds=qty*price*(1-fee);cash+=proceeds
            trade_rows.append({"entry_date":str(data.index[entry_bar]),"exit_date":str(data.index[i]),
                "entry":entry,"exit":price,"shares":qty,"pnl":proceeds-entry_cost,
                "return_pct":(proceeds/entry_cost-1)*100,"reason":"period_end"})
            qty=0.
        equity_rows.append({"date":data.index[i],"equity":cash+qty*cl})
    eq=pd.DataFrame(equity_rows).set_index("date")
    trades=pd.DataFrame(trade_rows,columns=["entry_date","exit_date","entry","exit","shares","pnl","return_pct","reason"])
    return eq,trades,_metrics(eq,trades,initial,periods_per_year)

def walk_forward(data:pd.DataFrame, history:pd.DataFrame|None=None, initial:float=1000.,
                 train_bars:int=126,test_bars:int=63,**kwargs):
    """Anchored chronological folds with fixed parameters (no model fitting or tuning).

    'Training' bars provide a causal warmup/context only, not optimized parameters.
    Each evaluation segment has reset capital; folds are NOT compounded.
    """
    if train_bars<65 or test_bars<2:raise ValueError("Invalid fold lengths")
    if history is None:history=signal_history(data,rr=kwargs.get("rr",2.))
    rows=[]
    for start in range(train_bars,len(data)-1,test_bars):
        end=min(start+test_bars,len(data))
        if end-start<2:break
        _,trades,m=backtest(data,history,initial=initial,start_index=start,end_index=end,**kwargs)
        rows.append({"train_start":str(data.index[0]),"train_end":str(data.index[start-1]),
                     "test_start":str(data.index[start]),"test_end":str(data.index[end-1]),**m})
    return pd.DataFrame(rows)
