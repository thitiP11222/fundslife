"""Standalone market scan, separate from Streamlit. Run using Windows Task Scheduler."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from options_core.market import history
from options_core.strategy_v2 import strategy_v2
from options_core.alerts import dispatch_telegram

def run_scan(symbols, send=False, state_path=Path("alert_state.json")):
    if not symbols: return []
    spy=history("SPY")
    try: state=json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError): state={}
    results=[]
    now=datetime.now(ZoneInfo("America/New_York"))
    for symbol in dict.fromkeys(s.strip().upper() for s in symbols if s.strip()):
        try:
            bars=history(symbol)
            if bars.index[-1].date()==now.date() and (now.hour<16 or (now.hour==16 and now.minute<15)):
                bars=bars.iloc[:-1]
            if len(bars)<65: raise ValueError("Insufficient completed daily bars")
            spy_aligned=spy.loc[spy.index<=bars.index[-1]]
            signal_df=strategy_v2(bars, spy_aligned)
            last=signal_df.iloc[-1]
            record={"symbol":symbol,"bar_date":str(signal_df.index[-1].date()),"signal":str(last["action"]),"close":round(float(last["Close"]),2)}
            results.append(record)
            key="|".join([symbol,record["bar_date"],record["signal"]])
            if send and record["signal"]!="HOLD" and not state.get(key):
                msg="Fundslife V2: "+symbol+" "+record["signal"]+" on "+record["bar_date"]+". Research alert only."
                if dispatch_telegram(msg):
                    state[key]=True
                    state_path.parent.mkdir(parents=True,exist_ok=True)
                    state_path.write_text(json.dumps(state),encoding="utf-8")
        except Exception as error:
            results.append({"symbol":symbol,"error":str(error)})
    return results

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--symbols",default="SPY,QQQ,NVDA,AMD,AAPL")
    parser.add_argument("--telegram",action="store_true")
    args=parser.parse_args()
    print(json.dumps(run_scan(args.symbols.split(","),send=args.telegram),indent=2))
