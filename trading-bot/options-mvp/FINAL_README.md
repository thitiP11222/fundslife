# Fundslife Options Scanner — release candidate

This is a **research-only**, single-page Streamlit terminal. The strategy is rule based (not trained PPO). It cannot submit broker orders or guarantee wins.

## Install / run (Windows PowerShell)

```powershell
git fetch origin
git switch release/options-scanner-v1
git pull origin release/options-scanner-v1
cd trading-bot/options-mvp
py -m pip install -r requirements.txt
python -m pytest -q
streamlit run final_app.py
```

If already in options-mvp, skip cd. Or run `./RUN_FINAL.ps1` after installation.

Features: EMA20/50, RSI14, ATR, SPY regime, latest daily candle check, long-option contract eligibility by full premium risk, 70/30 descriptive split, ATR stop and R target, user-controlled settings, CSV export. Option chain is provider research data only.

**Not included:** true historical options P/L (requires licensed/suitable historical chain snapshots), verified Dime fee schedule, unattended trading, trained PPO, scientific proof of win rate advantage. Test results require local run. The 1% risk threshold of a THB 1,000 portfolio is THB 10; most single-leg US options must be rejected. Verify OHLC freshness and complete broker costs before decisions.
