# Fundslife Price Action Scanner V4

Standalone Streamlit prototype with delayed-confirmation swing pivots, support and resistance, HH/HL/LH/LL structure, simple double top/bottom **candidates**, breakout/breakdown confirmation, SPY regime, and next-open directional OHLC backtest.

## Windows quickstart
```powershell
git fetch origin
git switch feat/price-action-pattern-scanner-v4
git pull origin feat/price-action-pattern-scanner-v4
cd trading-bot/price-action-v4
py -m pip install -r requirements.txt
py -m pytest -q
py -m streamlit run app.py
```
If already in the folder, skip cd.

The app intentionally does NOT issue automated orders, choose actual options contracts, or claim its underlying proxy Win Rate is an option Win Rate. Full-premium risk with THB 1,000 and 1% max means most US option contracts are financially infeasible. This scanner is a rule-based research baseline, not trained AI/PPO, and pattern candidates are heuristics. Confirm options contract metrics in Dime.

## Backtest constraints
Closed daily candles only; pivots are recorded at confirmation time (after `right` bars), not retroactively available on earlier dates. Breakout uses previously confirmed support/resistance, and trades enter next session open. Both target/stop intraday touched = stop first. Historical Yahoo OHLCV may be adjusted, revised or delayed; no historical option premium, IV, Greeks or precise fill data. The 70/30 display is descriptive, not a pristine holdout if manually tuned. Backtests are not proof of profitability.
