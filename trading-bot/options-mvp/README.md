# Fundslife AI Options Swing MVP

Experimental **underlying-direction scanner** for independently reviewing US stock options in Dime. No automated orders. Not financial advice.

## Windows quickstart

```powershell
git fetch origin
git switch feat/fundslife-ai-options-swing-mvp
cd trading-bot/options-mvp
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

Optional Telegram manual alerts (PowerShell):

```powershell
$env:TELEGRAM_BOT_TOKEN="..."
$env:TELEGRAM_CHAT_ID="..."
streamlit run app.py
```

## Features
- Daily EMA20/50, RSI14, ATR14, 20-bar breakout/breakdown, volume filter, CALL/PUT/HOLD **underlying bias**.
- Next-open, five-session holding-period **underlying proxy** backtest with configurable two-sided costs.
- Yahoo Finance option-chain display (may be delayed / inaccurate).
- Contract risk gate using worst-case full option premium, manually entered FX, commission and fees.
- Optional on-demand Telegram notifications when credentials are set.

## Important limitations
1. Backtest is **not** options P/L: it does not reconstruct historical chains, implied volatility, theta, gamma, options spreads, early exercise/assignment or broker fills.
2. Daily downloaded Yahoo data may contain a partially formed current session bar. For trading decisions, independently verify the last completed US trading session.
3. No historical survivorship-free universe, walk-forward model training, or PPO training is included.
4. Risk gate is conservative: whole premium may be lost; a stop-loss does not assure execution. Calculations do not include every changing broker-specific cost or FX spread.
5. Telegram alert is manually triggered in Streamlit. To automate scheduled scans, deploy a runner/scheduler outside Streamlit.
6. Long single-leg options only; do not use for short option selling or multi-leg assignment risk. Check actual contract size, account permissions, product specification and expiry in Dime.
7. Public Yahoo Finance data is not broker-grade or guaranteed real-time.

## Research inspiration
Wang & Wang, 2024, *An adaptive financial trading strategy based on proximal policy optimization and financial signal representation* (https://doi.org/10.1016/j.engappai.2024.109365). Original work combines CEEMDAN, MRS and PPO; **this MVP only implements a rules-based baseline**, not paper replication. Proposed phase 2: causal denoising, fit normalizer only on training folds, trading-penalized reward, PPO vs rule baseline evaluated by rolling out-of-sample windows and option-contract-specific historical prices.

## Design safety
With THB 1,000 capital and a 1% risk rule, maximum modeled loss is THB 10. Standard US options usually cover 100 shares, so most contracts will be rejected, correctly. Consider paper trading before depositing or sizing up.
