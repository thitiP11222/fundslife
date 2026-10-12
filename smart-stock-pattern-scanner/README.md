# Fundslife Smart Stock Pattern Scanner

A Python + Streamlit technical-pattern research dashboard with an **embedded TradingView chart**.

## Features
- US equities/ETFs historical OHLCV download via yfinance
- Confirmed 20-bar breakouts/breakdowns with volume filtering
- Double Bottom / Double Top with swing-pivot confirmation and neckline crossing
- EMA20/50, RSI14, ATR14, prior 20-bar extremes
- Proposed entry, protective stop, 2R take profit, heuristic signal score
- Risk-budget position sizing for *theoretical* fractional-share trading
- TradingView Advanced Chart for exploring and Plotly for precisely overlaying Python levels
- Deterministic unit tests using synthetic historical data

## Quick start (Windows PowerShell)

```powershell
git clone https://github.com/thitiP11222/fundslife.git
cd fundslife
git checkout feat/smart-stock-pattern-scanner
cd smart-stock-pattern-scanner
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

Default Streamlit URL: http://localhost:8501

## Architecture

```text
app.py                 UI, TradingView widget, Plotly overlay
src/data.py            historical OHLCV validation & download
src/patterns.py        causal indicators, patterns, risk sizing
tests/test_patterns.py deterministic tests
```

## TradingView integration & limitations

The **free TradingView Advanced Chart widget** is used only for visual chart exploration. TradingView does *not* act as our indicator datafeed, trading execution venue or Python backtest engine. Python-computed levels appear on a separate Plotly chart, *not overlaid on the TradingView widget*. Data feeds may differ (including exchange selection, adjusted dividends/splits and delay). Widgets require an internet connection and may be subject to TradingView policies.

Documentation: https://www.tradingview.com/widget-docs/widgets/charts/advanced-chart/

Advanced Charts Library (different from free widget) requires permission and must not be committed to public repositories.

## Methodology & limitations

- Scanner evaluates the most recent completed **input** candle. For 1-hour data the app conservatively discards the most recent candle; on daily data, run after market close to avoid treating a forming daily candle as completed.
- Breakout threshold is **prior** 20 highs/lows, excluding current candle; volume must exceed previous 20-bar average by 10%.
- Confirmed pivot uses 3 bars on each side. The second trough/peak must be confirmed. A Double Bottom/Top signal requires crossing the neckline on the latest bar, with near-equal pivot levels.
- Entry/stop/target are scenario levels, not fill promises. Actual orders could have gaps, fees and slippage. Shorts may not be available in fractional share accounts.
- Indicator prices use auto-adjusted Yahoo data and can differ from raw trading prices.
- A heuristic score **does not** estimate success probability.
- No profitability claims, brokerage integration or live auto-trading. The historical backtest uses conservative assumptions, not guaranteed fills.
- Data provider reliability and intraday candle timing require additional production safeguards.

## One-day roadmap
1. Install dependencies and run tests.
2. Inspect NVDA, AAPL and SPY daily data; verify signal levels visually.
3. Run Phase 2 historical signal replay and backtest with next-bar open execution.
4. Deploy on Streamlit Community Cloud only after validating data-provider limitations.

**Research/demo only. Not investment advice.**

## Phase 2: Backtest and walk-forward analysis

- Click **Run historical analysis** on the Dashboard. Pattern history is reconstructed from causal prefixes of completed bars.
- Long-only signals are entered at the **next bar open** with configurable fees/slippage and capped fractional sizing; one position at a time.
- The stop and target are checked against the current candle range; stop gets priority if both would be touched. Gaps execute at open where appropriate. Positions are liquidated at evaluation-period end.
- Walk-forward is an **anchored chronological series of nonoverlapping test windows using frozen parameters**, not parameter optimization. Initial bars provide indicator context only. Each fold resets cash rather than compounding.
- Daily/1-hour Sharpe annualization uses approximations (252 and 1,638 bars per year). Historical adjusted data may be incompatible with true execution and may be incomplete.
- Result tables and CSV exports: trade log, historical signals, per-fold evaluation metrics.
- This is an exploratory retrospective simulation subject to selection bias, stock survivorship bias, fees, slippage, and intrabar fill uncertainty.

Run `python -m pytest -q` before deployment.
