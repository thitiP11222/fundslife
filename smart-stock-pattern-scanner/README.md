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
- No profitability or win-rate claims; no strategy backtest; no brokerage integration or live auto-trading.
- Data provider reliability and intraday candle timing require additional production safeguards.

## One-day roadmap
1. Install dependencies and run tests.
2. Inspect NVDA, AAPL and SPY daily data; verify signal levels visually.
3. Add a replay/backtest with strict next-bar fill logic as a separate follow-up.
4. Deploy on Streamlit Community Cloud only after validating data-provider limitations.

**Research/demo only. Not investment advice.**
