# Fundslife Stock Trading Bot v1

Educational/research trading bot for liquid stocks. It does not guarantee profit and does not place live orders.

## Strategy

The baseline long setup combines:

- EMA 20 > EMA 50: trend filter
- Close > previous 20-bar high: breakout confirmation
- RSI 14 between 50 and 70: momentum filter
- Volume >= 20-bar average: participation filter
- Stop loss = 1.5 ATR
- Take profit = 2R
- Risk per trade = 1% of equity
- Maximum position value = 25% of equity

Signals are generated on candle close and backtests enter on the next candle open to reduce look-ahead bias. Commission and slippage assumptions are included.

## Setup

From the repository root:

```powershell
cd trading-bot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run tests

```powershell
pytest -q
```

## Get the latest signal

```powershell
python run_signal.py
```

Default market configuration is NVDA on 15-minute candles. Edit `config.py` to change the symbol or parameters.

## Run backtest

```powershell
python run_backtest.py
```

Outputs:

```text
results/backtest_NVDA_15m.csv
results/metrics_NVDA_15m.json
```

Review these metrics together:

- total return
- number of trades
- win rate
- profit factor
- expectancy per trade
- maximum drawdown

Do not optimize for win rate alone.

## Architecture

```text
Yahoo Finance OHLCV
        |
        v
Technical Features
EMA / RSI / ATR / Breakout / Volume
        |
        v
Strategy Engine
BUY / NO_TRADE
        |
        v
Risk Engine
Position size / Stop / Target
        |
        v
Backtester
Costs / Slippage / Metrics
```

## Important limitations

This is v1 research infrastructure, not a production execution engine.

Before any real-money automation, add and validate:

1. walk-forward / out-of-sample testing
2. benchmark comparison such as buy-and-hold
3. survivorship-bias-aware historical data
4. market-session filters
5. stale-data handling
6. daily loss limits
7. maximum concurrent exposure
8. broker-specific execution and order validation
9. paper trading
10. monitoring and a kill switch

Parameters should be evaluated for robustness across symbols and periods rather than selected only for the best historical result.
