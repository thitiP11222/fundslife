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


## Validation workflow (V2)

After the normal backtest passes, run:

```powershell
python run_validation.py
```

This performs three checks:

### 1. Buy & Hold benchmark

The strategy is compared with a simple close-to-close NVDA benchmark. A trading
strategy should not be judged only by whether it made money; compare it with
what happened if the same market exposure had simply been held.

### 2. Rolling out-of-sample validation

The data is divided into sequential train/test windows. Strategy metrics are
measured only on later test windows that were not used as the preceding
training period.

The current implementation keeps parameters fixed. It is therefore a rolling
out-of-sample robustness check rather than an automatic parameter optimizer.

Output:

```text
results/walk_forward_NVDA_15m.csv
```

Important fields:

- strategy_return_pct
- buy_hold_return_pct
- trades
- win_rate_pct
- profit_factor
- expectancy

A more credible strategy should work across several independent windows, not
only one historical period.

### 3. Parameter sensitivity

The validator evaluates nearby values for:

- breakout period: 15 / 20 / 25
- ATR stop multiplier: 1.0 / 1.5 / 2.0
- risk-reward ratio: 1.5 / 2.0 / 2.5

Output:

```text
results/parameter_sensitivity_NVDA_15m.csv
```

Do not simply choose the row with the highest historical return. Look for a
stable region where several nearby parameter combinations produce reasonable
results. One isolated best combination is a warning sign for overfitting.

## Recommended decision gate before paper trading

Do not proceed to broker execution just because one backtest is profitable.
At minimum review:

1. enough total trades to make metrics meaningful
2. positive expectancy across multiple out-of-sample folds
3. drawdown acceptable for the intended account
4. results not dependent on one parameter combination
5. comparison against Buy & Hold
6. performance across more than one liquid stock and market regime


## Small-account profile: THB 1,000

The default configuration is now designed for a very small US-stock account.

Key changes:

- timeframe: 1 hour instead of 15 minutes
- breakout: 25 bars
- ATR stop: 2.0 ATR
- reward target: 2R
- risk per trade: 0.5% of account equity
- maximum capital in one position: 90%
- fractional shares enabled
- fractional step: 0.001 share
- backtest cash: USD 30 as an approximate USD-equivalent research account

Important: Yahoo Finance prices US stocks in USD, so the backtest account is also
in USD. If your real budget is THB 1,000, convert the budget using the actual FX
rate and broker conversion costs before comparing real-money results.

A broker must support fractional shares for expensive stocks such as NVDA.
Without fractional shares, a THB 1,000 account may not be able to buy even one
whole share.

The 0.5% risk setting is deliberately conservative. With a very small account,
minimum commissions, FX conversion fees, spread, or broker minimum order sizes
can dominate results. Update the cost assumptions to match the broker before
paper trading.

Run:

```powershell
git pull origin feat/stock-trading-bot-v1
pytest -q
python run_backtest.py
python run_validation.py
```
