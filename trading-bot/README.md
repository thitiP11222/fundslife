# Fundslife Trading Scanner

Research and decision-support toolkit for liquid US stocks. The current production flow is:

```text
Yahoo Finance OHLCV
        |
        v
Unified Momentum Scanner
        |
        v
Top momentum candidates
        |
        v
Streamlit Technical Dashboard
        |
        v
Support / Resistance / Pattern / ATR
        |
        v
Risk-based Position Sizing
        |
        v
Manual trade decision
```

The project does **not** place live orders and does not guarantee profit.

## Current files

```text
trading-bot/
├── streamlit_app.py
├── run_unified_momentum_scanner.py
├── requirements.txt
├── pytest.ini
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── indicators.py
│   ├── unified_momentum_scanner.py
│   ├── technical_dashboard.py
│   └── risk.py
└── tests/
    ├── test_indicators.py
    ├── test_risk.py
    ├── test_unified_momentum_scanner.py
    └── test_technical_dashboard.py
```

Legacy breakout, pullback, option-entry and rotation research code has been removed from the active branch after validation showed it was not suitable for the current trading workflow.

## 1. Unified Momentum Scanner

Run:

```powershell
python run_unified_momentum_scanner.py
```

The scanner ranks a liquid US-stock universe using a medium-term momentum model:

```text
Momentum Score
= 0.60 × 63-day return
+ 0.40 × 126-day return
```

Additional filters:

- Market regime: QQQ must be above its 200-day moving average.
- Stock trend: Close > EMA50 > EMA200.
- Liquidity: price >= USD 5 and 20-day average dollar volume >= USD 20M.
- 52-week-high proximity is used only as a diagnostic / options confirmation flag.

Outputs:

- `BUY_CANDIDATE`: Top-3 fractional-stock candidate.
- `WATCH`: Positive trend/momentum but outside Top-3.
- `CHECK_DR_OR_DRX`: Underlying qualifies; verify a listed DR/DRx and its spread/liquidity.
- `REVIEW_CALL_CHAIN`: Very selective options research flag; not an automatic buy signal.

The scanner saves:

```text
results/unified_momentum_candidates.csv
```

## 2. Streamlit Technical Dashboard

Run:

```powershell
streamlit run streamlit_app.py
```

The dashboard accepts a ticker and timeframe:

- 15m
- 30m
- 1h
- 1d

It displays:

- Candlestick chart
- EMA20
- EMA50
- RSI14
- ATR14
- volume relative to 20-bar average
- automatic support levels
- automatic resistance levels
- pullback-entry reference
- breakout-entry reference
- invalidation / stop reference
- Target 1
- Target 2
- chart-pattern detections
- technical setup score
- fractional-share position sizing

Yahoo Finance intraday data should be treated as near-real-time / potentially delayed, not exchange-grade real-time market data.

## 3. Support and Resistance

Support/resistance uses local swing structure rather than a single rolling high/low.

For each bar, the engine searches for local:

- Swing High
- Swing Low

Nearby levels are then clustered using an ATR-based tolerance so several nearby pivots become one price zone.

The dashboard returns the nearest three support and resistance levels around the current price.

This is a market-structure heuristic. It is not a statistically validated standalone entry signal.

## 4. Trend Analysis

The dashboard uses:

```text
Bullish structure:
Close > EMA20 > EMA50

Bearish structure:
Close < EMA20 < EMA50
```

EMA20 represents shorter-term trend while EMA50 represents the broader local trend.

The momentum scanner uses the slower:

```text
Close > EMA50 > EMA200
```

because stock selection and entry timing operate on different horizons.

## 5. RSI

RSI14 is used as a momentum-condition filter.

The dashboard currently considers approximately:

```text
RSI 50-68
```

a constructive bullish momentum zone.

RSI >= 75 is treated as an extension warning.

RSI is not used as a standalone buy/sell trigger.

## 6. ATR and Volatility

ATR14 measures recent price range / volatility.

ATR is used for:

- support/resistance clustering tolerance
- stop / invalidation distance
- breakout buffer
- risk-normalized trade planning

This avoids applying the same fixed percentage stop to low-volatility and high-volatility stocks.

## 7. Volume Confirmation

The dashboard calculates:

```text
Volume Ratio
= Current Volume / 20-bar Average Volume
```

A value >= 1.0 is treated as participation confirmation.

For intraday use, signals should be interpreted after the candle is closed because an unfinished candle naturally has incomplete volume.

## 8. Chart Patterns

The current engine detects:

- Bullish Engulfing
- Bearish Engulfing
- Hammer / Pin Bar
- Shooting Star / Bearish Pin
- 20-Bar Breakout
- Possible Double Top
- Possible Double Bottom

Patterns are confirmation features only.

They are combined with trend, momentum, volume and nearby support/resistance rather than being treated as automatic trades.

## 9. Technical Setup Score

The dashboard creates a heuristic long score from:

- bullish EMA structure
- RSI momentum zone
- volume confirmation
- bullish pattern confirmation
- room before nearby resistance

The output is:

```text
75-100  BUY_SETUP
55-74   WATCH
<55     AVOID
```

Important: this score is a technical-analysis heuristic. It has not been validated as an independent profitable strategy.

The medium-term momentum-selection model is the stronger research-backed component of the current system.

## 10. Pullback and Breakout References

Two possible technical entry references are displayed.

### Pullback

Uses nearby support / EMA20 as a reference for entering after price retraces instead of chasing strength.

### Breakout

Uses the nearest resistance plus a small ATR buffer as the confirmation reference.

These are planning levels, not guaranteed execution prices.

The previous dedicated 1-hour breakout strategy was removed because its historical expectancy was negative. The dashboard levels are therefore used as visual decision support, not as an automated strategy.

## 11. Stop and Targets

The stop reference is built from price structure and ATR.

Targets are expressed in R multiples:

```text
Risk = Entry - Stop

Target 1 = Entry + 1R
Target 2 = Entry + 2R
```

The user should still account for gaps, slippage and market news.

## 12. Position Sizing

The dashboard includes risk-based fractional-share sizing.

Inputs:

- account equity
- risk per trade %
- maximum position allocation %

Core calculation:

```text
Risk Budget
= Account Equity × Risk %

Shares by Risk
= Risk Budget / |Entry - Stop|

Shares by Allocation
= Max Position Value / Entry

Final Shares
= min(Shares by Risk, Shares by Allocation)
```

Fractional sizing currently rounds down to a 0.001-share step.

This is designed to prevent a volatile stock from receiving the same dollar allocation as a lower-risk setup.

## Research findings retained from previous versions

Several strategy families were tested before the current architecture was selected.

### Intraday breakout / pullback

These strategies did not show reliable positive expectancy across the tested universe and were removed from the active code.

### Grade-A 1H options entry

After correcting target calculations:

```text
Signals            18
T1 hit rate         33.33%
T2 hit rate         11.11%
Invalidation rate   77.78%
Avg return          -1.66%
Avg R               -0.515R
```

This entry model was rejected.

### Medium-term momentum

The non-overlapping 20-trading-day robustness study was materially stronger.

Top-3 aggregate:

```text
Events                 315
Average return          +3.57%
Median return           +2.38%
Win rate                59.05%
Average excess vs QQQ   +1.42%
Median excess vs QQQ    +0.05%
Beat QQQ rate           51.11%
```

This is why the active scanner uses medium-term momentum for stock selection and does not rely on the rejected 1H breakout system.

These historical results are research observations, not guarantees of future performance.

## Setup

```powershell
cd trading-bot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run tests:

```powershell
pytest -q
```

Run the stock scanner:

```powershell
python run_unified_momentum_scanner.py
```

Run the dashboard:

```powershell
streamlit run streamlit_app.py
```

## Intended workflow

```text
1. Run momentum scanner
2. Select Top-3 candidates
3. Open candidate in Streamlit
4. Review trend and market structure
5. Check support/resistance and patterns
6. Define entry, invalidation and targets
7. Calculate position size from account risk
8. Check spread / news / liquidity
9. Make the trade decision manually
10. Record and review the outcome
```

## Limitations

- Yahoo Finance is not guaranteed exchange-grade real-time data.
- Support/resistance and pattern recognition are heuristic.
- No broker execution is connected.
- DR/DRx mapping and local-market liquidity are not yet automated.
- News / earnings-event filters are not yet integrated.
- Position sizing does not include FX conversion or broker-specific minimum fees.
- Historical momentum evidence can weaken or disappear in future regimes.


