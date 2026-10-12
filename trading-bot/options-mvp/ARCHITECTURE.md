# Architecture decision

Keep Streamlit for charts and interactive quant research, but execute background scanning in a separate CLI.

- `options_core/`: independent rules, data access, position risk, underlying proxy tests.
- `app.py`: interactive read-only Dashboard with 15-minute data cache.
- `scan_cli.py`: standalone post-session signals, optional Telegram, local deduplication file.
- Future: persistent SQLite signal journal, deterministic OHLC bar snapshots, causal intrabar exit simulation, walk-forward cross-market evaluation and genuine historical options P&L.

Run from `options-mvp`:

```powershell
python -m pytest -q
python scan_cli.py --symbols SPY,QQQ,NVDA
# Set Telegram credentials in your shell; then
python scan_cli.py --telegram --symbols SPY,QQQ,NVDA
streamlit run app.py
```

Schedule scan_cli with Windows Task Scheduler after US market close (taking daylight savings into account). Never schedule through a Streamlit page session.

**Limitations:** Yahoo prices and chains can be delayed or unavailable; latest bar must be confirmed; spot/underlying-direction backtests do not equal options returns. One long US equity option could lose the entire premium. Strict 1% full-premium risk with THB 1,000 will generally forbid trades. No PPO model is trained. Local JSON deduplication is not concurrency-safe; use SQLite before multiple runners.
