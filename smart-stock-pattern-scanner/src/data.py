"""Historical market data adapter. Never assume quote data is live."""
import re
import pandas as pd
import yfinance as yf

def get_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    ticker = ticker.upper().strip()
    if not re.fullmatch(r"[A-Z0-9.\-^=]{1,20}", ticker):
        raise ValueError("Invalid ticker symbol")
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
    if data.empty:
        raise ValueError(f"No historical data returned for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not set(required).issubset(data.columns):
        raise ValueError("Data provider missing required OHLCV columns")
    data = data[required].copy().apply(pd.to_numeric, errors="coerce").dropna(subset=["Open", "High", "Low", "Close"])
    data = data.loc[~data.index.duplicated(keep="last")].sort_index()
    if len(data) < 65:
        raise ValueError("Not enough price history: at least 65 bars required")
    return data
